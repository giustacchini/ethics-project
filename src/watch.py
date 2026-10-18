"""Evaluate a checkpoint against random or its other trained team, optionally rendering."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from algorithms.mappo.checkpoints import load_checkpoint
from algorithms.mappo.device import resolve_device, seed_everything
from algorithms.mappo.networks import Actor, actions_to_env
from soccer.config import SoccerEnvConfig
from soccer.observation.encoding import encode_team
from soccer.env import SoccerEnv
from soccer.evaluation.statistics import summarize, evaluation_report


def watch(checkpoint_path, matches=3, device='auto', headless=False, team=0, self_play=False, seed=0, event_log_path=None, show_events=False, render_fps=20, output_root='evaluations'):
    if matches < 1 or team not in (0, 1):
        raise ValueError('Choose a positive match count and team 0 or 1')
    device = resolve_device(device)
    seed_everything(seed)
    checkpoint = load_checkpoint(checkpoint_path, device)
    selected = (0, 1) if self_play else (team,)
    if not set(selected) <= set(checkpoint['training_teams']):
        raise ValueError('Requested team was not trained in this checkpoint')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
    suffix = 0
    while True:
        directory = root / (timestamp if suffix == 0 else f'{timestamp}_{suffix}')
        try:
            directory.mkdir()
            break
        except FileExistsError:
            suffix += 1
    trace_path = directory / 'events.jsonl' if event_log_path is None else Path(event_log_path)
    metadata = dict(checkpoint=str(Path(checkpoint_path).resolve()), checkpoint_update=checkpoint['update'],
                    training_seed=checkpoint['seed'], evaluation_seed=seed, requested_matches=matches,
                    selected_teams=list(selected), opponent='self_play' if self_play else 'random',
                    device=str(device), env_config=checkpoint['env_config'], event_log=str(trace_path.resolve()))
    (directory / 'evaluation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Evaluation output: {directory}', flush=True)
    env = SoccerEnv(SoccerEnvConfig(**checkpoint['env_config']), render_mode=None if headless else 'human',
                    event_log_path=trace_path, show_events=show_events, render_fps=render_fps)
    actors = {}
    spec = checkpoint['network_config']
    for current in selected:
        actor = Actor(spec['obs_size'], spec['n_players'], spec['hidden']).to(device)
        actor.load_state_dict(checkpoint['teams'][current]['actor'])
        actors[current] = actor.eval()
    results = []
    try:
        for match in range(matches):
            observations, _ = env.reset(seed=seed + match)
            random_actions = env.action_space
            random_actions.seed(seed + match + 10000)
            while not env.match_over:
                actions = random_actions.sample()
                for current, actor in actors.items():
                    ids, obs, _, active = encode_team(observations, current, env.width, env.height, 3, env.max_steps)
                    with torch.no_grad():
                        chosen = actor(torch.as_tensor(obs, device=device), torch.as_tensor(active, device=device)).sample()
                    actions.update(actions_to_env(chosen, ids))
                observations, *_ = env.step(actions)
            result = dict(match=match + 1, score=env.team_scores.copy(), result=env.match_result,
                          team_rewards=env.last_team_rewards.copy(), transitions=env.step_count, statistics=summarize(env.match_log.records)['teams'])
            results.append(result)

        return results
    finally:
        env.close()
        with trace_path.open() as stream:
            statistics = evaluation_report(summarize(json.loads(line) for line in stream if line.strip()))
        (directory / 'statistics.json').write_text(json.dumps(statistics, indent=2, allow_nan=False) + '\n')
        print(f'Statistics: {directory / "statistics.json"} ({statistics["completed_matches"]} completed matches)', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--matches', type=int, default=3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--team', type=int, choices=(0, 1), default=0)
    parser.add_argument('--self-play', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--event-log', help='Save the full trace to a new JSONL file')
    parser.add_argument('--show-events', action='store_true', help='Print match events live')
    parser.add_argument('--render-fps', type=int, default=20, dest="render_fps")
    parser.add_argument('--output-root', default='evaluations', help='Parent folder for automatically named evaluation runs')
    args = parser.parse_args(argv)
    watch(args.checkpoint, args.matches, args.device, args.headless, args.team, args.self_play, args.seed, args.event_log, args.show_events, args.render_fps, args.output_root)


if __name__ == '__main__':
    main()
