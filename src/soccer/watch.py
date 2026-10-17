"""Evaluate a checkpoint against random or its other trained team, optionally rendering."""

import argparse
from pathlib import Path
import sys

# Support both python src/soccer/watch.py and python -m soccer.watch.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from algorithms.mappo.checkpoints import load_checkpoint
from algorithms.mappo.device import resolve_device, seed_everything
from algorithms.mappo.networks import Actor, actions_to_env
from soccer.config import SoccerEnvConfig
from soccer.encoding import encode_team
from soccer.env import SoccerEnv


def watch(checkpoint_path, matches=3, device='auto', headless=False, team=0, self_play=False, seed=0):
    if matches < 1 or team not in (0, 1):
        raise ValueError('Choose a positive match count and team 0 or 1')
    device = resolve_device(device)
    seed_everything(seed)
    checkpoint = load_checkpoint(checkpoint_path, device)
    selected = (0, 1) if self_play else (team,)
    if not set(selected) <= set(checkpoint['training_teams']):
        raise ValueError('Requested team was not trained in this checkpoint')
    env = SoccerEnv(SoccerEnvConfig(**checkpoint['env_config']), render_mode=None if headless else 'human')
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
                          team_rewards=env.last_team_rewards.copy(), transitions=env.step_count)
            results.append(result)
            print(result, flush=True)
        return results
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--matches', type=int, default=3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--team', type=int, choices=(0, 1), default=0)
    parser.add_argument('--self-play', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    watch(args.checkpoint, args.matches, args.device, args.headless, args.team, args.self_play, args.seed)


if __name__ == '__main__':
    main()
