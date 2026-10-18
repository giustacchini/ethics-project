"""Headless match validation and seeded trace reproducibility checks."""

import argparse
from dataclasses import asdict
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np

from soccer.entities.ball import CustomBall
from soccer.config import SoccerEnvConfig
from soccer.env import SoccerEnv
from soccer.evaluation.statistics import summarize


def canonical(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    return value


def fingerprint(value):
    data = json.dumps(canonical(value), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def check_state(env, observations):
    env.check_goalkeeper_invariants()
    positions = [tuple(a.state.pos) for a in env.active_players]
    assert len(positions) == len(set(positions)), 'Active players overlap'
    assert all(0 <= x < env.width and 0 <= y < env.height for x, y in positions), 'Player off pitch'
    assert env.observation_space.contains(observations), 'Invalid observation'
    ball = env.ball
    assert ball is not None, 'Default match has no ball'
    assert all(0 <= x < env.width and 0 <= y < env.height
               for x, y in [ball.position, *ball.remaining_trajectory]), 'Ball off pitch'
    carrying = [a.index for a in env.agents if a.state.carrying is not None]
    if ball.carrier_id is not None:
        carrier = env.agents[ball.carrier_id]
        assert not carrier.suspended, 'Suspended carrier'
        assert tuple(carrier.state.pos) == ball.position, 'Ball/carrier mismatch'
        assert carrying == [carrier.index], 'Carrying overlay mismatch'
        assert ball.last_possessing_team == carrier.team_index, 'Possession team mismatch'
        assert not ball.remaining_trajectory and ball.speed == 0, 'Carried ball in flight'
    else:
        assert not carrying, 'Unpossessed ball has carrying overlay'
        valid_speed = ball.speed in (1, 2) if ball.remaining_trajectory else ball.speed == 0
        assert valid_speed, 'Invalid flight speed'
    markers = [(x, y) for x in range(env.width) for y in range(env.height)
               if isinstance(env.grid.get(x, y), CustomBall)]
    assert markers == ([] if ball.carrier_id is not None else [ball.position]), 'Grid ball marker mismatch'
    if env.free_kick_taker is not None:
        assert ball.carrier_id == env.free_kick_taker, 'Free-kick possession mismatch'


def run_match(team_size, seed, max_steps=1000, event_log_path=None, show_events=False, render_fps=20):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=team_size, max_steps=max_steps),
                    event_log_path=event_log_path, show_events=show_events, render_fps=render_fps)
    hashes = []
    try:
        observations, infos = env.reset(seed=seed)
        action_space = env.action_space
        action_space.seed(seed + 10000)
        check_state(env, observations)
        hashes.append(fingerprint(dict(observations=observations, infos=infos)))
        while not env.match_over:
            actions = action_space.sample()
            observations, rewards, terms, truncs, infos = env.step(actions)
            check_state(env, observations)
            reached_target = max(env.team_scores.values()) >= 3
            assert all(terms.values()) == (env.match_over and reached_target), 'Incorrect termination'
            assert all(truncs.values()) == (env.match_over and not reached_target), 'Incorrect truncation'
            for agent in env.agents:
                expected = (1000.0 if agent.team_index == env.winner else -1000.0) if env.match_over else 0.0
                assert rewards[agent.index] == expected, 'Incorrect agent reward'
                assert env.last_reward_result.team_total(agent.team_index) == expected, 'Incorrect team reward'
            assert len(env.reward_events.events()) == (2 if env.match_over else 0), 'Repeated outcome event'
            if env.last_goal and not env.match_over:
                assert all(not a.suspended and a.current_role == a.original_role for a in env.agents)
                assert env.ball.carrier_id is None
            hashes.append(fingerprint(dict(actions=actions, observations=observations, rewards=rewards,
                terms=terms, truncs=truncs, infos=infos, movement=env.last_movement_results,
                events=[asdict(event) for event in env.reward_events.events()])))
        assert env.step_count <= max_steps
        expected_winner = None if env.team_scores[0] == env.team_scores[1] else max(env.team_scores, key=env.team_scores.get)
        assert env.winner == expected_winner, 'Incorrect winner'
        assert max(env.team_scores.values()) == 3 or env.step_count == max_steps
        return dict(team_size=team_size, seed=seed, transitions=env.step_count,
                    score=env.team_scores.copy(), result=env.match_result,
                    trace_sha256=fingerprint(hashes), statistics=summarize(env.match_log.records)['teams']), hashes
    except Exception as error:
        raise RuntimeError(f'team_size={team_size}, seed={seed}, transition={env.step_count}: {error}') from error
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--event-log-dir', type=Path, help='Save one full trace per validated match')
    parser.add_argument('--show-events', action='store_true', help='Print events from the first execution of each match')
    parser.add_argument('--render-fps', type=int, default=20, dest="render_fps")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.max_steps < 1:
        parser.error('seeds and max-steps must be positive')
    summaries = []
    for size in (2, 3):
        for seed in range(args.seeds):
            log_path = None if args.event_log_dir is None else args.event_log_dir / f'{size}v{size}_seed{seed}.jsonl'
            summary, trace = run_match(size, seed, args.max_steps, log_path, args.show_events, render_fps=args.render_fps)
            _, repeated = run_match(size, seed, args.max_steps, render_fps=args.render_fps)
            if trace != repeated:
                mismatch = next((i for i, (a, b) in enumerate(zip(trace, repeated)) if a != b), min(len(trace), len(repeated)))
                raise AssertionError(f'Reproducibility failure: team_size={size}, seed={seed}, transition={mismatch}')
            summary['reproducible'] = True
            summaries.append(summary)
            print(f'{size}v{size} seed={seed}: {summary["transitions"]} transitions, reproducible', flush=True)
    report = dict(matches=len(summaries), executions=2 * len(summaries), summaries=summaries)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(canonical(report), indent=2) + '\n')


if __name__ == '__main__':
    main()
