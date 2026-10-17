"""Render a random soccer match: python src/main.py [--seed 0]."""

import argparse

from soccer.env import SoccerEnv
from soccer.statistics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--event-log', help='Save the full trace to a new JSONL file')
    parser.add_argument('--show-events', action='store_true', help='Print match events live')
    parser.add_argument('--render-fps', type=int, default=20, dest="render_fps")
    args = parser.parse_args()
    env = SoccerEnv(render_mode='human', event_log_path=args.event_log, show_events=args.show_events, render_fps=args.render_fps)
    try:
        env.reset(seed=args.seed)
        action_space = env.action_space
        action_space.seed(args.seed + 10000)
        while not env.match_over:
            env.step(action_space.sample())
        print(f'Match finished after {env.step_count} steps: {env.team_scores}, {env.match_result}')
        print('Team rewards:', env.last_team_rewards)
        print('Statistics:', summarize(env.match_log.records)['teams'])
    finally:
        env.close()


if __name__ == '__main__':
    main()
