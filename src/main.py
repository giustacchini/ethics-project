"""Render a random soccer match: python src/main.py [--seed 0]."""

import argparse

from soccer.env import SoccerEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    env = SoccerEnv(render_mode='human')
    try:
        env.reset(seed=args.seed)
        action_space = env.action_space
        action_space.seed(args.seed + 10000)
        while not env.match_over:
            env.step(action_space.sample())
            # print(env.last_goal)
            # print(env.age)
            # print(f'Step {env.step_count}: {env.last_step_info}')
        print(f'Match finished after {env.step_count} steps: {env.team_scores}, {env.match_result}')
        print('Team rewards:', env.last_team_rewards)
    finally:
        env.close()


if __name__ == '__main__':
    main()
