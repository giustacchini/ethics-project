"""Train a soccer team against random, or both independent teams in self-play."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from algorithms.mappo.checkpoints import save_checkpoint
from algorithms.mappo.compute_gae import compute_advantages
from algorithms.mappo.device import resolve_device, seed_everything
from algorithms.mappo.experience_collector import ExperienceCollector
from algorithms.mappo.mappo_update import ppo_update
from algorithms.mappo.teams import create_teams
from soccer.config import SoccerEnvConfig
from soccer.encoding import encode_team
from soccer.env import SoccerEnv


@dataclass
class TrainingConfig:
    team_size: int = 3
    seed: int = 0
    updates: int = 10
    rollout_steps: int = 128
    max_steps: int = 1000
    hidden: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 0.001
    rewards: str = 'v1'
    epochs: int = 4
    minibatches: int = 4
    training_team: int = 0
    self_play: bool = False
    device: str = 'auto'
    output: str = 'checkpoints/mappo'


def train(config):
    if min(config.updates, config.rollout_steps, config.max_steps, config.epochs, config.minibatches, config.hidden) < 1:
        raise ValueError('Training counts and dimensions must be positive')
    if config.training_team not in (0, 1) or config.learning_rate <= 0:
        raise ValueError('Invalid training team or learning rate')
    if config.reward_scale <= 0:
        raise ValueError('Reward scale must be positive')
    if not 0 <= config.gamma <= 1 or not 0 <= config.gae_lambda <= 1:
        raise ValueError('Discount and GAE lambda must be in [0, 1]')
    device = resolve_device(config.device)
    seed_everything(config.seed)
    directory = Path(config.output)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / 'metrics.jsonl'
    if metrics_path.exists() or (directory / 'matches.jsonl').exists() or any(directory.glob('update_*.pt')):
        raise FileExistsError(f'Training output already contains a run: {directory}; choose a new output directory')
    env_config = SoccerEnvConfig(n_players_per_team=config.team_size, max_steps=config.max_steps, rewards=config.rewards, shaping_gamma=config.gamma)
    env = SoccerEnv(env_config, event_log_path=directory / 'matches.jsonl')
    try:
        observations, _ = env.reset(seed=config.seed)
        _, obs, state, _ = encode_team(observations, 0, max_steps=config.max_steps)
        network_config = dict(obs_size=obs.shape[-1], state_size=state.size,
                              n_players=config.team_size * 2, hidden=config.hidden)
        teams = create_teams(**network_config, learning_rate=config.learning_rate, device=device)
        training_teams = (0, 1) if config.self_play else (config.training_team,)
        collector = ExperienceCollector(env, teams, training_teams, config.seed)
        metrics = []
        print(f'Device: {device}; training teams: {training_teams}; {config.team_size}v{config.team_size}', flush=True)
        for update in range(1, config.updates + 1):
            buffers, bootstrap, completed = collector.collect(config.rollout_steps)
            losses = {}
            for team in training_teams:
                data = buffers[team].as_tensors()
                # Scale rewards (±1000 -> ±1) so the critic learns stable values.
                # Only changes the size of the numbers, not what is rewarded.
                data['rewards'] = data['rewards'] * config.reward_scale
                advantages, returns = compute_advantages(data['rewards'], data['values'], data['dones'],
                                                         bootstrap[team], config.gamma, config.gae_lambda)
                pair = teams[team]
                losses[team] = ppo_update(pair.actor, pair.critic, pair.actor_optimizer, pair.critic_optimizer,
                                          data, advantages, returns, epochs=config.epochs,
                                          n_minibatches=config.minibatches)
            record = dict(update=update, transitions=update * config.rollout_steps,
                          device=str(device), losses=losses, completed_matches=completed)
            with metrics_path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            checkpoint = directory / f'update_{update:04d}.pt'
            save_checkpoint(checkpoint, teams, env_config, network_config, config.seed, update,
                            training_teams, training_config=asdict(config))
            metrics.append(record)
            print(f'Update {update}: completed {len(completed)} matches; saved {checkpoint}', flush=True)
        return dict(teams=teams, metrics=metrics, device=device, checkpoint=checkpoint)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('team-size', 3), ('seed', 0), ('updates', 10), ('rollout-steps', 128),
                          ('max-steps', 1000), ('hidden', 64), ('epochs', 4), ('minibatches', 4), ('training-team', 0)):
        parser.add_argument(f'--{name}', type=int, default=default)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--reward-scale', type=float, default=0.001)
    parser.add_argument('--rewards', type=str, default='v1', choices=('v1', 'v2', 'v3'))
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda[:index], or mps')
    parser.add_argument('--output', default='checkpoints/mappo')
    parser.add_argument('--self-play', action='store_true')
    parser.add_argument('--torch-threads', type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error('torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    train(TrainingConfig(**{key: value for key, value in vars(args).items() if key != 'torch_threads'}))


if __name__ == '__main__':
    main()
