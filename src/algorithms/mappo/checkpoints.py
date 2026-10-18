"""Versioned, portable checkpoints for the current soccer schema."""

from dataclasses import asdict
from pathlib import Path

import torch

from soccer.observation.encoding import ENCODING_VERSION

CHECKPOINT_VERSION = 1


def save_checkpoint(path, teams, env_config, network_config, seed, update, training_teams, training_config=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint_version=CHECKPOINT_VERSION, encoding_version=ENCODING_VERSION,
                   env_config=asdict(env_config), network_config=network_config,
                   seed=seed, update=update, training_teams=list(training_teams),
                   training_config=training_config,
                   torch_version=str(torch.__version__),
                   teams={team: dict(actor=pair.actor.state_dict(), critic=pair.critic.state_dict(),
                                     actor_optimizer=pair.actor_optimizer.state_dict(),
                                     critic_optimizer=pair.critic_optimizer.state_dict())
                          for team, pair in teams.items()})
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('checkpoint_version') != CHECKPOINT_VERSION or payload.get('encoding_version') != ENCODING_VERSION:
        raise ValueError('Unsupported checkpoint/schema; old flat-action checkpoints cannot be loaded')
    return payload
