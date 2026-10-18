"""Two independent actor/critic pairs and their separate optimizers."""

from dataclasses import dataclass

import torch

from algorithms.mappo.networks import Actor, Critic
from algorithms.mappo.device import resolve_device


@dataclass
class TeamNetworks:
    actor: Actor
    critic: Critic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer


def create_teams(obs_size, state_size, n_players, hidden=64, learning_rate=3e-4, device='cpu'):
    device = resolve_device(device)
    teams = {}
    for team in (0, 1):
        actor = Actor(obs_size, n_players, hidden).to(device)
        critic = Critic(state_size, hidden).to(device)
        teams[team] = TeamNetworks(actor, critic, torch.optim.Adam(actor.parameters(), lr=learning_rate),
                                   torch.optim.Adam(critic.parameters(), lr=learning_rate))
    return teams
