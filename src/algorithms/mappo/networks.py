"""Structured soccer actor and centralized value network."""

import torch
from torch import nn
from torch.distributions import Categorical

from enums import MovementActions, SoccerActionType


class SoccerActionDistribution:
    """Factorized actions; unused heads do not affect likelihood or entropy."""

    def __init__(self, logits, active_mask=None):
        logits = dict(logits)
        if active_mask is not None:
            active = torch.as_tensor(active_mask, device=logits['type'].device, dtype=torch.bool)
            idle = torch.full_like(logits['type'], -torch.inf)
            idle[..., int(SoccerActionType.DO_NOTHING)] = 0
            logits['type'] = torch.where(active.unsqueeze(-1), logits['type'], idle)
        self.heads = {name: Categorical(logits=scores) for name, scores in logits.items()}

    def sample(self):
        sampled = {name: dist.sample() for name, dist in self.heads.items()}
        sampled['target'] = torch.stack((sampled.pop('target_x'), sampled.pop('target_y')), dim=-1)
        return sampled

    def _terms(self, actions, entropy=False):
        kind = actions['type']
        def value(name, selected):
            dist = self.heads[name]
            return dist.entropy() if entropy else dist.log_prob(selected)
        result = value('type', kind)
        move = kind == int(SoccerActionType.MOVE)
        kick = kind == int(SoccerActionType.KICK)
        dive = kind == int(SoccerActionType.DIVE)
        tackle = (kind == int(SoccerActionType.TACKLE)) | (kind == int(SoccerActionType.AGGRESSIVE_TACKLE))
        for name, selected, used in (
            ('movement', actions['movement'], move),
            ('contest_intensity', actions['contest_intensity'], move),
            ('target_x', actions['target'][..., 0], kick | dive),
            ('target_y', actions['target'][..., 1], kick | dive),
            ('strength', actions['strength'], kick),
            ('target_player_id', actions['target_player_id'], tackle),
        ):
            result = result + torch.where(used, value(name, selected), torch.zeros_like(result))
        return result

    def log_prob(self, actions):
        return self._terms(actions)

    def entropy(self, actions):
        """Conditional exploration term using only this action's relevant heads."""
        return self._terms(actions, entropy=True)


class Actor(nn.Module):
    """One shared team actor, evaluated separately for each teammate."""

    def __init__(self, obs_size, n_players, hidden=64):
        super().__init__()
        if n_players not in (4, 6):
            raise ValueError('Soccer actor requires a 2v2 or 3v3 roster')
        self.net = nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.heads = nn.ModuleDict({
            f'{name}_head': nn.Linear(hidden, size) for name, size in {
                'type': len(SoccerActionType), 'movement': len(MovementActions),
                'target_x': 7, 'target_y': 7, 'strength': 2,
                'target_player_id': n_players, 'contest_intensity': 2,
            }.items()
        })

    def forward(self, obs, active_mask=None):
        features = self.net(obs)
        return SoccerActionDistribution(
            {name.removesuffix('_head'): head(features) for name, head in self.heads.items()}, active_mask,
        )


class Critic(nn.Module):
    """One centralized scalar value per team and environment state."""

    def __init__(self, state_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


def actions_to_env(actions, agent_ids):
    """Decode one actor batch into the environment's agent-keyed dictionaries."""
    required = ('type', 'movement', 'target', 'strength', 'target_player_id', 'contest_intensity')
    if any(actions[name].shape[0] != len(agent_ids) for name in required):
        raise ValueError('Action batch must match agent IDs')
    return {agent_id: {name: (actions[name][row].detach().cpu().tolist() if name == 'target'
                             else int(actions[name][row].item())) for name in required}
            for row, agent_id in enumerate(agent_ids)}


def masked_actor_mean(values, active_mask):
    """Average actor loss over active players only; all-inactive batches give zero."""
    active = torch.as_tensor(active_mask, device=values.device, dtype=torch.bool)
    if values.shape != active.shape:
        raise ValueError('Actor values and active mask must have identical shapes')
    return values.masked_select(active).sum() / active.sum().clamp_min(1)
