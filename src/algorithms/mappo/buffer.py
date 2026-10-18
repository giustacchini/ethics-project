"""Detached per-team rollouts with structured actions and centralized states."""

import torch


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.rows = []

    def add(self, obs, state, actions, log_probs, value, reward, done, active_mask):
        self.rows.append(dict(
            obs=obs.detach().clone(), states=state.detach().clone(),
            actions={key: tensor.detach().clone() for key, tensor in actions.items()},
            log_probs=log_probs.detach().clone(), values=value.detach().clone().reshape(()),
            rewards=torch.as_tensor(reward, dtype=obs.dtype, device=obs.device),
            dones=torch.as_tensor(done, dtype=torch.bool, device=obs.device),
            active_masks=active_mask.detach().clone().bool(),
        ))

    def __len__(self):
        return len(self.rows)

    def as_tensors(self):
        if not self.rows:
            raise ValueError('Cannot train from an empty rollout')
        result = {key: torch.stack([row[key] for row in self.rows])
                  for key in self.rows[0] if key != 'actions'}
        result['actions'] = {key: torch.stack([row['actions'][key] for row in self.rows])
                             for key in self.rows[0]['actions']}
        return result
