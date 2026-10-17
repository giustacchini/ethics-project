"""One advantage/return per team transition, not duplicated critic targets."""

import torch


def compute_advantages(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    if rewards.ndim != 1 or rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError('Rewards, team values and match-end flags must have shape [T]')
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('gamma and lambda must be in [0, 1]')
    advantages = torch.zeros_like(values)
    next_advantage = torch.zeros((), device=values.device, dtype=values.dtype)
    next_value = torch.as_tensor(last_values, device=values.device, dtype=values.dtype).reshape(())
    for t in reversed(range(len(rewards))):
        # Both three-goal termination and match time-limit truncation are true
        # match outcomes here. Neither bootstraps through the following match.
        continuation = (~dones[t].bool()).to(values.dtype)
        delta = rewards[t] + gamma * next_value * continuation - values[t]
        next_advantage = delta + gamma * lam * continuation * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values
