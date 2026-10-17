"""Clipped PPO for shared actors and one centralized critic target per step."""

import torch
from torch import nn


def ppo_update(actor, critic, actor_opt, critic_opt, data, advantages, returns,
               epochs=4, n_minibatches=4, clip=0.2, ent_coef=0.01, max_grad_norm=0.5):
    if epochs < 1 or n_minibatches < 1 or len(data['states']) == 0:
        raise ValueError('PPO requires positive epochs/minibatches and a nonempty rollout')
    device = data['obs'].device
    time_steps, agents, features = data['obs'].shape
    obs = data['obs'].reshape(-1, features)
    actions = {name: value.reshape(time_steps * agents, *value.shape[2:]) for name, value in data['actions'].items()}
    old_log_probs = data['log_probs'].reshape(-1)
    active = data['active_masks'].reshape(-1)
    # A team advantage is shared across its active players; critic targets
    # remain [T] and are never multiplied by the team roster size.
    actor_advantages = advantages[:, None].expand(-1, agents).reshape(-1)
    actor_indices = active.nonzero(as_tuple=True)[0]
    selected = actor_advantages[actor_indices]
    normalized = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8) if selected.numel() else selected
    actor_stats, critic_stats = [], []
    for _ in range(epochs):
        if actor_indices.numel():
            order = torch.randperm(actor_indices.numel(), device=device)
            for minibatch in torch.tensor_split(order, min(n_minibatches, len(order))):
                index = actor_indices[minibatch]
                batch_actions = {name: value[index] for name, value in actions.items()}
                distribution = actor(obs[index])
                log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy(batch_actions).mean()
                ratio = (log_probs - old_log_probs[index]).exp()
                advantage = normalized[minibatch]
                objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
                loss = -objective.mean() - ent_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite actor loss')
                actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_opt.step()
                actor_stats.append((loss.item(), entropy.item()))
        order = torch.randperm(time_steps, device=device)
        for index in torch.tensor_split(order, min(n_minibatches, time_steps)):
            predicted = critic(data['states'][index])
            loss = 0.5 * (predicted - returns[index]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite critic loss')
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()
            critic_stats.append(loss.item())
    return dict(actor_loss=sum(x[0] for x in actor_stats) / max(1, len(actor_stats)),
                entropy=sum(x[1] for x in actor_stats) / max(1, len(actor_stats)),
                critic_loss=sum(critic_stats) / len(critic_stats),
                actor_samples=int(actor_indices.numel()), critic_samples=time_steps)
