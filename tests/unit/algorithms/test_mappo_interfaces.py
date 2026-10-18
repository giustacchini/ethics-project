import numpy as np
import pytest
import torch

from algorithms.mappo.networks import Actor, SoccerActionDistribution, actions_to_env, masked_actor_mean
from algorithms.mappo.teams import create_teams
from enums import SoccerActionType
from soccer.observation.encoding import encode_team
from soccer.env import SoccerEnv, SoccerEnvConfig


def logits(batch=2):
    return {name: torch.randn(batch, size, requires_grad=True) for name, size in
            dict(type=6, movement=9, target_x=7, target_y=7, strength=2,
                 target_player_id=6, contest_intensity=2).items()}


@pytest.mark.parametrize('kind,used', [
    (SoccerActionType.DO_NOTHING, {'type'}),
    (SoccerActionType.MOVE, {'type', 'movement', 'contest_intensity'}),
    (SoccerActionType.KICK, {'type', 'target_x', 'target_y', 'strength'}),
    (SoccerActionType.DIVE, {'type', 'target_x', 'target_y'}),
    (SoccerActionType.TACKLE, {'type', 'target_player_id'}),
    (SoccerActionType.AGGRESSIVE_TACKLE, {'type', 'target_player_id'}),
])
def test_only_relevant_action_heads_contribute_probability_and_gradients(kind, used):
    scores = logits()
    distribution = SoccerActionDistribution(scores)
    actions = distribution.sample()
    actions['type'][:] = int(kind)
    expected = distribution.heads['type'].log_prob(actions['type'])
    for name in used - {'type'}:
        selected = actions['target'][..., 0 if name == 'target_x' else 1] if name.startswith('target_') and name != 'target_player_id' else actions[name]
        expected = expected + distribution.heads[name].log_prob(selected)
    actual = distribution.log_prob(actions)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    for name, score in scores.items():
        if name not in used:
            assert score.grad is None or torch.count_nonzero(score.grad) == 0
    assert torch.isfinite(distribution.entropy(actions)).all()


def test_inactive_players_wait_and_contribute_no_actor_loss():
    actor = Actor(8, 6)
    active = torch.tensor([True, False])
    distribution = actor(torch.zeros(2, 8), active)
    actions = distribution.sample()
    assert actions['type'][1] == int(SoccerActionType.DO_NOTHING)
    assert distribution.log_prob(actions)[1] == 0
    assert distribution.entropy(actions)[1] == 0
    loss_terms = torch.tensor([2.0, 100.0], requires_grad=True)
    loss = masked_actor_mean(loss_terms, active)
    loss.backward()
    assert loss.item() == 2 and loss_terms.grad.tolist() == [1.0, 0.0]
    assert masked_actor_mean(loss_terms, torch.zeros(2, dtype=torch.bool)) == 0


@pytest.mark.parametrize('size', [2, 3])
def test_four_networks_step_soccer_with_random_baseline(size):
    torch.manual_seed(0)
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size), max_steps=10)
    observations, _ = env.reset(seed=0)
    ids, actor_obs, state, active = encode_team(observations, 0, max_steps=10)
    teams = create_teams(actor_obs.shape[-1], len(state), size * 2)
    parameter_sets = [{id(p) for p in network.parameters()}
                      for pair in teams.values() for network in (pair.actor, pair.critic)]
    for i, first in enumerate(parameter_sets):
        assert all(first.isdisjoint(other) for other in parameter_sets[i + 1:])
    action_space = env.action_space
    action_space.seed(7)
    for _ in range(10):
        ids, actor_obs, state, active = encode_team(observations, 0, max_steps=10)
        with torch.no_grad():
            distribution = teams[0].actor(torch.from_numpy(actor_obs), torch.from_numpy(active))
            sampled = distribution.sample()
            assert distribution.log_prob(sampled).shape == (size,)
            assert teams[0].critic(torch.from_numpy(state)).shape == ()
        actions = action_space.sample()  # Team B retains its random baseline
        actions.update(actions_to_env(sampled, ids))
        assert env.action_space.contains(actions)
        observations, *_ = env.step(actions)
        assert env.observation_space.contains(observations)
    assert env.match_over
    before = [p.detach().clone() for p in teams[1].actor.parameters()]
    loss = -teams[0].actor(torch.from_numpy(actor_obs)).log_prob(sampled).mean()
    teams[0].actor_optimizer.zero_grad()
    loss.backward()
    teams[0].actor_optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, teams[1].actor.parameters()))
