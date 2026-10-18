from copy import deepcopy

import numpy as np
import pytest

from soccer.observation.encoding import encode_team, obs_to_vector, state_to_vector
from soccer.env import SoccerEnv, SoccerEnvConfig


@pytest.mark.parametrize('size', [2, 3])
def test_fixed_shape_and_canonical_critic_state(size):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    observations, _ = env.reset(seed=0)
    for team in (0, 1):
        ids, actors, state, active = encode_team(observations, team)
        assert actors.shape == (size, state.size)
        assert actors.dtype == state.dtype == np.float32
        assert np.isfinite(actors).all()
        assert active.tolist() == [True] * size
        for agent_id in ids:
            assert np.array_equal(state, state_to_vector(observations[agent_id], team, 16, 11, 3, 1000))
    assert not np.array_equal(obs_to_vector(observations[0], 16, 11, 3, 1000),
                              obs_to_vector(observations[1], 16, 11, 3, 1000))


@pytest.mark.parametrize('change', ['suspended', 'current_roles', 'original_roles', 'ball_carrier',
    'ball_kicker', 'last_possessing_team', 'ball_state', 'ball_speed', 'ball_trajectory',
    'free_kick_taker', 'score', 'steps_left', 'goal_pending', 'match_result', 'ball_present'])
def test_every_game_state_component_affects_encoding(change):
    env = SoccerEnv()
    observations, _ = env.reset(seed=0)
    obs = observations[0]
    changed = deepcopy(obs)
    if change in ('suspended', 'current_roles', 'original_roles'):
        changed[change][1] = 1 - changed[change][1]
    elif change == 'score':
        changed[change][0] = 1
    elif change == 'ball_trajectory':
        changed[change] = ((8, 5), (9, 5))
    elif change in ('ball_carrier', 'ball_kicker', 'free_kick_taker', 'last_possessing_team'):
        changed[change] = 0
    elif change == 'steps_left':
        changed[change] -= 1
    else:
        changed[change] = 1 - changed[change] if change == 'ball_present' else changed[change] + 1
    assert not np.array_equal(obs_to_vector(obs, 16, 11, 3, 1000), obs_to_vector(changed, 16, 11, 3, 1000))


def test_suspension_mask_and_flight_padding_are_stable():
    env = SoccerEnv()
    obs, _ = env.reset(seed=0)
    before = encode_team(obs, 0)
    env._apply_red_card(2)
    env.ball.carrier_id = None
    env.ball.remaining_trajectory = [(8, 5), (9, 5)]
    env.ball.speed = 1
    after = encode_team(env.gen_obs(), 0)
    assert before[1].shape == after[1].shape
    assert after[3].tolist() == [True, True, False]
    assert np.isfinite(after[1]).all()
