from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def setup_tackle(seed=0, size=3):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    env.reset(seed=seed)
    target = size + 1
    env.agents[1].state.pos = (5, 5)
    env.agents[target].state.pos = (6, 5)
    env.ball = CustomBall(position=(6, 5), carrier_id=target, last_possessing_team=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = dict(type=int(SoccerActionType.TACKLE), movement=3,
                      target=(3, 3), strength=0, target_player_id=target)
    return env, actions, target


@pytest.mark.parametrize('size', [2, 3])
def test_failed_tackle_keeps_possession_and_has_no_penalty(size):
    env, actions, target = setup_tackle(size=size)
    env.np_random = Mock(random=Mock(return_value=0.5))
    _, rewards, *_ = env.step(actions)
    assert env.last_tackle_results[1]['status'] == 'failed'
    assert env.ball.carrier_id == target
    assert env.agents[1].state.pos == (5, 5)
    assert set(rewards.values()) == {0}


@pytest.mark.parametrize('index,receiver', [(0, None), (1, 1), (4, 4)])
def test_success_scatter_is_not_automatic_tackler_possession(index, receiver):
    env, actions, target = setup_tackle()
    env.np_random = Mock(random=Mock(return_value=0.1), integers=Mock(return_value=index))
    env.step(actions)
    assert env.last_tackle_results[1]['success'] is True
    assert env.ball.carrier_id == receiver
    assert env.ball.position == [(5, 4), (5, 5), (5, 6), (6, 4), (6, 5)][index]
    assert env.ball.last_possessing_team == (0 if receiver == 1 else 1)
    assert env.last_pass_completed is None
    assert env.ball.remaining_trajectory == []
    if receiver is None:
        assert env.grid.get(*env.ball.position) is not None
    else:
        assert env.agents[receiver].state.carrying is not None


@pytest.mark.parametrize('offset', [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)])
def test_all_eight_adjacent_cells_are_valid(offset):
    env, actions, target = setup_tackle()
    env.agents[1].state.pos = (6 + offset[0], 5 + offset[1])
    env.np_random = Mock(random=Mock(return_value=0.9))
    env.step(actions)
    assert env.last_tackle_results[1]['status'] == 'failed'


@pytest.mark.parametrize('case,status', [
    ('teammate', 'invalid_teammate'), ('distant', 'invalid_distance'),
    ('off_ball', 'foul'),
])
def test_invalid_and_pending_cases_are_diagnosed(case, status):
    env, actions, target = setup_tackle()
    if case == 'teammate':
        actions[1]['target_player_id'] = 2
    elif case == 'distant':
        env.agents[1].state.pos = (3, 5)
    elif case == 'off_ball':
        env.ball.carrier_id = 2
    _, rewards, _, _, infos = env.step(actions)
    assert env.last_tackle_results[1]['status'] == status
    assert env.last_tackle_results[1]['success'] is None
    assert infos[0]['tackles'][1]['status'] == status
    assert set(rewards.values()) == {0}


def test_scatter_at_pitch_corner_stays_in_bounds_and_is_reproducible():
    outcomes = []
    for _ in range(2):
        env, actions, target = setup_tackle(seed=1)
        env.agents[target].state.pos = (0, 0)
        env.agents[1].state.pos = (1, 1)
        env.ball.position = (0, 0)
        env.step(actions)
        outcomes.append((env.last_tackle_results, env.ball.position, env.ball.carrier_id))
        assert 0 <= env.ball.position[0] < 16 and 0 <= env.ball.position[1] < 11
    assert outcomes[0] == outcomes[1]


def test_start_adjacency_used_when_carrier_moves():
    env, actions, target = setup_tackle()
    actions[target] = MovementActions.RIGHT
    env.np_random = Mock(random=Mock(return_value=0.1), integers=Mock(return_value=8))
    env.step(actions)
    assert env.last_tackle_results[1]['status'] == 'success'
    assert env.ball.position == (8, 6)


def test_multiple_valid_tackles_resolve_independently():
    env, actions, target = setup_tackle()
    env.agents[2].state.pos = (6, 4)
    actions[2] = dict(actions[1])
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.9]), integers=Mock(return_value=0))
    env.step(actions)
    assert env.last_tackle_results[1]['status'] == 'success'
    assert env.last_tackle_results[2]['status'] == 'failed'
    env.np_random.integers.assert_called_once_with(9)
