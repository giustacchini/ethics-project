from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv


def arrange(aggressive=False):
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[1].state.pos = (5, 5)
    env.agents[4].state.pos = (6, 5)
    env.ball = CustomBall(position=(6, 5), carrier_id=4, last_possessing_team=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = dict(type=int(SoccerActionType.AGGRESSIVE_TACKLE if aggressive else SoccerActionType.TACKLE),
                      movement=0, target=(3, 3), strength=0, target_player_id=4)
    actions[4] = dict(type=int(SoccerActionType.KICK), movement=0, target=(5, 3), strength=1)
    return env, actions


@pytest.mark.parametrize('aggressive', [False, True])
@pytest.mark.parametrize('scatter_index,carrier', [(0, None), (1, 1), (4, 4)])
def test_success_cancels_kick_even_if_original_carrier_recovers(aggressive, scatter_index, carrier):
    env, actions = arrange(aggressive)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.9] if aggressive else [0.1]),
                         integers=Mock(return_value=scatter_index))
    obs, rewards, *_ = env.step(actions)
    assert env.last_tackle_results[1]['success'] is True
    assert env.last_kick_results[4] == 'cancelled_by_tackle'
    assert env.ball.carrier_id == carrier
    assert env.ball.remaining_trajectory == []
    assert env.ball.position == [(5, 4), (5, 5), (5, 6), (6, 4), (6, 5)][scatter_index]
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize('aggressive', [False, True])
def test_failed_tackle_allows_kick_and_same_transition_travel(aggressive):
    env, actions = arrange(aggressive)
    env.np_random = Mock(random=Mock(side_effect=[0.9, 0.9] if aggressive else [0.9]))
    env.step(actions)
    assert env.last_tackle_results[1]['success'] is False
    assert env.last_kick_results[4] == 'launched'
    assert env.ball.position == (8, 5)
    assert env.ball.carrier_id is None
    assert env.ball.flight_state == 'loose'


@pytest.mark.parametrize('success_draw', [0.1, 0.9])
def test_card_interrupts_kick_regardless_of_sampled_success(success_draw):
    env, actions = arrange(aggressive=True)
    env.np_random = Mock(random=Mock(side_effect=[success_draw, 0.1]))
    env.step(actions)
    assert env.agents[1].suspended
    assert env.last_restart['type'] == 'free_kick'
    assert env.last_kick_results[4] == 'interrupted'
    assert env.ball.position == (6, 5)
    assert env.ball.carrier_id == 4
    assert env.ball.remaining_trajectory == []
    assert not env.np_random.integers.called


def test_noncarrier_cannot_use_preselected_kick_after_scatter_reception():
    env, actions = arrange()
    env.agents[2].state.pos = (5, 4)
    actions[2] = dict(type=1, movement=0, target=(5, 3), strength=1)
    env.np_random = Mock(random=Mock(return_value=0.1), integers=Mock(return_value=0))
    env.step(actions)
    assert env.ball.carrier_id == 2
    assert env.ball.position == (5, 4)
    assert env.last_kick_results[2] == 'not_carrier'
    assert env.last_kick_results[4] == 'cancelled_by_tackle'


def test_action_dictionary_order_does_not_change_outcome():
    snapshots = []
    for reverse in (False, True):
        env, actions = arrange(aggressive=True)
        if reverse:
            actions = dict(reversed(list(actions.items())))
        env.np_random = Mock(random=Mock(side_effect=[0.1, 0.9]), integers=Mock(return_value=4))
        env.step(actions)
        snapshots.append((env.ball.position, env.ball.carrier_id, env.last_kick_results, env.last_tackle_results))
    assert snapshots[0] == snapshots[1]
