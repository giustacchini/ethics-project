from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv


def arrange(kick=False, aggressive=()):
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[1].state.pos = (5, 5)
    env.agents[2].state.pos = (6, 4)
    env.agents[4].state.pos = (6, 5)
    env.ball = CustomBall(position=(6, 5), carrier_id=4, last_possessing_team=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    for i in (1, 2):
        actions[i] = dict(type=int(SoccerActionType.AGGRESSIVE_TACKLE if i in aggressive else SoccerActionType.TACKLE),
                          movement=0, target=(3, 3), strength=0, target_player_id=4)
    if kick:
        actions[4] = dict(type=1, movement=0, target=(5, 3), strength=1)
    return env, actions


@pytest.mark.parametrize('draws,outcomes', [([0.1, 0.9], [True, False]),
    ([0.9, 0.1], [False, True]), ([0.1, 0.1], [True, True]), ([0.9, 0.9], [False, False])])
@pytest.mark.parametrize('kick', [False, True])
def test_independent_outcomes_produce_one_scatter(draws, outcomes, kick):
    env, actions = arrange(kick)
    env.np_random = Mock(random=Mock(side_effect=draws), integers=Mock(return_value=4))
    _, rewards, *_ = env.step(actions)
    assert [env.last_tackle_results[i]['success'] for i in (1, 2)] == outcomes
    assert env.np_random.random.call_count == 2
    if any(outcomes):
        env.np_random.integers.assert_called_once_with(9)
        assert env.ball.carrier_id == 4  # original carrier recovers the one scatter
        if kick:
            assert env.last_kick_results[4] == 'cancelled_by_tackle'
    else:
        assert not env.np_random.integers.called
        if kick:
            assert env.last_kick_results[4] == 'launched'
            assert env.ball.position == (8, 5)
        else:
            assert env.ball.carrier_id == 4
    assert set(rewards.values()) == {0}


def test_mixed_intensities_draw_independently():
    env, actions = arrange(aggressive=(2,))
    env.np_random = Mock(random=Mock(side_effect=[0.79, 0.9, 0.6]), integers=Mock(return_value=0))
    env.step(actions)
    assert env.last_tackle_results[1]['success'] is False
    assert env.last_tackle_results[2]['success'] is True
    env.np_random.integers.assert_called_once_with(9)


def test_one_card_cancels_every_physical_tackle_and_kick():
    env, actions = arrange(kick=True, aggressive=(2,))
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1]))
    env.step(actions)
    assert env.agents[2].suspended
    assert env.last_tackle_results[1]['success'] is None
    assert env.last_tackle_results[2]['success'] is None
    assert env.last_kick_results[4] == 'interrupted'
    assert env.last_restart['type'] == 'free_kick'
    assert env.ball.carrier_id == 4
    assert not env.np_random.integers.called


def test_two_cards_apply_together_and_concede_only_once():
    env, actions = arrange(kick=True, aggressive=(1, 2))
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1, 0.9, 0.1]))
    env.step(actions)
    assert env.team_scores == {0: 0, 1: 1}
    assert len(env.last_red_cards) == 2
    assert env.last_restart['type'] == 'goal_by_forfait'
    assert env.last_kick_results[4] == 'interrupted'
    assert all(not a.suspended for a in env.agents)
    assert not env.np_random.integers.called


def test_multiple_tackle_results_do_not_depend_on_action_order():
    results = []
    for reverse in (False, True):
        env, actions = arrange(kick=True)
        env.np_random = Mock(random=Mock(side_effect=[0.1, 0.9]), integers=Mock(return_value=0))
        if reverse:
            actions = dict(reversed(list(actions.items())))
        env.step(actions)
        results.append((env.last_tackle_results, env.last_kick_results, env.ball.position))
    assert results[0] == results[1]
