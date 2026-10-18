from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.entities.agent import SoccerAgentRole
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv


def arrange(temporary=False, target=(15, 4)):
    env = SoccerEnv()
    env.reset(seed=0)
    keeper_id = 4 if temporary else 3
    if temporary:
        env.agents[3].current_role = SoccerAgentRole.FIELD_PLAYER
        env.agents[3].state.pos = (13, 7)
        env.agents[4].current_role = SoccerAgentRole.GOALKEEPER
    keeper = env.agents[keeper_id]
    keeper.state.pos = (15, 5)
    env.agents[1].state.pos = (13, 4)
    env.ball = CustomBall(position=(13, 4), carrier_id=1, last_possessing_team=0)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[1] = dict(type=int(SoccerActionType.KICK), movement=0,
                      target=(5, 3), strength=1)
    actions[keeper_id] = dict(type=int(SoccerActionType.DIVE), movement=0,
                             target=(target[0] - 15 + 3, target[1] - 5 + 3), strength=0)
    return env, actions, keeper_id


@pytest.mark.parametrize('temporary,draw,saved', [
    (False, 0.69, True), (False, 0.7, False),
    (True, 0.29, True), (True, 0.3, False), (True, 0.5, False),
])
def test_regular_play_dive_probability_for_both_keeper_types(temporary, draw, saved):
    env, actions, keeper_id = arrange(temporary)
    env.np_random = Mock(random=Mock(return_value=draw))
    obs, rewards, *_ = env.step(actions)
    assert env.last_dive_results[keeper_id]['success'] == saved
    assert env.last_dive_results[keeper_id]['save_probability'] == (0.3 if temporary else 0.7)
    assert env.np_random.random.call_count == 1
    if saved:
        assert env.ball.carrier_id == keeper_id
        assert env.agents[keeper_id].state.pos == (15, 4)
        assert env.team_scores == {0: 0, 1: 0}
        assert env.agents[keeper_id].state.carrying is not None
    else:
        assert env.team_scores[0] == 1
        assert env.last_goal['position'] == (15, 4)
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)


def test_failed_dive_stays_put_when_ball_does_not_score():
    env, actions, keeper_id = arrange(target=(14, 4))
    actions[1]['target'] = (4, 3)
    env.np_random = Mock(random=Mock(return_value=0.8))
    env.step(actions)
    assert env.agents[keeper_id].state.pos == (15, 5)
    assert env.ball.position == (14, 4)
    assert env.ball.flight_state == 'loose'
    assert env.team_scores == {0: 0, 1: 0}


def test_occupied_current_cell_caught_without_dive_or_randomness():
    env, actions, keeper_id = arrange()
    env.agents[1].state.pos = (13, 5)
    env.ball.position = (13, 5)
    env.np_random = Mock()
    env.step(actions)
    assert env.ball.carrier_id == keeper_id
    assert not env.np_random.random.called
    assert env.last_goal is None


@pytest.mark.parametrize('case,status', [
    ('field', 'invalid_role'), ('far', 'invalid_displacement'),
    ('zero', 'invalid_displacement'), ('outside', 'invalid_target'),
])
def test_invalid_dive_choices(case, status):
    env, actions, keeper_id = arrange()
    actions[1] = MovementActions.DO_NOTHING
    if case == 'field':
        env.agents[keeper_id].current_role = SoccerAgentRole.FIELD_PLAYER
    elif case == 'far':
        actions[keeper_id]['target'] = (1, 3)
    elif case == 'zero':
        actions[keeper_id]['target'] = (3, 3)
    else:
        actions[keeper_id]['target'] = (4, 3)
    env.np_random = Mock()
    env.step(actions)
    assert env.last_dive_results[keeper_id]['status'] == status
    assert not env.np_random.random.called


def test_dive_cannot_enter_initially_occupied_cell_even_if_vacated():
    env, actions, keeper_id = arrange(target=(14, 4))
    env.agents[2].state.pos = (14, 4)
    actions[2] = MovementActions.UP
    actions[1]['target'] = (4, 3)
    env.np_random = Mock()
    env.step(actions)
    assert env.last_dive_results[keeper_id]['status'] == 'blocked_occupied_cell'
    assert not env.np_random.random.called
    assert env.agents[keeper_id].state.pos == (15, 5)


def test_intermediate_cell_save_stops_strength_two_ball():
    env, actions, keeper_id = arrange(target=(14, 4))
    env.np_random = Mock(random=Mock(return_value=0.1))
    env.step(actions)
    assert env.ball.position == (14, 4)
    assert env.ball.carrier_id == keeper_id
    assert env.ball.remaining_trajectory == []
    assert env.last_goal is None


def test_wrong_target_does_not_automatically_save_adjacent_goal():
    env, actions, keeper_id = arrange(target=(15, 6))
    env.np_random = Mock()
    _, _, _, _, infos = env.step(actions)
    assert env.team_scores[0] == 1
    assert env.last_dive_results[keeper_id]['status'] == 'not_triggered'
    assert not env.np_random.random.called
    assert infos[0]['dives'][keeper_id]['status'] == 'not_triggered'
