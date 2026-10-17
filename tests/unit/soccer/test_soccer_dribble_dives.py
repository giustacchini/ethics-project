from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.agent import SoccerAgentRole
from soccer.ball import CustomBall
from soccer.env import SoccerEnv


def arrange(team=1, temporary=False, goal=True):
    env = SoccerEnv()
    env.reset(seed=0)
    keeper_id = team * 3 + (1 if temporary else 0)
    carrier_id = (1 - team) * 3 + 1
    if temporary:
        old_keeper = env.agents[team * 3]
        old_keeper.current_role = SoccerAgentRole.FIELD_PLAYER
        old_keeper.state.pos = (13 if team else 2, 7)
        env.agents[keeper_id].current_role = SoccerAgentRole.GOALKEEPER
    target = (15 if team else 0, 4) if goal else (14 if team else 1, 4)
    start = (target[0] - 1 if team else target[0] + 1, 4)
    env.agents[keeper_id].state.pos = (15 if team else 0, 5)
    env.agents[carrier_id].state.pos = start
    env.ball = CustomBall(position=start, carrier_id=carrier_id, last_possessing_team=1 - team)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[carrier_id] = MovementActions.RIGHT if team else MovementActions.LEFT
    dx = target[0] - env.agents[keeper_id].state.pos[0]
    actions[keeper_id] = dict(type=int(SoccerActionType.DIVE), movement=0,
                             target=(dx + 3, 2), strength=0)
    return env, actions, keeper_id, carrier_id, start, target


@pytest.mark.parametrize('team', [0, 1])
@pytest.mark.parametrize('temporary', [False, True])
@pytest.mark.parametrize('goal', [False, True])
def test_saved_dribble_returns_carrier_and_updates_both_records(team, temporary, goal):
    env, actions, keeper, carrier, start, target = arrange(team, temporary, goal)
    env.np_random = Mock(random=Mock(return_value=0.29 if temporary else 0.69))
    obs, rewards, *_ = env.step(actions)
    assert env.ball.carrier_id == keeper
    assert env.ball.position == env.agents[keeper].state.pos == target
    assert env.agents[carrier].state.pos == start
    assert env.last_movement_results[carrier]['status'] == 'blocked_by_dive'
    assert tuple(env.last_movement_results[carrier]['resolved_position']) == start
    assert env.last_movement_results[keeper]['status'] == 'dived'
    assert tuple(env.last_movement_results[keeper]['resolved_position']) == target
    assert env.last_possession_transfer['to_team'] == team
    assert env.team_scores == {0: 0, 1: 0}
    assert len({a.state.pos for a in env.active_players}) == 6
    assert env.agents[keeper].state.carrying is not None
    assert env.agents[carrier].state.carrying is None
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)
    env.np_random.random.assert_called_once()


@pytest.mark.parametrize('temporary', [False, True])
@pytest.mark.parametrize('goal', [False, True])
def test_failed_dive_preserves_dribble_and_permits_goal(temporary, goal):
    env, actions, keeper, carrier, start, target = arrange(temporary=temporary, goal=goal)
    env.np_random = Mock(random=Mock(return_value=0.3 if temporary else 0.7))
    env.step(actions)
    assert env.last_dive_results[keeper]['success'] is False
    assert tuple(env.last_movement_results[carrier]['resolved_position']) == target
    assert env.last_movement_results[carrier]['status'] == 'moved'
    assert env.last_movement_results[keeper]['status'] == 'waited'
    if goal:
        assert env.last_goal['position'] == target
        assert env.team_scores[0] == 1
    else:
        assert env.ball.carrier_id == carrier
        assert env.agents[carrier].state.pos == target
        assert env.agents[keeper].state.pos == (15, 5)
    env.np_random.random.assert_called_once()


def test_other_player_cannot_enter_dribblers_starting_cell_before_rollback():
    env, actions, keeper, carrier, start, target = arrange()
    env.agents[2].state.pos = (13, 4)
    actions[2] = MovementActions.RIGHT
    env.np_random = Mock(random=Mock(return_value=0.1))
    env.step(actions)
    assert env.agents[carrier].state.pos == start
    assert env.agents[2].state.pos == (13, 4)
    assert env.last_movement_results[2]['status'] == 'blocked_occupied_cell'
    assert len({a.state.pos for a in env.active_players}) == 6


@pytest.mark.parametrize('team', [0, 1])
@pytest.mark.parametrize('row,inside', [(2, False), (3, True), (7, True), (8, False)])
def test_goalkeeper_movement_penalty_classification_and_shading_agree(team, row, inside):
    env = SoccerEnv()
    env.reset(seed=0)
    x = 1 if team == 0 else 14
    keeper = env.agents[team * 3]
    assert env.is_move_allowed(env.grid, keeper, keeper.state.pos, (x, row)) == inside
    assert env._is_penalty_foul(team, (x, row)) == inside
    assert (env.grid.background_at(x, row) == env.grid.GOALKEEPER_COLOR) == inside
