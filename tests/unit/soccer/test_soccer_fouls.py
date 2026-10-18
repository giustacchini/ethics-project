from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.entities.agent import SoccerAgentRole
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def arrange_foul(team=0, spot=(5, 5), size=3, max_steps=1000):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size), max_steps=max_steps)
    env.reset(seed=0)
    offender = env.agents[team * size + 1]
    taker = env.agents[(1 - team) * size + 1]
    taker.state.pos = spot
    offender.state.pos = (spot[0] + (1 if team == 0 else -1), spot[1])
    # Keep the ball off the fouled player and in flight to test interruption.
    env.ball = CustomBall(position=(8, 8), remaining_trajectory=[(9, 8)], speed=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[offender.index] = dict(type=int(SoccerActionType.TACKLE), movement=0,
                                  target=(3, 3), strength=0, target_player_id=taker.index)
    return env, actions, offender, taker


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('team', [0, 1])
def test_free_kick_awards_fouled_player_and_cancels_joint_action(size, team):
    env, actions, offender, taker = arrange_foul(team=team, size=size)
    actions[taker.index] = MovementActions.RIGHT
    obs, rewards, _, _, infos = env.step(actions)
    assert env.last_restart['type'] == 'free_kick'
    assert env.last_foul['position'] == (5, 5)
    assert env.ball.carrier_id == taker.index
    assert env.ball.position == taker.state.pos == (5, 5)
    assert env.ball.remaining_trajectory == []
    assert env.grid.get(8, 8) is None
    assert set(rewards.values()) == {0}
    assert env.last_movement_results == {}
    assert len({a.state.pos for a in env.agents}) == size * 2
    for agent in env.agents:
        if agent.index != taker.index:
            assert env._manhattan(agent.state.pos, taker.state.pos) >= 3
        if agent.current_role == SoccerAgentRole.GOALKEEPER:
            assert env.is_move_allowed(env.grid, agent, agent.state.pos, agent.state.pos)
    assert infos[0]['foul']['offender'] == offender.index
    assert env.observation_space.contains(obs)
    env.step({a.index: MovementActions.DO_NOTHING for a in env.agents})
    assert env.last_foul is None and env.last_restart is None


@pytest.mark.parametrize('team', [0, 1])
@pytest.mark.parametrize('sample,saved', [(0.69, True), (0.7, False)])
def test_penalty_automatic_save_or_goal(team, sample, saved):
    spot = (2 if team == 0 else 13, 5)
    env, actions, offender, taker = arrange_foul(team=team, spot=spot)
    env.np_random = Mock(integers=Mock(return_value=0), random=Mock(return_value=sample))
    obs, rewards, *_ = env.step(actions)
    assert env.last_restart['type'] == 'penalty'
    assert env.last_restart['position'] == spot
    assert env.last_restart['saved'] == saved
    assert env.last_restart['save_probability'] == 0.7
    assert set(rewards.values()) == {0}
    if saved:
        keeper = env.agents[team * 3]
        assert env.ball.carrier_id == keeper.index
        assert env.ball.position == (0 if team == 0 else 15, 4)
        assert env.team_scores == {0: 0, 1: 0}
    else:
        assert env.team_scores[1 - team] == 1
        assert env.last_goal['team'] == 1 - team
        assert env.ball.position == ((7, 5) if team == 0 else (8, 5))
    assert env.observation_space.contains(obs)


def test_temporary_goalkeeper_penalty_save_probability():
    env, actions, _, _ = arrange_foul(spot=(2, 5))
    env.agents[0].current_role = SoccerAgentRole.FIELD_PLAYER
    env.agents[2].current_role = SoccerAgentRole.GOALKEEPER
    env.agents[2].state.pos = (1, 5)
    env.np_random = Mock(integers=Mock(return_value=1), random=Mock(return_value=0.3))
    env.step(actions)
    assert env.last_restart['save_probability'] == 0.3
    assert not env.last_restart['saved']


def test_penalty_can_end_match_with_only_outcome_rewards():
    env, actions, _, _ = arrange_foul(spot=(2, 5))
    env.team_scores[1] = 2
    env.np_random = Mock(integers=Mock(return_value=0), random=Mock(return_value=0.9))
    _, rewards, terms, _, _ = env.step(actions)
    assert all(terms.values()) and env.winner == 1
    assert env.team_scores[1] == 3
    assert all(rewards[a.index] == (1000 if a.team_index == 1 else -1000) for a in env.agents)


def test_foul_in_other_teams_area_is_not_penalty():
    env, actions, _, _ = arrange_foul(team=0, spot=(13, 5))
    env.step(actions)
    assert env.last_restart['type'] == 'free_kick'


def test_restart_placement_is_deterministic():
    snapshots = []
    for _ in range(2):
        env, actions, _, _ = arrange_foul()
        env.step(actions)
        snapshots.append([a.state.pos for a in env.agents])
    assert snapshots[0] == snapshots[1]


def test_same_team_fouls_against_same_player_share_one_restart():
    env, actions, _, taker = arrange_foul()
    env.agents[2].state.pos = (5, 4)
    actions[2] = dict(type=3, movement=0, target=(3, 3), strength=0, target_player_id=taker.index)
    env.step(actions)
    assert env.last_foul['offenders'] == [1, 2]
    assert env.last_restart['type'] == 'free_kick'
    assert all(env.last_tackle_results[i]['status'] == 'foul' for i in (1, 2))
