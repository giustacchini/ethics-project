from unittest.mock import Mock

import numpy as np
import pytest

from enums import MovementActions, SoccerActionType
from soccer.agent import SoccerAgentRole
from soccer.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def carded_foul(env, offender_id, target_id, monkeypatch):
    # Supply a confirmed card at the future aggressive-tackle integration seam.
    original = env._validate_tackles
    def confirmed(requests):
        valid = original(requests)
        env.last_tackle_results[offender_id]['red_card'] = True
        return valid
    monkeypatch.setattr(env, '_validate_tackles', confirmed)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[offender_id] = dict(type=int(SoccerActionType.TACKLE), movement=0,
                               target=(3, 3), strength=0, target_player_id=target_id)
    return env.step(actions)


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('offender_slot', [0, 1])
def test_one_player_concedes_instead_of_free_kick_or_penalty(size, offender_slot, monkeypatch):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    env.reset(seed=0)
    if size == 3:
        env._apply_red_card(2)
    offender = env.agents[offender_slot]
    target = env.agents[size + 1]
    offender.state.pos = (1, 5)
    target.state.pos = (2, 5)
    env.ball = CustomBall(position=(8, 8))
    env.np_random = Mock()  # Automatic concession must not sample a penalty.
    obs, rewards, terms, _, infos = carded_foul(env, offender.index, target.index, monkeypatch)
    assert env.team_scores == {0: 0, 1: 1}
    assert env.last_goal['reason'] == 'one_active_player'
    assert infos[0]['restart']['type'] == 'automatic_goal'
    assert infos[0]['red_card']['player'] == offender.index
    assert not env.np_random.random.called and not env.np_random.integers.called
    assert all(not a.suspended for a in env.agents)
    assert all(a.current_role == a.original_role for a in env.agents)
    assert env.ball.position == (7, 5)
    assert not any(terms.values())
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)


def test_automatic_concession_can_end_match(monkeypatch):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=2))
    env.reset(seed=0)
    env.team_scores[1] = 2
    env.agents[1].state.pos = (5, 5)
    env.agents[3].state.pos = (6, 5)
    env.ball = CustomBall(position=(8, 8))
    obs, rewards, terms, truncs, _ = carded_foul(env, 1, 3, monkeypatch)
    assert env.team_scores[1] == 3 and env.winner == 1
    assert all(terms.values()) and not any(truncs.values())
    assert env.agents[1].suspended
    assert obs[0]['agent_positions'][1] == (-1, -1)
    assert all(rewards[a.index] == (1000 if a.team_index == 1 else -1000) for a in env.agents)


def test_dismissed_goalkeeper_replaced_by_closest_field_player(monkeypatch):
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[0].state.pos = (1, 5)
    env.agents[1].state.pos = (8, 3)
    env.agents[2].state.pos = (3, 5)
    env.agents[4].state.pos = (2, 5)
    env.ball = CustomBall(position=(8, 8))
    env.np_random = Mock(integers=Mock(return_value=0), random=Mock(return_value=0.1))
    obs, _, terms, _, _ = carded_foul(env, 0, 4, monkeypatch)
    assert env.agents[0].suspended
    assert env.last_red_card['replacement'] == 2
    assert env.agents[2].current_role == SoccerAgentRole.GOALKEEPER
    assert env.agents[2].original_role == SoccerAgentRole.FIELD_PLAYER
    assert env.last_restart['type'] == 'penalty'
    assert env.last_restart['save_probability'] == 0.3
    assert env.ball.carrier_id == 2
    assert not any(terms.values())  # Suspension is not episode termination.
    assert obs[1]['suspended'][0] == 1
    assert env.observation_space.contains(obs)


def test_replacement_tie_breaks_by_agent_id():
    env = SoccerEnv()
    env.reset(seed=0)
    assert env._apply_red_card(0) is False
    assert env.last_red_card['replacement'] == 1
    assert env.agents[1].state.pos == (0, 5)


def test_suspended_players_are_not_obstacles_receivers_or_rendered():
    env = SoccerEnv()
    env.reset(seed=0)
    old_position = env.agents[2].state.pos
    env._apply_red_card(2)
    env.agents[4].state.pos = (4, 7)
    env.ball = CustomBall(position=(4, 7), carrier_id=4, last_possessing_team=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[4] = MovementActions.LEFT
    obs, _, terms, _, _ = env.step(actions)
    assert env.agents[4].state.pos == old_position
    assert env.ball.carrier_id == 4
    assert obs[0]['agent_positions'][2] == (-1, -1)
    assert obs[0]['suspended'][2] == 1
    assert not terms[2]
    assert env.observation_space.contains(obs)
    image = env.grid.render(tile_size=16, agents=[env.agents[2]])
    empty = env.grid.render(tile_size=16, agents=[])
    assert np.array_equal(image, empty)
    env.reset(seed=0)
    assert all(not a.suspended for a in env.agents)


def test_goal_restores_suspended_players_and_original_roles():
    env = SoccerEnv()
    env.reset(seed=0)
    env._apply_red_card(0)
    env.agents[4].state.pos = (1, 4)
    env.ball = CustomBall(position=(1, 4), carrier_id=4, last_possessing_team=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[4] = MovementActions.LEFT
    env.step(actions)
    assert env.team_scores[1] == 1
    assert all(not a.suspended and a.current_role == a.original_role for a in env.agents)
    assert env.agents[0].state.pos == (0, 5)


def test_tackling_suspended_target_is_invalid():
    env = SoccerEnv()
    env.reset(seed=0)
    env._apply_red_card(2)
    env.agents[4].state.pos = (4, 7)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[4] = dict(type=3, movement=0, target=(3, 3), strength=0, target_player_id=2)
    env.step(actions)
    assert env.last_tackle_results[4]['status'] == 'invalid_suspended_player'
