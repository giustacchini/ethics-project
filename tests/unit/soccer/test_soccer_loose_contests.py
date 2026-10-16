from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def arrange(left=0, right=0, size=3):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    env.reset(seed=0)
    ids = (1, size + 1)
    env.agents[ids[0]].state.pos = (5, 5)
    env.agents[ids[1]].state.pos = (7, 5)
    env.ball = CustomBall(position=(6, 5))
    env.grid.set(6, 5, env.ball)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    for player_id, movement, intensity in zip(ids, (3, 4), (left, right)):
        actions[player_id] = dict(type=int(SoccerActionType.MOVE), movement=movement,
                                 target=(3, 3), strength=0, contest_intensity=intensity)
    return env, actions, ids


@pytest.mark.parametrize('draws,winner', [([0.1, 0.9], 1), ([0.9, 0.1], 4), ([0.9, 0.9], None)])
def test_normal_independent_challenges(draws, winner):
    env, actions, ids = arrange()
    env.np_random = Mock(random=Mock(side_effect=draws))
    obs, rewards, *_ = env.step(actions)
    assert env.ball.carrier_id == winner
    assert env.last_loose_ball_contest['winner'] == winner
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)
    if winner is None:
        assert env.agents[ids[0]].state.pos == (5, 5)
        assert env.agents[ids[1]].state.pos == (7, 5)
        assert env.ball.position == (6, 5)


@pytest.mark.parametrize('index,winner', [(0, 1), (1, 4)])
def test_equal_success_intensity_uses_fair_tiebreak(index, winner):
    env, actions, _ = arrange()
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1]), integers=Mock(return_value=index))
    env.step(actions)
    assert env.ball.carrier_id == winner
    env.np_random.integers.assert_called_once_with(2)


def test_aggressive_success_beats_normal_success_without_card():
    env, actions, _ = arrange(left=1)
    env.np_random = Mock(random=Mock(side_effect=[0.79, 0.49, 0.5]))
    env.step(actions)
    assert env.ball.carrier_id == 1
    assert env.last_loose_ball_contest['red_cards'] == []
    assert env.last_foul is None


def test_single_card_preempts_contest_and_restarts_at_ball_cell():
    env, actions, _ = arrange(left=1)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1, 0.1]))
    env.step(actions)
    assert env.last_loose_ball_contest['winner'] is None
    assert env.agents[1].suspended
    assert env.last_foul['position'] == (6, 5)
    assert env.last_restart['type'] == 'free_kick'
    assert env.ball.position == (6, 5)
    assert env.ball.carrier_id == 4
    assert env.agents[1].state.pos == (5, 5)  # no movement committed


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('choice', [0, 1])
def test_double_cards_reducing_both_to_one_use_neutral_restart(size, choice):
    env, actions, _ = arrange(left=1, right=1, size=size)
    if size == 3:
        env._apply_red_card(2)
        env._apply_red_card(5)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1, 0.1, 0.1]),
                         integers=Mock(return_value=choice))
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert env.team_scores == {0: 0, 1: 0}
    assert env.ball.position == (7 + choice, 5)
    assert env.ball.carrier_id is None
    assert env.ball.last_possessing_team is None
    assert all(not a.suspended and a.current_role == a.original_role for a in env.agents)
    assert env.last_restart['type'] == 'neutral'
    assert len(infos[0]['red_cards']) == 2
    assert env.last_goal is None
    assert env.step_count == 1
    assert not any(terms.values()) and not any(truncs.values())
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)


def test_single_card_in_two_v_two_concedes_one_goal():
    env, actions, _ = arrange(left=1, size=2)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1, 0.1]))
    env.step(actions)
    assert env.team_scores == {0: 0, 1: 1}
    assert env.last_restart['type'] == 'automatic_goal'
    assert env.last_loose_ball_contest['winner'] is None


def test_contest_rule_requires_initially_loose_ball():
    env, actions, _ = arrange(left=1, right=1)
    env.ball.remaining_trajectory = [(6, 6)]
    env.ball.speed = 1
    env.np_random = Mock(integers=Mock(return_value=0))
    env.step(actions)
    assert env.last_loose_ball_contest is None
    assert not env.np_random.random.called


def test_teammates_do_not_trigger_loose_ball_challenges():
    env, actions, _ = arrange()
    env.agents[2].state.pos = (7, 5)
    env.agents[4].state.pos = (12, 3)
    actions[2] = actions[4]
    actions[4] = MovementActions.DO_NOTHING
    env.np_random = Mock()
    env.step(actions)
    assert env.last_loose_ball_contest is None
    assert env.ball.carrier_id is None
    assert not env.np_random.random.called


def test_double_cards_with_two_players_remaining_preserve_suspensions():
    env, actions, _ = arrange(left=1, right=1)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1, 0.1, 0.1]), integers=Mock(return_value=1))
    obs, rewards, *_ = env.step(actions)
    assert env.team_scores == {0: 0, 1: 0}
    assert env.agents[1].suspended and env.agents[4].suspended
    assert len(env.active_players) == 4
    assert env.ball.carrier_id is None
    assert env.ball.position == (8, 5)
    assert env.agents[2].state.pos == (3, 7)
    assert env.agents[5].state.pos == (12, 7)
    assert env.last_restart['type'] == 'neutral'
    assert env.step_count == 1
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)


def test_aggressive_contest_can_fail_at_eighty_percent_boundary():
    env, actions, _ = arrange(left=1)
    env.np_random = Mock(random=Mock(side_effect=[0.8, 0.49, 0.5]))
    env.step(actions)
    assert env.ball.carrier_id == 4
    assert env.last_loose_ball_contest['successes'] == {1: False, 4: True}


def test_equal_aggressive_successes_use_tiebreak_after_independent_cards():
    env, actions, _ = arrange(left=1, right=1)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1, 0.8, 0.8]),
                         integers=Mock(return_value=1))
    env.step(actions)
    assert env.ball.carrier_id == 4
    assert env.last_loose_ball_contest['red_cards'] == []
    assert env.np_random.random.call_count == 4


def test_double_card_restart_retains_temporary_goalkeeper_and_restores_after_goal():
    env, _, _ = arrange()
    contest = dict(red_cards=[0, 4], players=[0, 4], position=(2, 5))
    env.np_random = Mock(integers=Mock(return_value=0))
    env._resolve_contest_cards(contest)
    assert env.agents[0].suspended and env.agents[4].suspended
    replacement = next(a for a in env.active_players if a.team_index == 0
                       and a.current_role.name == 'GOALKEEPER')
    assert replacement.index == 1  # equal Manhattan distance, lowest ID wins
    assert replacement.original_role.name == 'FIELD_PLAYER'
    assert replacement.state.pos == (0, 5)
    assert len({a.state.pos for a in env.active_players}) == 4
    assert env.ball.position == (7, 5)
    # A later ordinary goal restores both suspended players and original roles.
    env.agents[5].state.pos = (1, 4)
    env.ball = CustomBall(position=(1, 4), carrier_id=5, last_possessing_team=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[5] = MovementActions.LEFT
    env.step(actions)
    assert all(not a.suspended and a.current_role == a.original_role for a in env.agents)


def test_double_cards_with_asymmetric_rosters_keep_single_team_concession():
    env, actions, _ = arrange(left=1, right=1)
    env._apply_red_card(2)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1, 0.1, 0.1]))
    env.step(actions)
    assert env.team_scores == {0: 0, 1: 1}
    assert env.last_restart['type'] == 'automatic_goal'
    assert all(not a.suspended for a in env.agents)


@pytest.mark.parametrize('players,winner', [((1, 2, 4), 4), ((1, 4, 5), 1), ((1, 2, 4, 5), None)])
def test_multi_player_loose_ball_destination_filters_teammates_without_card_rolls(players, winner):
    env = SoccerEnv()
    env.reset(seed=0)
    positions = {1: (5, 5), 2: (6, 4), 4: (7, 5), 5: (6, 6)}
    moves = {1: 3, 2: 2, 4: 4, 5: 1}
    env.ball = CustomBall(position=(6, 5))
    env.grid.set(6, 5, env.ball)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    for player_id in players:
        env.agents[player_id].state.pos = positions[player_id]
        actions[player_id] = dict(type=0, movement=moves[player_id], target=(3, 3), strength=0,
                                  contest_intensity=1)
    env.np_random = Mock()
    obs, rewards, *_ = env.step(actions)
    assert env.ball.carrier_id == winner
    assert env.last_loose_ball_contest is None  # no pair survives filtering
    assert env.last_red_cards == []
    assert not env.np_random.random.called and not env.np_random.integers.called
    for player_id in players:
        assert env.agents[player_id].state.pos == ((6, 5) if player_id == winner else positions[player_id])
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)
