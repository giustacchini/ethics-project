from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.ball import CustomBall
from soccer.env import SoccerEnv


def foul_action(target, card=False):
    return dict(type=int(SoccerActionType.AGGRESSIVE_TACKLE if card else SoccerActionType.TACKLE),
                movement=0, target=(3, 3), strength=0, target_player_id=target)


def arrange(both_teams=False, penalty=False, cards=False):
    env = SoccerEnv()
    env.reset(seed=7)
    env.agents[1].state.pos = (4, 5)
    env.agents[4].state.pos = (5, 5)
    env.agents[2].state.pos = (3, 5) if penalty else (6, 7)
    env.agents[5].state.pos = (2, 5) if penalty else (7, 7)
    env.ball = CustomBall(position=(8, 8), remaining_trajectory=[(9, 8)], speed=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[1] = foul_action(4, cards)
    actions[5 if both_teams else 2] = foul_action(2 if both_teams else 5, cards)
    return env, actions


def test_same_team_penalty_takes_priority_over_midfield_free_kick():
    env, actions = arrange(penalty=True)
    env.np_random = Mock(integers=Mock(return_value=0), random=Mock(return_value=0.1))
    _, rewards, _, _, infos = env.step(actions)
    assert env.last_restart['type'] == 'penalty'
    assert env.last_restart['taker'] == 5
    assert env.last_foul['position'] == (2, 5)
    assert len(infos[0]['fouls']) == 2
    assert all(env.last_tackle_results[i]['status'] == 'foul' for i in (1, 2))
    assert env.last_movement_results == {}
    assert set(rewards.values()) == {0}


@pytest.mark.parametrize('choice,target,spot', [(0, 4, (5, 5)), (1, 5, (7, 7))])
def test_equal_priority_fouls_are_randomly_selected(choice, target, spot):
    env, actions = arrange()
    env.np_random = Mock(integers=Mock(return_value=choice))
    env.step(actions)
    assert env.last_restart['type'] == 'free_kick'
    assert env.last_restart['taker'] == target
    assert env.ball.position == spot
    env.np_random.integers.assert_called_once_with(2)


@pytest.mark.parametrize('cards', [False, True])
def test_both_team_fouls_restart_neutrally_preserving_cards(cards):
    env, actions = arrange(both_teams=True, cards=cards)
    env.np_random = Mock(random=Mock(return_value=0.1), integers=Mock(return_value=1))
    obs, rewards, *_ = env.step(actions)
    assert env.last_restart['type'] == 'neutral'
    assert env.ball.position == (8, 5)
    assert env.team_scores == {0: 0, 1: 0}
    assert env.agents[1].suspended == cards
    assert env.agents[5].suspended == cards
    assert len(env.last_red_cards) == (2 if cards else 0)
    assert len(env.last_fouls) == 2
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)


def test_both_teams_reduced_to_one_restore_everyone_without_goals():
    env, actions = arrange(both_teams=True, cards=True)
    env._apply_red_card(0)
    env._apply_red_card(3)
    # Dismissing goalkeepers relocates their replacements; arrange fresh
    # adjacent challenges after those earlier disciplinary restarts.
    env.agents[1].state.pos = (1, 5)
    env.agents[5].state.pos = (2, 5)
    env.agents[4].state.pos = (14, 5)
    env.agents[2].state.pos = (13, 5)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[1] = foul_action(5, True)
    actions[4] = foul_action(2, True)
    env.np_random = Mock(random=Mock(return_value=0.1), integers=Mock(return_value=0))
    env.step(actions)
    assert env.last_restart['type'] == 'neutral'
    assert env.team_scores == {0: 0, 1: 0}
    assert env.ball.position == (7, 5)
    assert all(not a.suspended and a.current_role == a.original_role for a in env.agents)


def test_only_one_team_reduced_to_one_concedes_instead_of_neutral_restart():
    env, actions = arrange(both_teams=True, cards=True)
    env._apply_red_card(0)
    env.agents[1].state.pos = (1, 5)
    env.agents[4].state.pos = (2, 5)
    env.np_random = Mock(random=Mock(return_value=0.1))
    env.step(actions)
    assert env.last_restart['type'] == 'automatic_goal'
    assert env.team_scores == {0: 0, 1: 1}
    assert len(env.last_red_cards) == 2
    assert not env.np_random.integers.called


def test_same_team_different_target_cards_concede_only_once():
    env, actions = arrange(cards=True)
    env.np_random = Mock(random=Mock(return_value=0.1))
    env.step(actions)
    assert env.last_restart['type'] == 'automatic_goal'
    assert env.team_scores == {0: 0, 1: 1}
    assert len(env.last_fouls) == len(env.last_red_cards) == 2


def test_selection_is_seeded_and_independent_of_action_map_order():
    snapshots = []
    for reverse in (False, True):
        env, actions = arrange()
        if reverse:
            actions = dict(reversed(list(actions.items())))
        env.step(actions)
        snapshots.append((env.last_foul, env.last_restart, env.ball.position))
    assert snapshots[0] == snapshots[1]
