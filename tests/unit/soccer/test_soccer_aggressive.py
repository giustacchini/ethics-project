from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def arrange(size=3, off_ball=False, tackler=1):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    env.reset(seed=0)
    target = size + 1
    env.agents[tackler].state.pos = (5, 5)
    env.agents[target].state.pos = (6, 5)
    carrier = 0 if off_ball else target
    env.ball = CustomBall(position=env.agents[carrier].state.pos,
                          carrier_id=carrier, last_possessing_team=env.agents[carrier].team_index)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.active_players}
    actions[tackler] = dict(type=int(SoccerActionType.AGGRESSIVE_TACKLE),
                           movement=3, target=(3, 3), strength=0, target_player_id=target)
    return env, actions, target


@pytest.mark.parametrize('success_draw,success', [(0.79, True), (0.8, False)])
@pytest.mark.parametrize('card_draw,card', [(0.49, True), (0.5, False)])
def test_independent_success_and_card_with_card_precedence(success_draw, success, card_draw, card):
    env, actions, target = arrange()
    env.np_random = Mock(random=Mock(side_effect=[success_draw, card_draw]),
                         integers=Mock(return_value=0))
    obs, rewards, terms, _, infos = env.step(actions)
    result = env.last_tackle_results[1]
    assert result['intensity'] == 'aggressive'
    assert result['sampled_success'] == success
    assert result['red_card'] == card
    assert env.np_random.random.call_count == 2
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)
    if card:
        assert result['status'] == 'foul' and result['success'] is None
        assert env.agents[1].suspended
        assert env.last_restart['type'] == 'free_kick'
        assert env.ball.carrier_id == target
        assert not env.np_random.integers.called  # no tackle scatter
        assert not any(terms.values())
        assert infos[0]['red_card']['player'] == 1
    else:
        assert result['success'] == success
        assert result['status'] == ('success' if success else 'failed')
        assert env.last_foul is None
        assert env.ball.carrier_id == (None if success else target)
        assert not env.agents[1].suspended


@pytest.mark.parametrize('card_draw,card', [(0.1, True), (0.9, False)])
def test_aggressive_off_ball_tackle_is_always_foul(card_draw, card):
    env, actions, target = arrange(off_ball=True)
    env.np_random = Mock(random=Mock(return_value=card_draw))
    _, rewards, *_ = env.step(actions)
    assert env.np_random.random.call_count == 1
    assert env.last_foul is not None
    assert env.last_tackle_results[1]['red_card'] == card
    assert env.last_tackle_results[1]['success'] is None
    assert env.agents[1].suspended == card
    assert env.ball.carrier_id == target
    assert set(rewards.values()) == {0}


def test_two_player_card_awards_only_one_goal():
    env, actions, _ = arrange(size=2)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.1]))
    env.step(actions)
    assert env.team_scores == {0: 0, 1: 1}
    assert env.last_restart['type'] == 'goal_by_forfait'
    assert all(not a.suspended for a in env.agents)
    assert env.np_random.random.call_count == 2


def test_invalid_aggressive_attempt_does_not_roll_card():
    env, actions, _ = arrange()
    env.agents[1].state.pos = (3, 5)
    env.np_random = Mock()
    env.step(actions)
    assert env.last_tackle_results[1]['status'] == 'invalid_distance'
    assert not env.np_random.random.called
    assert env.last_red_card is None


def test_normal_tackle_keeps_fifty_percent_and_no_card_roll():
    env, actions, target = arrange()
    actions[1]['type'] = int(SoccerActionType.TACKLE)
    env.np_random = Mock(random=Mock(return_value=0.6))
    env.step(actions)
    assert env.np_random.random.call_count == 1
    assert env.last_tackle_results[1]['status'] == 'failed'
    assert env.last_tackle_results[1]['red_card'] is False
    assert env.ball.carrier_id == target
