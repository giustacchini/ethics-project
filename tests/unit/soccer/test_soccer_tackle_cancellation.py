from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv


def arrange(aggressive=False, recipient_moves=False):
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[1].state.pos = (5, 5)
    env.agents[4].state.pos = (6, 5)
    env.agents[5].state.pos = (5, 4)
    env.ball = CustomBall(position=(5, 5), carrier_id=1, last_possessing_team=0)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = MovementActions.RIGHT
    actions[5] = dict(type=int(SoccerActionType.AGGRESSIVE_TACKLE if aggressive else SoccerActionType.TACKLE),
                      movement=0, target=(3, 3), strength=0, target_player_id=1)
    if recipient_moves:
        actions[4] = MovementActions.RIGHT
    return env, actions


@pytest.mark.parametrize('aggressive', [False, True])
@pytest.mark.parametrize('recipient_moves', [False, True])
def test_transfer_cancels_tackle_without_failure_or_scatter(aggressive, recipient_moves):
    env, actions = arrange(aggressive, recipient_moves)
    env.np_random = Mock(random=Mock(side_effect=[0.1, 0.9] if aggressive else []))
    obs, rewards, *_ = env.step(actions)
    result = env.last_tackle_results[5]
    assert result['status'] == 'cancelled_possession_changed'
    assert result['success'] is None
    assert result['new_carrier'] == 4
    assert 'ball_position' not in result
    assert env.ball.carrier_id == 4
    assert env.ball.position == ((7, 5) if recipient_moves else (6, 5))
    assert env.agents[1].state.pos == (5, 5)
    assert env.last_possession_transfer == dict(from_agent=1, to_agent=4, from_team=0, to_team=1)
    assert env.np_random.random.call_count == (2 if aggressive else 0)
    assert not env.np_random.integers.called
    assert set(rewards.values()) == {0}
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize('success_draw', [0.1, 0.9])
def test_aggressive_card_interrupts_before_movement_transfer(success_draw):
    env, actions = arrange(aggressive=True, recipient_moves=True)
    env.np_random = Mock(random=Mock(side_effect=[success_draw, 0.1]))
    env.step(actions)
    assert env.agents[5].suspended
    assert env.last_tackle_results[5]['status'] == 'foul'
    assert env.last_tackle_results[5]['success'] is None
    assert env.last_possession_transfer is None
    assert env.last_restart['type'] == 'free_kick'
    assert env.ball.carrier_id == 1
    assert env.ball.position == (5, 5)
    assert not env.np_random.integers.called
