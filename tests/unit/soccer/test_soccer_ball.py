import pytest

from Enums import MovementActions
from soccer.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def test_ball_defaults_and_independent_trajectories():
    ball = CustomBall(position=(7, 5))
    other = CustomBall(position=(8, 5))
    assert ball.flight_state == 'loose'
    assert ball.carrier_id is None
    assert ball.last_possessing_team is None
    assert ball.speed == 0
    ball.remaining_trajectory.append((8, 5))
    assert ball.flight_state == 'in_flight'
    assert other.remaining_trajectory == []


@pytest.mark.parametrize('blocked', [False, True])
def test_ball_follows_committed_carrier_position(blocked):
    env = SoccerEnv()
    env.reset(seed=0)
    carrier = env.agents[1]
    env.ball = CustomBall(position=carrier.state.pos, carrier_id=carrier.index)
    if blocked:
        env.agents[2].state.pos = (4, 3)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[carrier.index] = MovementActions.RIGHT
    env.step(actions)
    expected = (3, 3) if blocked else (4, 3)
    assert env.ball.position == carrier.state.pos == expected
    assert env.ball.carrier_id == carrier.index
    assert env.ball.last_possessing_team == carrier.team_index
    assert env.ball.flight_state == 'possessed'


def test_loose_ball_stays_put_and_reset_clears_possession():
    env = SoccerEnv(SoccerEnvConfig(initial_ball_position=(7, 5)))
    env.reset(seed=0)
    env.step({a.index: MovementActions.DO_NOTHING for a in env.agents})
    assert env.ball.position == (7, 5)
    env.ball.carrier_id = 1
    env.ball.last_possessing_team = 0
    env.reset(seed=0)
    assert env.ball.position == (7, 5)
    assert env.ball.carrier_id is None
    assert env.ball.last_possessing_team is None


def test_unspecified_opening_has_no_ball():
    env = SoccerEnv()
    env.reset(seed=0)
    assert env.ball is None
