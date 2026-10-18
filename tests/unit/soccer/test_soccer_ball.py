import pytest

from enums import MovementActions
from soccer.entities.ball import CustomBall
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


def test_explicitly_disabled_ball_for_controlled_scenarios():
    env = SoccerEnv(SoccerEnvConfig(initial_ball_position=None))
    env.reset(seed=0)
    assert env.ball is None


def test_collection_clears_grid_and_carried_ball_follows():
    env = SoccerEnv(SoccerEnvConfig(initial_ball_position=(4, 3)))
    env.reset(seed=0)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = MovementActions.RIGHT
    env.step(actions)
    assert env.ball.carrier_id == 1
    assert env.ball.last_possessing_team == 0
    assert env.grid.get(4, 3) is None
    assert env.agents[1].state.carrying is not None
    env.step(actions)
    assert env.ball.position == (5, 3)
    env.reset(seed=0)
    assert env.ball.carrier_id is None
    assert env.grid.get(4, 3) is not None
    assert all(a.state.carrying is None for a in env.agents)


@pytest.mark.parametrize('opponent_moves', [False, True])
def test_dribbling_into_opponent_transfers_to_resolved_position(opponent_moves):
    env = SoccerEnv()
    env.reset(seed=0)
    carrier, opponent = env.agents[1], env.agents[4]
    opponent.state.pos = (4, 3)
    env.ball = CustomBall(position=(3, 3), carrier_id=1, last_possessing_team=0)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = MovementActions.RIGHT
    if opponent_moves:
        actions[4] = MovementActions.RIGHT
    env.step(actions)
    assert carrier.state.pos == (3, 3)
    assert env.ball.carrier_id == 4
    assert env.ball.position == opponent.state.pos == ((5, 3) if opponent_moves else (4, 3))
    assert env.ball.last_possessing_team == 1
    assert carrier.state.carrying is None
    assert opponent.state.carrying is not None
    assert env.last_possession_transfer == dict(from_agent=1, to_agent=4, from_team=0, to_team=1)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    env.step(actions)
    assert env.last_possession_transfer is None


def test_teammate_blocks_dribble_without_transfer():
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[2].state.pos = (4, 3)
    env.ball = CustomBall(position=(3, 3), carrier_id=1, last_possessing_team=0)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = actions[2] = MovementActions.RIGHT
    env.step(actions)
    assert env.agents[2].state.pos == (5, 3)
    assert env.ball.position == (3, 3)
    assert env.ball.carrier_id == 1
    assert env.last_possession_transfer is None


def test_invalid_goalkeeper_move_does_not_transfer_possession():
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[0].state.pos = (2, 5)
    env.agents[4].state.pos = (3, 5)
    env.ball = CustomBall(position=(2, 5), carrier_id=0, last_possessing_team=0)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[0] = MovementActions.RIGHT
    env.step(actions)
    assert env.ball.carrier_id == 0
    assert env.ball.position == (2, 5)
    assert env.last_possession_transfer is None
