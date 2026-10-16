import pytest

from soccer.ball import build_trajectory


@pytest.mark.parametrize('start,target,expected', [
    ((3, 3), (5, 3), [(4, 3), (5, 3)]),
    ((3, 3), (3, 5), [(3, 4), (3, 5)]),
    ((3, 3), (5, 5), [(4, 4), (5, 5)]),
    ((5, 5), (3, 3), [(4, 4), (3, 3)]),
    ((3, 3), (6, 4), [(4, 3), (5, 4), (6, 4)]),
    ((3, 3), (3, 3), []),
])
def test_bresenham_trajectory(start, target, expected):
    assert build_trajectory(start, target) == expected

from enums import MovementActions, SoccerActionType
from soccer.agent import SoccerAgentRole
from soccer.ball import CustomBall
from soccer.env import SoccerEnv


def setup_kick(carrier_id=1, position=(5, 5)):
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[carrier_id].state.pos = position
    env.ball = CustomBall(position=position, carrier_id=carrier_id,
                          last_possessing_team=env.agents[carrier_id].team_index)
    return env


def kick(env, agent_id, dx, dy, strength=1, extra_actions=None):
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[agent_id] = dict(type=int(SoccerActionType.KICK), movement=0,
                            target=(dx + 3, dy + 3), strength=strength - 1)
    actions.update(extra_actions or {})
    return env.step(actions)


def wait(env):
    return env.step({a.index: MovementActions.DO_NOTHING for a in env.agents})


def test_same_transition_flight_continues_and_stops_loose():
    env = setup_kick()
    obs, *_ = kick(env, 1, 2, 0)
    assert env.ball.position == (6, 5)
    assert env.ball.remaining_trajectory == [(7, 5)]
    assert env.ball.carrier_id is None
    assert env.agents[1].state.pos == (5, 5)
    assert env.agents[1].state.carrying is None
    assert env.grid.get(6, 5) is not None
    assert obs[0]['ball_kicker'] == 1
    assert env.observation_space.contains(obs)
    wait(env)
    assert env.ball.position == (7, 5)
    assert env.ball.flight_state == 'loose'
    assert env.ball.speed == 0
    assert env.grid.get(6, 5) is None
    assert env.grid.get(7, 5) is not None
    assert env.last_pass_completed is None


@pytest.mark.parametrize('receiver_id,is_pass', [(2, True), (4, False)])
def test_strength_two_checks_intermediate_cell(receiver_id, is_pass):
    env = setup_kick()
    env.agents[receiver_id].state.pos = (6, 5)
    kick(env, 1, 2, 0, strength=2)
    assert env.ball.carrier_id == receiver_id
    assert env.ball.position == (6, 5)
    assert env.ball.remaining_trajectory == []
    assert env.grid.get(6, 5) is None
    assert (env.last_pass_completed is not None) == is_pass
    assert (env.last_possession_transfer is not None) == (not is_pass)
    wait(env)
    assert env.last_pass_completed is None


@pytest.mark.parametrize('role,expected', [
    (SoccerAgentRole.FIELD_PLAYER, 'out_of_range'),
    (SoccerAgentRole.GOALKEEPER, 'launched'),
])
def test_kick_range_uses_current_role(role, expected):
    env = setup_kick(position=(1, 5))
    env.agents[1].current_role = role
    kick(env, 1, 3, 0, strength=2)
    assert env.last_kick_results[1] == expected
    assert env.ball.position == ((3, 5) if expected == 'launched' else (1, 5))


@pytest.mark.parametrize('dx,dy,reason', [
    (0, 0, 'zero_displacement'),
])
def test_invalid_kick_leaves_possession(dx, dy, reason):
    env = setup_kick(position=(1, 5))
    kick(env, 1, dx, dy)
    assert env.last_kick_results[1] == reason
    assert env.ball.carrier_id == 1
    assert env.ball.position == (1, 5)


def test_noncarrier_cannot_kick_ball_collected_this_transition():
    env = setup_kick()
    env.ball.carrier_id = None
    env.agents[2].state.pos = (5, 5)
    env.agents[1].state.pos = (4, 5)
    kick(env, 2, 2, 0)
    assert env.last_kick_results[2] == 'not_carrier'
    assert env.ball.carrier_id == 2
    assert env.ball.position == (5, 5)


def test_receiving_uses_resolved_positions():
    env = setup_kick()
    env.agents[2].state.pos = (6, 5)
    kick(env, 1, 2, 0, strength=2, extra_actions={2: MovementActions.DOWN})
    assert env.ball.position == (7, 5)
    assert env.ball.carrier_id is None
    assert env.last_pass_completed is None


def test_delayed_reception_credits_pass_once():
    env = setup_kick()
    env.agents[2].state.pos = (7, 6)
    kick(env, 1, 2, 0, strength=2)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[2] = MovementActions.UP
    env.step(actions)
    assert env.ball.carrier_id == 2
    assert env.last_pass_completed == dict(from_agent=1, to_agent=2, team=0)
    wait(env)
    assert env.last_pass_completed is None
