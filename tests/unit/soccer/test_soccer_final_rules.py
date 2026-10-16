import pytest

from enums import MovementActions, SoccerActionType
from soccer.ball import CustomBall
from soccer.env import SoccerEnv


def waits(env):
    return {a.index: MovementActions.DO_NOTHING for a in env.active_players}


def kick(dx, dy, strength=2):
    return dict(type=int(SoccerActionType.KICK), movement=0, target=(dx + 3, dy + 3), strength=strength - 1)


def test_random_opening_is_loose_seeded_and_covers_both_cells():
    seen = set()
    env = SoccerEnv()
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        position = env.ball.position
        assert position in ((7, 5), (8, 5))
        assert env.ball.carrier_id is None and env.ball.last_possessing_team is None
        assert all(a.state.pos != position for a in env.agents)
        assert env.observation_space.contains(obs)
        env.reset(seed=seed)
        assert env.ball.position == position
        seen.add(position)
    assert seen == {(7, 5), (8, 5)}


@pytest.mark.parametrize('start,displacement,expected', [
    ((1, 1), (-2, 0), (0, 1)), ((14, 1), (2, 0), (15, 1)),
    ((7, 1), (0, -2), (7, 0)), ((7, 9), (0, 2), (7, 10)),
    ((14, 3), (2, 1), (15, 4)),
])
def test_off_pitch_kick_stops_at_last_in_bounds_cell(start, displacement, expected):
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[1].state.pos = start
    env.ball = CustomBall(position=start, carrier_id=1, last_possessing_team=0)
    actions = waits(env)
    actions[1] = kick(*displacement)
    obs, rewards, *_ = env.step(actions)
    assert env.last_kick_results[1] == 'launched'
    # A goal on the traversed boundary still uses ordinary scoring/reset rules.
    actual = env.last_goal['position'] if env.last_goal else env.ball.position
    assert actual == expected
    assert env.ball.remaining_trajectory == []
    assert env.ball.speed == 0
    assert env.observation_space.contains(obs)


def free_kick_env():
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[1].state.pos = (5, 5)
    env.agents[4].state.pos = (6, 5)
    env.ball = CustomBall(position=(8, 8))
    actions = waits(env)
    actions[1] = dict(type=3, movement=0, target=(3, 3), strength=0, target_player_id=4)
    env.step(actions)
    assert env.free_kick_taker == 4
    return env


def test_free_kick_waits_for_legal_kick_and_blocks_everyone_else():
    env = free_kick_env()
    before = [a.state.pos for a in env.agents]
    actions = {a.index: MovementActions.RIGHT for a in env.active_players}
    obs, _, *_ = env.step(actions)
    assert [a.state.pos for a in env.agents] == before
    assert env.free_kick_taker == 4
    assert obs[0]['free_kick_taker'] == 4
    actions[4] = kick(0, 0)
    env.step(actions)
    assert env.last_kick_results[4] == 'zero_displacement'
    assert env.free_kick_taker == 4
    actions[4] = kick(2, 0)
    obs, _, *_ = env.step(actions)
    assert [a.state.pos for a in env.agents] == before
    assert env.last_kick_results[4] == 'launched'
    assert env.ball.position == (8, 5)
    assert env.free_kick_taker is None
    assert obs[0]['free_kick_taker'] == -1
    actions = waits(env)
    actions[4] = MovementActions.RIGHT
    env.step(actions)
    assert env.agents[4].state.pos == (7, 5)


def test_actions_against_free_kick_taker_cannot_create_another_foul():
    env = free_kick_env()
    env.agents[1].state.pos = (5, 5)
    actions = waits(env)
    actions[1] = dict(type=4, movement=0, target=(3, 3), strength=0, target_player_id=4)
    env.step(actions)
    assert env.last_tackle_results == {}
    assert env.last_red_cards == []
    assert env.free_kick_taker == 4
    env.reset(seed=0)
    assert env.free_kick_taker is None
