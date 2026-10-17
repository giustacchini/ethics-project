import pytest

from enums import MovementActions
from soccer.agent import SoccerAgentRole
from soccer.env import SoccerEnv, SoccerEnvConfig


@pytest.mark.parametrize('size,expected', [
    (2, [(0, 5), (3, 5), (15, 5), (12, 5)]),
    (3, [(0, 5), (3, 3), (3, 7), (15, 5), (12, 3), (12, 7)]),
])
def test_reset_restores_formation_and_roles(size, expected):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    obs, _ = env.reset(seed=0)
    assert len(obs) == size * 2
    assert [a.state.pos for a in env.agents] == expected
    env.agents[1].current_role = SoccerAgentRole.GOALKEEPER
    env.agents[1].state.pos = (1, 4)
    env.reset(seed=0)
    assert env.agents[1].current_role is SoccerAgentRole.FIELD_PLAYER
    assert [a.state.pos for a in env.agents] == expected


def test_goalkeeper_bounds_follow_current_role_and_team():
    env = SoccerEnv()
    env.reset(seed=0)
    for agent in env.agents:
        agent.current_role = SoccerAgentRole.GOALKEEPER
        for x in range(16):
            for y in range(11):
                expected = (0 <= x <= 2 if agent.team_index == 0 else 13 <= x <= 15) and 3 <= y <= 7
                assert env.is_move_allowed(env.grid, agent, agent.state.pos, (x, y)) == expected
    field_player = env.agents[1]
    field_player.current_role = SoccerAgentRole.FIELD_PLAYER
    assert env.is_move_allowed(env.grid, field_player, field_player.state.pos, (15, 10))


def test_step_blocks_keeper_exit_and_allows_diagonal_field_move():
    env = SoccerEnv(max_steps=2)
    env.reset(seed=0)
    env.agents[0].state.pos = (2, 5)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[0] = MovementActions.RIGHT
    actions[1] = MovementActions.DOWN_RIGHT
    env.step(actions)
    assert env.agents[0].state.pos == (2, 5)
    assert env.agents[1].state.pos == (4, 4)
    _, _, _, truncations, _ = env.step(env.action_space.sample())
    assert all(truncations.values())
