import numpy as np

from mosaic_multigrid.core.grid import Grid
from soccer.agent import SoccerAgentRole
from soccer.env import SoccerEnv, SoccerEnvConfig
from soccer.grid import SoccerGrid
from enums import MovementActions


def test_roles_render_differently_and_role_changes_refresh():
    env = SoccerEnv()
    env.reset(seed=0)
    keeper, field = env.agents[:2]
    keeper.state.dir = field.state.dir = 0
    assert keeper.state.color == field.state.color
    SoccerGrid._tile_cache.clear()
    assert SoccerGrid._tile_cache is not Grid._tile_cache
    image = env.grid.render(tile_size=32, agents=env.agents)
    keeper_tile = image[160:192, 0:32].copy()
    field_tile = image[96:128, 96:128].copy()
    assert not np.array_equal(keeper_tile, field_tile)
    field.current_role = SoccerAgentRole.GOALKEEPER
    image = env.grid.render(tile_size=32, agents=env.agents)
    expected = env.grid.render_tile(
        agent=field, tile_size=32, background=env.grid.background_at(3, 3),
    )
    assert np.array_equal(expected, image[96:128, 96:128])
    field.current_role = SoccerAgentRole.FIELD_PLAYER
    image = env.grid.render(tile_size=32, agents=env.agents)
    assert np.array_equal(field_tile, image[96:128, 96:128])


def test_collected_ball_overlay_changes_carrier_tile():
    env = SoccerEnv(SoccerEnvConfig(initial_ball_position=(4, 3)))
    env.reset(seed=0)
    before = env.grid.render(tile_size=32, agents=env.agents)[96:128, 96:128].copy()
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = MovementActions.RIGHT
    env.step(actions)
    after = env.grid.render(tile_size=32, agents=env.agents)[96:128, 128:160]
    assert not np.array_equal(before, after)
