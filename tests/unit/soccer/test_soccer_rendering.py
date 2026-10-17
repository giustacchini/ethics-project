"""Tests for the Mosaic-style renderer in soccer/rendering.py.

They compare rendered pixels of single tiles, so no window or display is needed.
"""

import numpy as np
import gymnasium as gym

import soccer  # noqa: F401  (registers the EthicsSoccer-* ids)
from enums import MovementActions
from soccer.agent import SoccerAgentRole
from soccer.env import SoccerEnv, SoccerEnvConfig
from soccer.rendering import (
    BALL_COLOR, TEAM_COLORS, keeper_area_cells, render_soccer_frame,
)

TILE = 32


def tile_at(image, x, y):
    return image[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE]


def frame(env):
    return render_soccer_frame(env, TILE)


def has_color(tile, rgb):
    return bool(np.any(np.all(tile == rgb, axis=-1)))


def test_frame_has_full_pitch_shape_and_dtype():
    env = SoccerEnv()
    env.reset(seed=0)
    image = frame(env)
    assert image.shape == (env.height * TILE, env.width * TILE, 3)
    assert image.dtype == np.uint8


def test_get_frame_uses_new_renderer():
    # Mosaic's get_frame/render go through SoccerEnv.get_full_render.
    env = SoccerEnv()
    env.reset(seed=0)
    assert np.array_equal(env.get_frame(highlight=False, tile_size=TILE), frame(env))


def test_goalkeeper_square_and_field_player_triangle_differ():
    env = SoccerEnv()
    env.reset(seed=0)
    keeper, field = env.agents[:2]
    keeper.state.dir = field.state.dir = 0
    image = frame(env)
    keeper_tile = tile_at(image, 0, 5).copy()
    field_tile = tile_at(image, 3, 3).copy()
    assert not np.array_equal(keeper_tile, field_tile)
    # Same team -> same body colour in both shapes.
    assert has_color(keeper_tile, TEAM_COLORS[0]) and has_color(field_tile, TEAM_COLORS[0])


def test_shape_follows_current_role_and_refreshes():
    env = SoccerEnv()
    env.reset(seed=0)
    field = env.agents[1]
    before = tile_at(frame(env), 3, 3).copy()
    field.current_role = SoccerAgentRole.GOALKEEPER
    as_keeper = tile_at(frame(env), 3, 3).copy()
    assert not np.array_equal(before, as_keeper)
    field.current_role = SoccerAgentRole.FIELD_PLAYER
    assert np.array_equal(before, tile_at(frame(env), 3, 3))


def test_triangle_rotates_with_facing_direction():
    env = SoccerEnv()
    env.reset(seed=0)
    field = env.agents[1]
    tiles = []
    for direction in range(4):
        field.state.dir = direction
        tiles.append(tile_at(frame(env), 3, 3).copy())
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(tiles[i], tiles[j])


def test_teams_use_different_colors():
    env = SoccerEnv()
    env.reset(seed=0)
    image = frame(env)
    assert has_color(tile_at(image, 3, 3), TEAM_COLORS[0])
    assert has_color(tile_at(image, 12, 3), TEAM_COLORS[1])


def test_loose_ball_is_drawn_and_carried_ball_moves_to_carrier():
    env = SoccerEnv(SoccerEnvConfig(initial_ball_position=(4, 3)))
    env.reset(seed=0)
    assert has_color(tile_at(frame(env), 4, 3), BALL_COLOR)
    before = tile_at(frame(env), 4, 3).copy()
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = MovementActions.RIGHT  # player 1 walks onto the ball
    env.step(actions)
    after = tile_at(frame(env), 4, 3)
    assert env.agents[1].state.carrying is not None
    assert not np.array_equal(before, after)  # carrier overlay now drawn here
    assert has_color(after, BALL_COLOR)       # small ball on the carrier


def test_suspended_player_is_not_drawn():
    env = SoccerEnv()
    env.reset(seed=0)
    with_player = tile_at(frame(env), 3, 3).copy()
    env.agents[1].suspended = True
    assert not np.array_equal(with_player, tile_at(frame(env), 3, 3))
    assert not has_color(tile_at(frame(env), 3, 3), TEAM_COLORS[0])


def test_rendering_does_not_change_game_state():
    env = SoccerEnv()
    env.reset(seed=0)
    before = (env.get_observable_grid().copy(), [tuple(a.state.pos) for a in env.agents],
              env.ball.position, dict(env.team_scores))
    frame(env)
    after = (env.get_observable_grid(), [tuple(a.state.pos) for a in env.agents],
             env.ball.position, dict(env.team_scores))
    assert np.array_equal(before[0], after[0]) and before[1:] == after[1:]


def test_drawn_keeper_area_matches_movement_rule():
    # The white box is drawn from keeper_area_cells(); it must equal the cells
    # a goalkeeper may actually enter according to SoccerEnv.is_move_allowed.
    env = SoccerEnv()
    env.reset(seed=0)
    for team, keeper in ((0, env.agents[0]), (1, env.agents[3])):
        allowed = {(x, y) for x in range(env.width) for y in range(env.height)
                   if env.is_move_allowed(env.grid, keeper, keeper.state.pos, (x, y))}
        assert allowed == keeper_area_cells(team, env.width)


def test_goal_cells_are_tinted_in_defending_team_color():
    env = SoccerEnv()
    env.reset(seed=0)
    image = frame(env)
    # Goal cells at rows 4 and 6 are empty at kick-off, so the net is visible.
    assert not np.array_equal(tile_at(image, 0, 4), tile_at(image, 1, 4))
    assert not np.array_equal(tile_at(image, 15, 6), tile_at(image, 14, 6))


def test_gym_make_ids_create_working_envs():
    for env_id, team_size in (("EthicsSoccer-3v3-v0", 3), ("EthicsSoccer-2v2-v0", 2)):
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=0)
        soccer_env = env.unwrapped
        assert len(soccer_env.agents) == 2 * team_size
        assert env.render().shape == (11 * 32, 16 * 32, 3)
        env.close()
