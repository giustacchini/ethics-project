"""Pitch rendering for ``SoccerEnv`` in the style of Mosaic's original soccer.

WHY THIS FILE EXISTS
--------------------
Mosaic's own soccer environments draw the pitch with ``render_fifa()``
(``mosaic_multigrid.rendering.fifa``): striped grass, white markings, net-style
goals, a red ball. We cannot call that function directly because it assumes
Mosaic's own game layout:

* it expects ``env.goal_pos`` / ``env.goal_index`` (our env has ``goal_cells``),
* it treats the outer ring of cells as a decorative wall (our pitch uses the
  *whole* 16x11 grid as playable, with goals on columns 0 and 15),
* it expects team ids 1 and 2 (ours are 0 and 1),
* it draws every player as a triangle (we want goalkeeper = square).

So this module re-implements the same *look* on top of OUR geometry. It only
reads the environment; it never changes game state, so game logic is untouched.

HOW IT IS USED NOW
------------------
``SoccerEnv.get_full_render`` calls :func:`render_soccer_frame`. Mosaic's
``get_frame()`` / ``render()`` (human window and ``rgb_array``) go through that
hook, so ``render_mode='human'`` and ``'rgb_array'`` both use this renderer.
:func:`save_frame` is a small helper to dump a PNG (e.g. for the report).

Local-view (field-of-view) overlays from the original are intentionally NOT
drawn: our observations are the full structured state, not a local image.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.freetype
from mosaic_multigrid.core.world_object import Ball

from soccer.agent import SoccerAgentRole

# ---------------------------------------------------------------------------
# Palette (copied from mosaic_multigrid.rendering.fifa so the look matches, but
# kept here so we do not depend on that module's private internals).
# ---------------------------------------------------------------------------
GRASS_LIGHT = (76, 153, 60)
GRASS_DARK = (58, 128, 45)
FIELD_LINE = (255, 255, 255)
GOAL_NET = (220, 220, 220)
BALL_COLOR = (255, 60, 60)
BALL_OUTLINE = (180, 30, 30)

# Our team ids are 0 (left side) and 1 (right side).
# Team 0 is green and team 1 is blue, exactly like Mosaic's green/blue teams.
TEAM_COLORS = {0: (30, 200, 60), 1: (60, 80, 220)}
TEAM_OUTLINE = {0: (15, 120, 30), 1: (30, 40, 140)}
FALLBACK_COLOR = (200, 200, 200)
FALLBACK_OUTLINE = (100, 100, 100)

# Goalkeeper area in grid cells. This is only a *drawing* of the rule enforced
# in ``SoccerEnv.is_move_allowed`` (keeper stays in 3 columns x rows 3..7 next
# to its own goal). A unit test checks the two stay in sync.
KEEPER_AREA_COLS = 3
KEEPER_AREA_ROWS = (3, 7)  # inclusive


def keeper_area_cells(team_index: int, width: int):
    """Cells of the goalkeeper area of a team (used by the drawing and test)."""
    x_min, x_max = (0, KEEPER_AREA_COLS - 1) if team_index == 0 else (
        width - KEEPER_AREA_COLS, width - 1)
    return {(x, y) for x in range(x_min, x_max + 1)
            for y in range(KEEPER_AREA_ROWS[0], KEEPER_AREA_ROWS[1] + 1)}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_grass(surface, tile, gw, gh):
    """Alternating vertical grass stripes plus thin black cell lines."""
    for col in range(gw):
        color = GRASS_LIGHT if col % 2 == 0 else GRASS_DARK
        pygame.draw.rect(surface, color, (col * tile, 0, tile, gh * tile))
    # Cell lines make the discrete grid readable (the game is cell-based).
    for col in range(1, gw):
        pygame.draw.line(surface, (0, 0, 0), (col * tile, 0), (col * tile, gh * tile), 1)
    for row in range(1, gh):
        pygame.draw.line(surface, (0, 0, 0), (0, row * tile), (gw * tile, row * tile), 1)


def _draw_markings(surface, tile, gw, gh):
    """White pitch markings laid over the FULL grid (the whole grid is playable)."""
    lw = max(2, tile // 16)
    width_px, height_px = gw * tile, gh * tile
    cx, cy = width_px // 2, height_px // 2

    # Touchlines / goal lines on the very edge of the grid.
    # Drawn with width lw*2 at the edge so half of it stays visible (pygame clips
    # anything outside the surface).
    pygame.draw.rect(surface, FIELD_LINE, (0, 0, width_px, height_px), lw * 2)
    # Halfway line: passes between columns 7 and 8, where the ball is placed
    # at kick-off (cells (7, 5) or (8, 5)).
    pygame.draw.line(surface, FIELD_LINE, (cx, 0), (cx, height_px), lw)
    # Centre circle and spot.
    pygame.draw.circle(surface, FIELD_LINE, (cx, cy), int(tile * 2.1), lw)
    pygame.draw.circle(surface, FIELD_LINE, (cx, cy), max(3, tile // 8))

    # Penalty-area style boxes = the goalkeeper areas (see KEEPER_AREA_*).
    top = KEEPER_AREA_ROWS[0] * tile
    box_h = (KEEPER_AREA_ROWS[1] - KEEPER_AREA_ROWS[0] + 1) * tile
    box_w = KEEPER_AREA_COLS * tile
    pygame.draw.rect(surface, FIELD_LINE, (0, top, box_w, box_h), lw)
    pygame.draw.rect(surface, FIELD_LINE, (width_px - box_w, top, box_w, box_h), lw)


def _draw_goals(surface, tile, env):
    """Goal cells as team-coloured boxes with a net pattern (as in Mosaic)."""
    for team, cells in env.goal_cells.items():
        color = TEAM_COLORS.get(team, FALLBACK_COLOR)
        for gx, gy in sorted(cells):
            rect = pygame.Rect(gx * tile + 2, gy * tile + 2, tile - 4, tile - 4)
            tint = pygame.Surface((tile - 4, tile - 4), pygame.SRCALPHA)
            tint.fill((*color, 80))
            surface.blit(tint, (gx * tile + 2, gy * tile + 2))
            for ly in range(4, tile - 4, max(3, tile // 8)):
                pygame.draw.line(surface, GOAL_NET, (gx * tile + 4, gy * tile + ly),
                                 (gx * tile + tile - 4, gy * tile + ly), 1)
            pygame.draw.rect(surface, color, rect, max(2, tile // 12))


def _draw_ball(surface, tile, x, y):
    """Loose / in-flight ball: red circle with shadow and highlight."""
    cx, cy = x * tile + tile // 2, y * tile + tile // 2
    radius = int(tile * 0.25)
    shadow = pygame.Surface((tile, tile), pygame.SRCALPHA)
    pygame.draw.circle(shadow, (0, 0, 0, 50), (tile // 2 + 1, tile // 2 + 1), radius)
    surface.blit(shadow, (x * tile, y * tile))
    pygame.draw.circle(surface, BALL_COLOR, (cx, cy), radius)
    pygame.draw.circle(surface, BALL_OUTLINE, (cx, cy), radius, max(1, tile // 16))
    pygame.draw.circle(surface, (255, 180, 180),
                       (cx - radius // 3, cy - radius // 3), max(1, radius // 3))


def _draw_player(surface, tile, agent):
    """Goalkeeper = square, field player = triangle pointing where it faces.

    The shape follows ``current_role`` (not ``original_role``) so a field
    player temporarily acting as goalkeeper is drawn as a square.
    """
    x, y = int(agent.state.pos[0]), int(agent.state.pos[1])
    cx, cy = x * tile + tile // 2, y * tile + tile // 2
    size = tile * 0.40
    team = agent.team_index
    color = TEAM_COLORS.get(team, FALLBACK_COLOR)
    outline = TEAM_OUTLINE.get(team, FALLBACK_OUTLINE)
    carrying = agent.state.carrying is not None

    if agent.current_role is SoccerAgentRole.GOALKEEPER:
        half = size * 0.85
        points = [(cx - half, cy - half), (cx + half, cy - half),
                  (cx + half, cy + half), (cx - half, cy + half)]
    else:
        # Triangle facing RIGHT, then rotated by the facing direction
        # (0=right, 1=down, 2=left, 3=up).
        angle = int(agent.state.dir) * math.pi / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        base = [(cx + size, cy), (cx - size * 0.6, cy - size * 0.7),
                (cx - size * 0.6, cy + size * 0.7)]
        points = [(cx + (px - cx) * cos_a - (py - cy) * sin_a,
                   cy + (px - cx) * sin_a + (py - cy) * cos_a) for px, py in base]

    # Soft shadow, drawn on a tile-sized layer so it never leaks to neighbours.
    shadow = pygame.Surface((tile, tile), pygame.SRCALPHA)
    pygame.draw.polygon(shadow, (0, 0, 0, 50),
                        [(px + 2 - x * tile, py + 2 - y * tile) for px, py in points])
    surface.blit(shadow, (x * tile, y * tile))

    if carrying:  # yellow glow around the ball carrier
        glow = pygame.Surface((tile, tile), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 255, 100, 60), (tile // 2, tile // 2), int(size + 4))
        surface.blit(glow, (x * tile, y * tile))

    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, outline, points, max(2, tile // 14))

    if carrying:  # small ball on top of the carrier
        ball_r = max(3, tile // 8)
        pygame.draw.circle(surface, BALL_COLOR, (cx, cy), ball_r)
        pygame.draw.circle(surface, BALL_OUTLINE, (cx, cy), ball_r, 1)


_FONTS = {}


def _draw_label(surface, tile, agent):
    """Agent index in white, centred on the player."""
    size = max(9, tile // 3)
    if size not in _FONTS:
        _FONTS[size] = pygame.freetype.SysFont('monospace', size)
    font = _FONTS[size]
    text = str(agent.index)
    rect = font.get_rect(text)
    x, y = int(agent.state.pos[0]), int(agent.state.pos[1])
    rect.center = (x * tile + tile // 2, y * tile + tile // 2)
    font.render_to(surface, rect.topleft, text, fgcolor=(255, 255, 255))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_soccer_frame(env, tile_size: int = 32) -> np.ndarray:
    """Render the current match state to an RGB image of shape (H, W, 3).

    Suspended (red-carded) and terminated players are not drawn, matching the
    previous renderer. Only reads from ``env``.
    """
    if not pygame.get_init():
        pygame.init()
    if not pygame.freetype.get_init():
        pygame.freetype.init()

    gw, gh = env.width, env.height
    surface = pygame.Surface((gw * tile_size, gh * tile_size))

    _draw_grass(surface, tile_size, gw, gh)
    _draw_markings(surface, tile_size, gw, gh)
    _draw_goals(surface, tile_size, env)

    # Ball on the ground (loose or in flight). A carried ball is not in the
    # grid; it is drawn on its carrier by _draw_player instead.
    for x in range(gw):
        for y in range(gh):
            obj = env.grid.get(x, y)
            # isinstance (not obj.type) because our CustomBall has its own type tag.
            if isinstance(obj, Ball):
                _draw_ball(surface, tile_size, x, y)

    visible = [a for a in env.agents
               if not getattr(a, 'suspended', False) and not a.state.terminated]
    for agent in visible:
        _draw_player(surface, tile_size, agent)
    for agent in visible:
        _draw_label(surface, tile_size, agent)

    # pygame arrays are (W, H, 3); Mosaic expects (H, W, 3).
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


def save_frame(env, path: str, tile_size: int = 48) -> None:
    """Save the current frame as a PNG (handy for README / report figures)."""
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(render_soccer_frame(env, tile_size))).save(path)
