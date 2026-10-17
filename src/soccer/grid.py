from mosaic_multigrid.core import Grid

import numpy as np

from collections import defaultdict
from numpy.typing import NDArray as ndarray
from typing import Any, Iterable

from mosaic_multigrid.core.agent import Agent
from mosaic_multigrid.core.constants import TILE_PIXELS
from mosaic_multigrid.core.world_object import WorldObj

from mosaic_multigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
)


class SoccerGrid(Grid):
    SURFACE_COLOR = (124, 191, 1)
    GOALKEEPER_COLOR = (78, 198, 57)
    GOAL_COLOR = (21, 178, 31)
    # Keep soccer-specific keys separate from Mosaic's global tile cache.
    _tile_cache = {}

    def __init__(self, width, height, goal_rows=(4, 5, 6)):
        super().__init__(width, height)
        self.goal_rows = frozenset(goal_rows)

    def background_at(self, x, y):
        if x in (0, self.width - 1) and y in self.goal_rows:
            return self.GOAL_COLOR
        if (x <= 2 or x >= self.width - 3) and 3 <= y <= 7:
            return self.GOALKEEPER_COLOR
        return self.SURFACE_COLOR

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None = None,
        agent: Agent | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
        background: tuple[int, int, int] = SURFACE_COLOR,
    ) -> ndarray[np.uint8]:
        """
        Render a tile and cache the result.

        Parameters
        ----------
        obj : WorldObj or None
            Object to render.
        agent : Agent or None
            Agent to render.
        highlight : bool
            Whether to highlight the tile.
        tile_size : int
            Tile size in pixels.
        subdivs : int
            Downsampling factor for supersampling / anti-aliasing.
        """
        # Build a hashable cache key
        key: tuple[Any, ...] = (highlight, tile_size, subdivs, background)
        if agent is not None:
            # Include carrying info so the ball overlay is not skipped by cache
            carrying_key = agent.state.carrying.encode() if agent.state.carrying is not None else None
            key += (
                type(agent), getattr(agent, 'current_role', None),
                agent.state.color, agent.state.dir, carrying_key,
                bool(agent.state.terminated),
            )
        else:
            key += (None,) * 6
        key = (type(obj), obj.encode() if obj is not None else None, key)

        if key in cls._tile_cache:
            return cls._tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            dtype=np.uint8,
        )
        img[:] = background

        # Grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Draw the object
        if obj is not None:
            obj.render(img)

        # Draw the agent
        if agent is not None and not agent.state.terminated:
            agent.render(img)

        # Highlight
        if highlight:
            highlight_img(img)

        # Anti-alias via downsampling
        img = downsample(img, subdivs)

        cls._tile_cache[key] = img
        return img

    def render(
        self,
        tile_size: int,
        agents: Iterable[Agent] = (),
        highlight_mask: ndarray[np.bool_] | None = None,
    ) -> ndarray[np.uint8]:
        """
        Render the entire grid at a given scale.

        Parameters
        ----------
        tile_size : int
            Tile size in pixels.
        agents : Iterable[Agent]
            Agents to overlay on the grid.
        highlight_mask : ndarray[bool] of shape (width, height) or None
            Boolean mask indicating which cells to highlight.
        """
        if highlight_mask is None:
            highlight_mask = np.zeros(
                shape=(self.width, self.height), dtype=bool)

        # Build agent-location lookup (non-terminated agents get priority)
        location_to_agent = defaultdict(type(None))
        for agent in sorted(agents, key=lambda a: not a.terminated):
            if getattr(agent, 'suspended', False):
                continue
            location_to_agent[tuple(agent.pos)] = agent

        # Initialize pixel array
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render each cell
        for j in range(self.height):
            for i in range(self.width):
                cell = self.get(i, j)
                tile_img = self.render_tile(
                    cell,
                    agent=location_to_agent[i, j],
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    background=self.background_at(i, j),
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
