# python
# snow_plow_env.py

from mosaic_multigrid.base import AgentID, MultiGridEnv
from mosaic_multigrid.core.grid import Grid

from Enums import MovementActions
from shared.entities import Obstacle
from shared.movement import handle_simultaneous_movement


class SnowPlowEnv(MultiGridEnv):
    """
    `is_move_allowed` is the extension point issue #6 (road network and
    traffic rules) will fill in, RIGHT NOW  it currently allows every move that
    passes the shared grid/bounds checks
    """

    def __init__(self, agents, width, height, max_steps):
        super().__init__(agents=agents, width=width, height=height, max_steps=max_steps)
        self._gen_grid(width, height)

    def _gen_grid(self, width: int, height: int):
        # Required by MultiGridEnv.reset(): must set self.grid and give
        # every agent a valid (non-negative) position and direction
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for agent in self.agents:
            self.place_agent(agent)
         
    def get_observable_grid(self):
        """Grid that the agents may see."""
        return self.grid.state.copy()
    
    def is_move_allowed(self, grid, agent, current_pos, intended_pos) -> bool:
        """
        Domain-specific movement hook.
        Issue #6
        overrides this to reject wrong-direction moves on one-way lanes.
        """
        return True

    def handle_actions(self, actions: dict[AgentID, MovementActions]):
        movement_results = handle_simultaneous_movement(
            agents=self.agents,
            grid=self.grid,
            actions=actions,
            width=self.width,
            height=self.height,
            is_move_allowed=self.is_move_allowed,
        )
        self.last_movement_results = movement_results

        # Placeholder rewards -- actual SnowPlow rewards are for future issues
        rewards = {agent.index: 0 for agent in self.agents}
        return rewards
