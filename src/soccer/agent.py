import numpy as np
from numpy.typing import NDArray as ndarray
from gymnasium import spaces

from enums import SoccerActionType, MovementActions

from mosaic_multigrid.core.world_object import Ball
from mosaic_multigrid.core.agent import Agent
from mosaic_multigrid.utils.rendering import point_in_rect, point_in_triangle, point_in_circle, fill_coords, rotate_fn
from enum import Enum

class SoccerAgentRole(Enum):
    GOALKEEPER = 0
    FIELD_PLAYER = 1

class SoccerAgent(Agent):
    def __init__(self, index, team_index, original_role, current_role = None, n_players=6):
        super().__init__(index=index, team_index=team_index)
        self.original_role = original_role
        self.suspended = False
        if current_role is not None:
            self.current_role = current_role
        else:
            self.current_role = original_role
        self.action_space = spaces.Dict({
            "type": spaces.Discrete(len(SoccerActionType)),
            "movement": spaces.Discrete(len(MovementActions)),
            # Encoded 0..6 maps to dx/dy -3..3; 0..1 maps to strength 1..2.
            "target": spaces.MultiDiscrete([7, 7]),
            "strength": spaces.Discrete(2),
            "target_player_id": spaces.Discrete(n_players),
            "contest_intensity": spaces.Discrete(2),  # normal=0, aggressive=1
        })

    def render(self, img: ndarray[np.uint8]) -> None:
        """
        Draw goalkeepers as squares and field players as oriented triangles.
        If carrying a ball, draw a small colored circle on top of the agent.

        Parameters
        ----------
        img : ndarray[np.uint8] of shape (width, height, 3)
            RGB image array to render agent on.
        """
        if self.current_role is SoccerAgentRole.GOALKEEPER:
            tri_fn = point_in_rect(
                0.12, 0.87, 0.12, 0.87
            )
        else:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

        # Rotate the triangle based on the agent's direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * self.state.dir)
        fill_coords(img, tri_fn, self.state.color.rgb())

        # Draw ball-carrying indicator: small circle overlay (half the normal ball size)
        if isinstance(self.state.carrying, Ball):
            # Small circle at center of agent (radius = 0.15, half of ball's 0.3)
            circle_fn = point_in_circle(cx=0.5, cy=0.5, r=0.15)
            # Use the ball's color for the indicator
            ball_color = self.state.carrying.color.rgb()
            fill_coords(img, circle_fn, ball_color)
