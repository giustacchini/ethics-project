from gymnasium import spaces

from enums import SoccerActionType, MovementActions

from mosaic_multigrid.core.agent import Agent
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

    # NOTE: drawing used to live here (Agent.render, called per tile by SoccerGrid).
    # It now lives in soccer/rendering/pitch.py, which draws goalkeeper = square and
    # field player = triangle in the original Mosaic style.
