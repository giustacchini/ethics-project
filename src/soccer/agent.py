from mosaic_multigrid.core.agent import Agent
from enum import Enum

class SoccerAgentRole(Enum):
    GOALKEEPER = 0
    FIELD_PLAYER = 1

class SoccerAgent(Agent):
    def __init__(self, index, team_index, original_role, current_role = None):
        super().__init__(index=index, team_index=team_index)
        self.original_role = original_role
        if current_role is not None:
            self.current_role = current_role
        else:
            self.current_role = original_role