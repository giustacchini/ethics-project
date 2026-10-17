"""Match configuration and scenario overrides."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SoccerEnvConfig:
    n_players_per_team: int = 3
    max_steps: int = 1000
    # Explicit positions/None remain available for controlled test scenarios.
    initial_ball_position: tuple[int, int] | Literal['random'] | None = 'random'
    goal_rows: tuple[int, ...] = (4, 5, 6)
    # Reward experiment: 'v1 match outcome only, 'v2' adds goals.
    rewards: str = 'v1'

