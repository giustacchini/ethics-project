import enum
from enum import Enum, auto


class MovementActions(enum.Enum):
    DO_NOTHING = (0, 0)
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    UP_RIGHT = (1, -1)
    UP_LEFT = (-1, -1)
    DOWN_RIGHT = (1, 1)
    DOWN_LEFT = (-1, 1)


class EventType(Enum):
    MOVED = auto()
    INVALID_MOVE = auto()
    AGENT_COLLISION = auto()
    DO_NOTHING = auto()


class SnowplowEvent(Enum):
    PARTIALLY_CLEARED = auto()
    SECONDARY_STREET_CLEARED = auto()
    MAIN_STREET_CLEARED = auto()
    ALL_CLEAN = auto()

class SoccerEvent(Enum):
    MATCH_WIN = auto()
    MATCH_LOSS = auto()
    MATCH_DRAW = auto()
    GOAL_SCORED = auto()
    GOAL_CONCEDED = auto()
    PASS_COMPLETED = auto()
    POSSESSION_RECOVERED = auto()
    POSSESSION_LOST = auto()
