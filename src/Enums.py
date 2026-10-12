import enum
from enum import Enum, auto

class MovementActions(enum.Enum):
    DO_NOTHING = (0, 0)
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

class EventType(Enum):
    MOVED = auto()
    INVALID_MOVE = auto()
    AGENT_COLLISION = auto()

