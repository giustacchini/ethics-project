from dataclasses import dataclass, field
from typing import Any

from mosaic_multigrid.base import AgentID
from enum import Enum


@dataclass(frozen=True)
class Event:
    """A record of something that happened in the game.

    When an agent does something (moves, makes an invalid move,
    bumps into another agent), the environment creates an Event
    and saves it in an EventLog. The reward code then looks at
    these events to decide which rewards to give.

    Fields:
        type: What happened (an EventType, e.g. MOVED).
        agent_id: Which agent it happened to, if any.
        team_id: Which team it happened to, if any.
        step: The step number when it happened.
        data: Any extra details about the event.
    """
    type: Enum
    agent_id: AgentID | None = None
    team_id: int | None = None
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)


class EventLog:
    """A list of everything that has happened in the game.

    Use emit() to add an event, events() to read them all,
    and clear() to empty the list, for example before a new step.
    """
    def __init__(self) -> None:
        self._events: list[Event] = []

    def emit(
        self,
        type: Enum,
        agent_id: AgentID | None = None,
        team_id: int | None = None,
        step: int = 0,
        **data: Any,
    ) -> None:
        self._events.append(
            Event(type=type, agent_id=agent_id, team_id=team_id, step=step, data=data)
        )

    def events(self) -> list[Event]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
