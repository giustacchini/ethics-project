from shared.events import Event, EventLog
from enum import Enum, auto
from enums import EventType



def test_event_stores_given_values():
    event = Event(
        type=EventType.MOVED,
        agent_id=0,
    )

    assert event.type == EventType.MOVED
    assert event.agent_id == 0


def test_event_uses_default_values():
    event = Event(type=EventType.MOVED)

    assert event.agent_id is None
    assert event.team_id is None
    assert event.data == {}


def test_event_stores_team_id():
    event = Event(
        type=EventType.MOVED,
        agent_id=0,
        team_id=1,
    )

    assert event.team_id == 1


def test_event_stores_extra_data():
    event = Event(
        type=EventType.AGENT_COLLISION,
        agent_id=0,
        data={"other_agent_id": 2},
    )

    assert event.data == {"other_agent_id": 2}


def test_each_event_has_its_own_data():
    first_event = Event(type=EventType.MOVED, agent_id=0)
    second_event = Event(type=EventType.MOVED, agent_id=1)

    first_event.data["position"] = (2, 3)

    assert second_event.data == {}


def test_emit_stores_fields():
    log = EventLog()
    log.emit(
        EventType.INVALID_MOVE, agent_id=0, team_id=1, step=5, reason="out_of_bounds"
    )
    e = log.events()[0]
    assert e.type == EventType.INVALID_MOVE
    assert e.agent_id == 0
    assert e.team_id == 1
    assert e.step == 5
    assert e.data == {"reason": "out_of_bounds"}


def test_clear_empties_log():
    log = EventLog()
    log.emit(EventType.MOVED, agent_id=0)
    log.clear()
    assert log.events() == []


def test_events_returns_copy():
    log = EventLog()
    log.emit(EventType.MOVED, agent_id=0)
    log.events().clear()
    assert len(log.events()) == 1


def test_new_log_is_empty():
    assert EventLog().events() == []


def test_emit_keeps_order():
    log = EventLog()
    log.emit(EventType.MOVED, agent_id=0)
    log.emit(EventType.INVALID_MOVE, agent_id=1)
    types = [e.type for e in log.events()]
    assert types == [EventType.MOVED, EventType.INVALID_MOVE]


class FakeDomainEvent(Enum):
    CLEARED_SNOW = auto()


def test_log_accepts_domain_enum():
    log = EventLog()
    log.emit(FakeDomainEvent.CLEARED_SNOW, agent_id=0, amount=2)
    e = log.events()[0]
    assert e.type == FakeDomainEvent.CLEARED_SNOW
    assert e.data == {"amount": 2}
