from enum import Enum, auto
from shared.events import Event
from Enums import EventType
from shared.rewards import EventRewardCalculator


class DomainEventType(Enum):
    GOAL = auto()
    ALL_CLEAN = auto()


def test_collision_penalty_applies_only_to_given_agent():
    calculator = EventRewardCalculator(
        reward_mapping={EventType.AGENT_COLLISION: -10.0},
    )
    events = [Event(type=EventType.AGENT_COLLISION, agent_id=0, team_id=1)]

    result = calculator.calculate_rewards(events, {0: 1, 1: 1})

    assert result.individual == {0: -10.0, 1: 0.0}
    assert result.shared_by_team == {}


def test_rewards_from_multiple_events_are_added():
    calculator = EventRewardCalculator(
        reward_mapping={
            EventType.MOVED: -1.0,
            EventType.AGENT_COLLISION: -10.0,
        },
    )
    events = [
        Event(type=EventType.MOVED, agent_id=0),
        Event(type=EventType.AGENT_COLLISION, agent_id=0),
    ]

    result = calculator.calculate_rewards(events, {0: None})

    assert result.individual == {0: -11.0}


def test_no_events_returns_zero_for_all_agents():
    calculator = EventRewardCalculator(
        reward_mapping={EventType.AGENT_COLLISION: -10.0},
    )

    result = calculator.calculate_rewards(events=[], team_of_agent={0: None, 1: None})

    assert result.individual == {0: 0.0, 1: 0.0}
    assert result.shared_by_team == {}


def test_event_without_reward_mapping_gives_zero():
    calculator = EventRewardCalculator(
        reward_mapping={EventType.AGENT_COLLISION: -10.0},
    )
    events = [Event(type=EventType.MOVED, agent_id=0)]

    result = calculator.calculate_rewards(events, {0: None})

    assert result.individual == {0: 0.0}


def test_shared_event_goes_to_shared_reward_not_to_agents():
    # snowplow: collision is shared, movement is individual
    calculator = EventRewardCalculator(
        reward_mapping={EventType.MOVED: -1.0, EventType.AGENT_COLLISION: -5.0},
        shared_events={EventType.AGENT_COLLISION},
    )
    events = [
        Event(type=EventType.MOVED, agent_id=0),
        Event(type=EventType.AGENT_COLLISION, agent_id=0),
    ]

    result = calculator.calculate_rewards(events, {0: None, 1: None})

    assert result.individual == {0: -1.0, 1: 0.0}
    assert result.shared_by_team == {None: -5.0}
    assert result.team_total(None) == -6.0


def test_shared_event_with_team_id_goes_to_that_team():
    # soccer: a goal belongs to the team, not to one player
    calculator = EventRewardCalculator(
        reward_mapping={DomainEventType.GOAL: 10.0},
        shared_events={DomainEventType.GOAL},
    )
    events = [Event(type=DomainEventType.GOAL, agent_id=0, team_id=1)]

    result = calculator.calculate_rewards(events, {0: 1, 1: 1, 2: 2})

    assert result.individual == {0: 0.0, 1: 0.0, 2: 0.0}
    assert result.shared_by_team == {1: 10.0}


def test_shared_event_team_is_taken_from_agent_when_missing():
    calculator = EventRewardCalculator(
        reward_mapping={DomainEventType.GOAL: 10.0},
        shared_events={DomainEventType.GOAL},
    )
    events = [Event(type=DomainEventType.GOAL, agent_id=2)]  # no team_id

    result = calculator.calculate_rewards(events, {0: 1, 1: 1, 2: 2})

    assert result.shared_by_team == {2: 10.0}


def test_shared_event_without_agent_and_team_goes_to_none():
    # e.g. ALL_CLEAN emitted by the environment itself
    calculator = EventRewardCalculator(
        reward_mapping={DomainEventType.ALL_CLEAN: 100.0},
        shared_events={DomainEventType.ALL_CLEAN},
    )
    events = [Event(type=DomainEventType.ALL_CLEAN)]

    result = calculator.calculate_rewards(events, {0: None, 1: None})

    assert result.shared_by_team == {None: 100.0}


def test_shared_event_from_unknown_agent_is_ignored():
    calculator = EventRewardCalculator(
        reward_mapping={EventType.AGENT_COLLISION: -5.0},
        shared_events={EventType.AGENT_COLLISION},
    )
    events = [Event(type=EventType.AGENT_COLLISION, agent_id=5)]

    result = calculator.calculate_rewards(events, {0: None})

    assert result.individual == {0: 0.0}
    assert result.shared_by_team == {}


def test_team_total_adds_individual_and_shared_of_that_team_only():
    calculator = EventRewardCalculator(
        reward_mapping={EventType.MOVED: -1.0, DomainEventType.GOAL: 10.0},
        shared_events={DomainEventType.GOAL},
    )
    events = [
        Event(type=EventType.MOVED, agent_id=0),
        Event(type=DomainEventType.GOAL, agent_id=0, team_id=1),
    ]

    result = calculator.calculate_rewards(events, {0: 1, 1: 1, 2: 2})

    assert result.team_total(1) == 9.0  # -1 (agent 0) + 10 (shared)
    assert result.team_total(2) == 0.0
