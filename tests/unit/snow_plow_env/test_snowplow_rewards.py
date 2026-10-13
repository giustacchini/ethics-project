from Enums import EventType, SnowplowEvent
from shared.events import Event
from snowplow.rewards import make_snowplow_reward_calculator


def test_collision_is_shared_and_movement_is_individual():
    calculator = make_snowplow_reward_calculator()
    events = [
        Event(type=EventType.MOVED, agent_id=0),
        Event(type=EventType.AGENT_COLLISION, agent_id=0),
    ]

    result = calculator.calculate_rewards(events, {0: None, 1: None})

    assert result.individual == {0: -1.0, 1: 0.0}
    assert result.shared_by_team == {None: -10.0}
    assert result.team_total(None) == -11.0


def test_partial_cleaning_is_individual_and_all_clean_is_shared():
    calculator = make_snowplow_reward_calculator()
    events = [
        Event(type=SnowplowEvent.PARTIALLY_CLEARED, agent_id=1),
        Event(type=SnowplowEvent.ALL_CLEAN),
    ]

    result = calculator.calculate_rewards(events, {0: None, 1: None})

    assert result.individual == {0: 0.0, 1: 1.0}
    assert result.shared_by_team == {None: 100.0}
    assert result.team_total(None) == 101.0