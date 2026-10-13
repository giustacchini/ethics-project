"""Reward values and calculator for the snowplow environment."""

from Enums import EventType, SnowplowEvent
from shared.rewards import EventRewardCalculator

# Rewards and penalties for snowplow events
SNOWPLOW_REWARDS = {
    EventType.MOVED: -1,
    EventType.DO_NOTHING: -2,
    EventType.AGENT_COLLISION: -10,
    SnowplowEvent.PARTIALLY_CLEARED: 1,
    SnowplowEvent.SECONDARY_STREET_CLEARED: 2,
    SnowplowEvent.MAIN_STREET_CLEARED: 3,
    SnowplowEvent.ALL_CLEAN: 100,
}


# Shared rewards/penalties: they belong to all plows, not to one plow.
SNOWPLOW_SHARED_EVENTS = {
    EventType.AGENT_COLLISION,
    SnowplowEvent.ALL_CLEAN,
}


def make_snowplow_reward_calculator() -> EventRewardCalculator:
    return EventRewardCalculator(
        SNOWPLOW_REWARDS,
        shared_events=SNOWPLOW_SHARED_EVENTS,
    )
