"""Match-outcome-only team rewards and their shared calculator."""

from enums import SoccerEvent
from shared.rewards import EventRewardCalculator

# Session choice supersedes the Markdown's shaped baseline: no intermediate
# rewards. Keep the existing public constant name for callers.
SOCCER_REWARDS_V1 = {
    SoccerEvent.MATCH_WIN: 1000,
    SoccerEvent.MATCH_LOSS: -1000,
    SoccerEvent.MATCH_DRAW: -1000
}

# In soccer every event belongs to the team, not to one player
SOCCER_SHARED_EVENTS = set(SOCCER_REWARDS_V1)


def make_soccer_reward_calculator() -> EventRewardCalculator:
    return EventRewardCalculator(
        SOCCER_REWARDS_V1,
        shared_events=SOCCER_SHARED_EVENTS,
    )
