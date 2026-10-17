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

# Experiment 2: match outcome plus goals.
SOCCER_REWARDS_V2 = {
    **SOCCER_REWARDS_V1,
    SoccerEvent.GOAL_SCORED: 100,
    SoccerEvent.GOAL_CONCEDED: -100,
}
REWARD_PRESETS = {'v1': SOCCER_REWARDS_V1, 'v2': SOCCER_REWARDS_V2}

# In soccer every event belongs to the team, not to one player
#SOCCER_SHARED_EVENTS = set(SOCCER_REWARDS_V1)
SOCCER_SHARED_EVENTS = set(SOCCER_REWARDS_V2)


def make_soccer_reward_calculator(preset: str = 'v1') -> EventRewardCalculator:
    if preset not in REWARD_PRESETS:
        raise ValueError(f"Unknown reward preset: {preset}")
    return EventRewardCalculator(
        REWARD_PRESETS[preset],
        shared_events=SOCCER_SHARED_EVENTS,
    )
