"""Reward values (configuration v1.0) and calculator for the soccer environment."""

from Enums import SoccerEvent
from shared.rewards import EventRewardCalculator

# Reward configuration v1.0 (see the rules and rewards spec, Section 8)
SOCCER_REWARDS_V1 = {
    SoccerEvent.MATCH_WIN: 1000,
    SoccerEvent.MATCH_LOSS: -1000,
    SoccerEvent.MATCH_DRAW: -1000,  # a draw counts as a defeat for each team
    SoccerEvent.GOAL_SCORED: 100,
    SoccerEvent.GOAL_CONCEDED: -100,
    # SoccerEvent.PASS_COMPLETED: 2,
    # SoccerEvent.POSSESSION_RECOVERED: 5,
    # SoccerEvent.POSSESSION_LOST: -5,
}

# In soccer every event belongs to the team, not to one player
SOCCER_SHARED_EVENTS = set(SOCCER_REWARDS_V1)


def make_soccer_reward_calculator() -> EventRewardCalculator:
    return EventRewardCalculator(
        SOCCER_REWARDS_V1,
        shared_events=SOCCER_SHARED_EVENTS,
    )