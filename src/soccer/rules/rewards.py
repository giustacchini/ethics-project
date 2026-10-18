"""Match-outcome-only team rewards and their shared calculator."""

from enums import SoccerEvent
from shared.rewards import EventRewardCalculator
from soccer.entities.agent import SoccerAgentRole

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

# Experiment 3: the same events as experiment 2, plus potential-based shaping
# (team_potential below). The env adds the shaping only when the preset is 'v3'.
REWARD_PRESETS = {'v1': SOCCER_REWARDS_V1, 'v2': SOCCER_REWARDS_V2, 'v3': SOCCER_REWARDS_V2}
SHAPED_PRESETS = {'v3'}

# In soccer every event belongs to the team, not to one player
#SOCCER_SHARED_EVENTS = set(SOCCER_REWARDS_V1)
SOCCER_SHARED_EVENTS = set(SOCCER_REWARDS_V2)


def make_soccer_reward_calculator(preset: str = 'v1') -> EventRewardCalculator:
    if preset not in REWARD_PRESETS:
        raise ValueError(f'Unknown reward preset: {preset!r}')
    return EventRewardCalculator(
        REWARD_PRESETS[preset],
        shared_events=SOCCER_SHARED_EVENTS,
    )

SHAPING_PER_CELL = 1.0  # reward per cell of progress (a goal is worth 100)


def _distance(a, b) -> int:
    """Chebyshev distance: players move in 8 directions, so a diagonal step is 1 cell."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def team_potential(env, team: int) -> float:
    """How promising the current position is for `team` (higher = better)."""
    max_dist = max(env.width, env.height)
    ball = env.ball
    if ball is None:
        return -SHAPING_PER_CELL * 2 * max_dist
    agents = {agent.index: agent for agent in env.agents}
    carrier = agents.get(ball.carrier_id)
    if carrier is not None and carrier.team_index == team:
        goal_cells = env.goal_cells[1 - team]
        return -SHAPING_PER_CELL * min(_distance(carrier.state.pos, cell) for cell in goal_cells)
    chasers = [agent for agent in env.active_players if agent.team_index == team
               and agent.current_role is not SoccerAgentRole.GOALKEEPER]
    if not chasers:
        return -SHAPING_PER_CELL * 2 * max_dist
    closest = min(_distance(agent.state.pos, ball.position) for agent in chasers)
    return -SHAPING_PER_CELL * (max_dist + closest)


def approach_shaping(potential_before: float, potential_after: float,
                     done: bool, gamma: float = 0.99) -> float:
    """Potential-based shaping (Ng et al., 1999): F = gamma * phi(s') - phi(s)."""
    if done:
        potential_after = 0.0
    return gamma * potential_after - potential_before