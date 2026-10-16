"""Observation providers for all environments.

A provider only reads the environment and never changes it. It is separate
from the environment's step logic, so the view (full or local) can be
changed without rewriting the environment.
"""

from abc import ABC, abstractmethod
from typing import Any

from mosaic_multigrid.base import AgentID

Observation = dict[str, Any]


class ObservationProvider(ABC):
    @abstractmethod
    def get_observations(self, env) -> dict[AgentID, Observation]:
        """Generate observations for all active agents."""


class FullObservationProvider(ObservationProvider):
    """Every active agent sees the whole observable grid.

    Each observation contains:
        grid: env.get_observable_grid(). The environment decides what
            is visible (e.g. snowplow hides clouds).
        agent_positions: position of every active agent.
        self_position: position of the observed agent.

    Terminated agents get no observation.
    """

    def get_observations(self, env) -> dict[AgentID, Observation]:
        observations = {}

        active_agents = [agent for agent in env.agents if not agent.state.terminated]

        agent_positions = {
            agent.index: tuple(agent.state.pos) for agent in active_agents
        }

        for agent in active_agents:
            observations[agent.index] = {
                "grid": env.get_observable_grid(),
                "agent_positions": agent_positions.copy(),
                "self_position": tuple(agent.state.pos),
            }

        return observations


class LocalObservationProvider(ObservationProvider):
    def __init__(self, view_size: int):
        if view_size not in (3, 5):
            raise ValueError("view_size must be 3 or 5")

        self.view_size = view_size

    def get_observations(self, env) -> dict[AgentID, Observation]:
        raise NotImplementedError("Local observations will be implemented later.")

class SoccerObservationProvider(ObservationProvider):
    """Full observation for soccer: positions, teams, ball, score and time.

    Every active agent sees the whole game. Extra fields (roles, red cards,
    ball flight) will be added when our own soccer env has them.
    """

    def get_observations(self, env) -> dict[AgentID, Observation]:
        observations = {}
        active_agents = [agent for agent in env.agents if not agent.state.terminated]

        # Same for every agent: the whole game state
        agent_positions = {a.index: tuple(int(v) for v in a.state.pos) for a in active_agents}
        agent_teams = {a.index: a.team_index for a in active_agents}
        ball_carrier = next(
            (a.index for a in active_agents if a.state.carrying is not None), None
        )
        loose_ball = env._find_loose_ball_pos()  # None if someone carries the ball
        score = dict(env.team_scores)
        steps_left = env.max_steps - env.step_count

        # Different for each agent: "who am I"
        for agent in active_agents:
            observations[agent.index] = {
                "self_id": agent.index,
                "self_team": agent.team_index,
                "self_position": agent_positions[agent.index],
                "agent_positions": agent_positions.copy(),
                "agent_teams": agent_teams.copy(),
                "ball_carrier": ball_carrier,
                "ball_position": loose_ball,
                "score": score.copy(),
                "steps_left": steps_left,
            }

        return observations