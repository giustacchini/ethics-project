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
