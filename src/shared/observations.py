"""Observation provider interface.

A provider only reads the environment and never changes it. It is separate
from the environment's step logic, so the view can be changed without
rewriting the environment. ``soccer.observation.provider.SoccerObservationProvider``
is the only implementation.

Cleanup note: ``FullObservationProvider`` and ``LocalObservationProvider``
were removed; they were generic leftovers (the first came from snowplow, the
second was never implemented) and soccer does not use either.
"""

from abc import ABC, abstractmethod
from typing import Any

from mosaic_multigrid.base import AgentID

Observation = dict[str, Any]


class ObservationProvider(ABC):
    @abstractmethod
    def get_observations(self, env) -> dict[AgentID, Observation]:
        """Generate observations for all active agents."""
