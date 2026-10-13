from abc import ABC, abstractmethod
from mosaic_multigrid.base import AgentID
from shared.events import Event
from enum import Enum
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardResult:
    """Result of one step.

    individual: rewards for each agent.
    team_of_agent: team of each agent. Agents with the same value belong
        to the same team. None means no team (all snowplows).
    shared_by_team: reward/penalty that belongs to no single agent, but is shared by all agents in one team.
        e.g. collision, all streets cleaned.
        OBS: in snowplows team is None, so shared_by_team[None] is the shared reward for all snowplows.
    """

    individual: dict[AgentID, float]
    team_of_agent: dict[AgentID, int | None]
    shared_by_team: dict[int | None, float] = field(default_factory=dict)

    def team_total(self, team_id: int | None) -> float:
        """Cumulative reward for a team: sum of individual rewards + shared."""
        members = sum(
            reward
            for agent_id, reward in self.individual.items()
            if self.team_of_agent.get(agent_id) == team_id
        )
        return members + self.shared_by_team.get(team_id, 0.0)


class RewardStrategy(ABC):
    @abstractmethod
    def calculate_rewards(
        self,
        events: list[Event],
        team_of_agent: dict[AgentID, int | None],
    ) -> RewardResult:
        """Calculate rewards for the agents in team_of_agent."""


class EventRewardCalculator(RewardStrategy):
    """Calculate rewards from events.

    Normal events go to the individual reward of event.agent_id.
    Events in shared_events go to the shared reward of a team.

    The team is event.team_id if set, otherwise the agent's team in
    team_of_agent. None means no team (e.g. snowplows).
    """

    def __init__(
        self,
        reward_mapping: dict[Enum, float],
        shared_events: set[Enum] | None = None,
    ):
        self.reward_mapping = reward_mapping.copy()
        self.shared_events = set(shared_events) if shared_events is not None else set()

    def calculate_rewards(
        self,
        events: list[Event],
        team_of_agent: dict[AgentID, int | None],
    ) -> RewardResult:
        rewards = {agent_id: 0.0 for agent_id in team_of_agent}
        shared_by_team: dict[int | None, float] = {}

        for event in events:
            reward = self.reward_mapping.get(event.type, 0.0)

            if event.type in self.shared_events:
                team_id = event.team_id
                if team_id is None and event.agent_id is not None:
                    if event.agent_id not in team_of_agent:
                        continue
                    team_id = team_of_agent.get(event.agent_id)
                shared_by_team[team_id] = shared_by_team.get(team_id, 0.0) + reward
                continue

            if event.agent_id in rewards:
                rewards[event.agent_id] += reward

        return RewardResult(
            individual=rewards,
            team_of_agent=dict(team_of_agent),
            shared_by_team=shared_by_team,
        )
