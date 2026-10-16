"""Soccer formations and role-constrained movement on the MultiGrid backbone."""

from dataclasses import dataclass

from gymnasium import spaces
from mosaic_multigrid.base import MultiGridEnv
from mosaic_multigrid.core.grid import Grid

from Enums import MovementActions
from soccer.movement import handle_simultaneous_movement
from soccer.agent import SoccerAgent, SoccerAgentRole
from soccer.ball import CustomBall


@dataclass
class SoccerEnvConfig:
    n_players_per_team: int = 3
    max_steps: int = 1000
    # None leaves the opening kickoff undecided in this movement scaffold.
    initial_ball_position: tuple[int, int] | None = None


class SoccerEnv(MultiGridEnv):
    """Movement scaffold; ball mechanics and soccer rewards are not yet implemented."""

    # Integer actions: wait, up, down, right, left, then four diagonals.
    movement_actions = tuple(MovementActions)

    def __init__(self, config=None, width=16, height=11, max_steps=None, render_mode=None):
        self.config = config if config is not None else SoccerEnvConfig()
        if self.config.n_players_per_team not in (2, 3):
            raise ValueError("Soccer supports 2 or 3 players per team")
        if (width, height) != (16, 11):
            raise ValueError("The baseline soccer pitch must be 16 by 11")
        team_size = self.config.n_players_per_team
        agents = [
            SoccerAgent(
                index=team * team_size + slot,
                team_index=team,
                original_role=(SoccerAgentRole.GOALKEEPER if slot == 0 else SoccerAgentRole.FIELD_PLAYER),
            )
            for team in range(2)
            for slot in range(team_size)
        ]
        super().__init__(
            agents=agents, width=width, height=height,
            max_steps=self.config.max_steps if max_steps is None else max_steps,
            render_mode=render_mode, allow_agent_overlap=False,
        )
        for agent in self.agents:
            agent.action_space = spaces.Discrete(len(self.movement_actions))

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        team_size = self.config.n_players_per_team
        formations = (
            (((0, 5), (3, 5)), ((15, 5), (12, 5))) if team_size == 2 else
            (((0, 5), (3, 3), (3, 7)), ((15, 5), (12, 3), (12, 7)))
        )
        for agent in self.agents:
            agent.current_role = agent.original_role
            agent.state.pos = formations[agent.team_index][agent.index % team_size]
            agent.state.dir = 0 if agent.team_index == 0 else 2
        self.last_movement_results = {}
        position = self.config.initial_ball_position
        if position is not None and not (0 <= position[0] < width and 0 <= position[1] < height):
            raise ValueError('Initial ball position must be on the pitch')
        self.ball = CustomBall(position=position) if position is not None else None

    def get_observable_grid(self):
        return self.grid.state.copy()

    def is_move_allowed(self, grid, agent, current_pos, intended_pos):
        x, y = intended_pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if agent.current_role is SoccerAgentRole.GOALKEEPER:
            x_min, x_max = (0, 2) if agent.team_index == 0 else (13, 15)
            return x_min <= x <= x_max and 2 <= y <= 8
        return True

    def handle_actions(self, actions):
        movements = {
            agent_id: action if isinstance(action, MovementActions) else self.movement_actions[action]
            for agent_id, action in actions.items()
        }
        self.last_movement_results = handle_simultaneous_movement(
            agents=self.agents, grid=self.grid, actions=movements,
            width=self.width, height=self.height, rng=self.np_random,
            is_move_allowed=self.is_move_allowed,
        )
        if self.ball is not None:
            self.ball.sync_with_carrier({agent.index: agent for agent in self.agents})
        return {agent.index: 0.0 for agent in self.agents}
