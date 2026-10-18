"""Soccer formations and role-constrained movement on the MultiGrid backbone."""

from copy import deepcopy

from gymnasium import spaces
from mosaic_multigrid.base import MultiGridEnv
from mosaic_multigrid.core import Grid
from mosaic_multigrid.core.constants import TILE_PIXELS

from enums import MovementActions, SoccerActionType, SoccerEvent
from shared.events import EventLog
from soccer.rules.movement import resolve_simultaneous_conflicts
from shared.movement import build_movement_intentions, validate_proposed_moves, commit_resolved_positions
from soccer.entities.agent import SoccerAgent, SoccerAgentRole
from soccer.entities.ball import CustomBall
from soccer.rendering.pitch import render_soccer_frame
from soccer.rules.rewards import make_soccer_reward_calculator, SHAPED_PRESETS, approach_shaping, team_potential
from soccer.config import SoccerEnvConfig
from soccer.rules.ball_rules import BallRules
from soccer.rules.discipline import DisciplineRules
from soccer.observation.provider import SoccerObservationProvider
from soccer.evaluation.match_logging import MatchLog


class SoccerEnv(BallRules, DisciplineRules, MultiGridEnv):
    """Soccer scaffold with team-shared rewards for match outcomes only."""

    # Integer actions: wait, up, down, right, left, then four diagonals.
    movement_actions = tuple(MovementActions)

    def __init__(self, config=None, width=16, height=11, max_steps=None, render_mode=None,
                 event_log_path=None, show_events=False, render_fps: int = 20, event_log_append=False):
        self.metadata["render_fps"] = render_fps
        self.match_log = MatchLog(event_log_path, show_events=show_events, append=event_log_append)
        self.config = config if config is not None else SoccerEnvConfig()
        if self.config.n_players_per_team not in (2, 3):
            raise ValueError("Soccer supports 2 or 3 players per team")
        if (width, height) != (16, 11):
            raise ValueError("The baseline soccer pitch must be 16 by 11")
        if not self.config.goal_rows or any(y not in range(height) for y in self.config.goal_rows):
            raise ValueError('Goal rows must be nonempty and on the pitch')
        self.goal_cells = {
            0: frozenset((0, y) for y in self.config.goal_rows),
            1: frozenset((width - 1, y) for y in self.config.goal_rows),
        }
        team_size = self.config.n_players_per_team
        agents = [
            SoccerAgent(
                index=team * team_size + slot,
                team_index=team,
                n_players=2 * team_size,
                original_role=(SoccerAgentRole.GOALKEEPER if slot == 0 else SoccerAgentRole.FIELD_PLAYER),
            )
            for team in range(2)
            for slot in range(team_size)
        ]
        super().__init__(
            agents=agents, width=width, height=height,
            max_steps=self.config.max_steps if max_steps is None else max_steps,
            render_mode=render_mode, allow_agent_overlap=False, highlight=False,
        )
        self.reward_calculator = make_soccer_reward_calculator(self.config.rewards)
        self.reward_range = (-1000, 1000)
        self.observation_provider = SoccerObservationProvider()
        for agent in self.agents:
            agent.observation_space = self.observation_provider.build_space(self)

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.match_log.start(self)
        return result

    def _gen_grid(self, width, height):
        self._reset_formation()
        self.match_over = False
        self.winner = None
        self.match_result = 'ongoing'
        self.last_movement_results = {}
        self.last_possession_transfer = None
        self.last_kick_requests = {}
        self.last_kick_results = {}
        self.last_pass_completed = None
        self.last_tackle_results = {}
        self.last_dive_results = {}
        self._dive_targets = {}
        self._dive_start_occupied = set()
        self._aggressive_tacklers = set()
        self.last_foul = None
        self.last_fouls = []
        self.last_restart = None
        self.last_red_card = None
        self.last_loose_ball_contest = None
        self.last_red_cards = []
        self.team_scores = {0: 0, 1: 0}
        self.last_goal = None
        self.pending_goal = None
        self.reward_events = EventLog()
        self.last_reward_result = self.reward_calculator.calculate_rewards(
            [], {a.index: a.team_index for a in self.agents},
        )
        self.last_team_rewards = {0: 0.0, 1: 0.0}
        position = self.config.initial_ball_position
        if position == 'random':
            position = (7 + int(self.np_random.integers(2)), 5)
        if position is not None and not (0 <= position[0] < width and 0 <= position[1] < height):
            raise ValueError('Initial ball position must be on the pitch')
        self.ball = CustomBall(position=position) if position is not None else None
        if self.ball is not None:
            self.grid.set(*self.ball.position, self.ball)
        self.potentials = {team: team_potential(self, team) for team in (0, 1)}

    def _reset_formation(self, restore_suspended=True):
        """Restore players without resetting scores, elapsed time or events."""
        width, height = self.width, self.height
        self.free_kick_taker = None
        self.grid = Grid(width, height)  # plain Mosaic grid: it only stores the ball; drawing lives in soccer/rendering/pitch.py
        team_size = self.config.n_players_per_team
        formations = (
            (((0, 5), (3, 5)), ((15, 5), (12, 5))) if team_size == 2 else
            (((0, 5), (3, 3), (3, 7)), ((15, 5), (12, 3), (12, 7)))
        )
        for agent in self.agents:
            if not restore_suspended and agent.suspended:
                continue
            if restore_suspended:
                agent.suspended = False
                agent.current_role = agent.original_role
            agent.state.carrying = None
            agent.state.terminated = False
            agent.state.pos = formations[agent.team_index][agent.index % team_size]
            if not restore_suspended and agent.current_role is SoccerAgentRole.GOALKEEPER:
                agent.state.pos = (0 if agent.team_index == 0 else 15, 5)
            agent.state.dir = 0 if agent.team_index == 0 else 2

    def get_observable_grid(self):
        return self.grid.state.copy()

    @property
    def active_players(self):
        return [agent for agent in self.agents if not agent.suspended]

    def check_goalkeeper_invariants(self):
        """Validation hook for stable states, excluding suspended players."""
        for team in (0, 1):
            keepers = [a for a in self.active_players if a.team_index == team
                       and a.current_role is SoccerAgentRole.GOALKEEPER]
            if len(keepers) > 1:
                raise AssertionError(f'Team {team} has multiple active goalkeepers')
            for keeper in keepers:
                if not self.is_move_allowed(self.grid, keeper, keeper.state.pos, keeper.state.pos):
                    raise AssertionError(f'Goalkeeper {keeper.index} is outside its team area')

    def is_move_allowed(self, grid, agent, current_pos, intended_pos):
        x, y = intended_pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if agent.current_role is SoccerAgentRole.GOALKEEPER:
            x_min, x_max = (0, 2) if agent.team_index == 0 else (13, 15)
            return x_min <= x <= x_max and 3 <= y <= 7
        return True

    def gen_obs(self):
        return self.observation_provider.get_observations(self)

    def get_full_render(self, highlight=False, tile_size=TILE_PIXELS):
        """Render the full pitch in the original Mosaic soccer style.

        Mosaic's get_frame()/render() (human window and rgb_array) call this hook.
        It used to draw flat green tiles through a custom Grid subclass; it now
        delegates to soccer/rendering/pitch.py. ``highlight`` is ignored on purpose:
        our observations are the full structured state, so there is no local
        field of view to highlight. Game logic is not touched by rendering.
        """
        return render_soccer_frame(self, tile_size)

    def step(self, actions):
        if self.match_over:
            raise RuntimeError('Match finished; call reset before stepping again')
        before = [dict(id=a.index, team=a.team_index, position=tuple(a.state.pos),
                       suspended=a.suspended,
                       decision_eligible=not a.suspended and (self.free_kick_taker is None or self.free_kick_taker == a.index))
                  for a in self.agents]
        submitted = deepcopy(actions)
        observations, rewards, terminations, truncations, infos = super().step(actions)
        # Reaching three goals is a natural terminal, even at the time limit.
        if self.match_over and max(self.team_scores.values()) >= 3:
            truncations = {agent.index: False for agent in self.agents}
        for agent in self.agents:
            infos[agent.index]['score'] = self.team_scores.copy()
            infos[agent.index]['team_rewards'] = self.last_team_rewards.copy()
            infos[agent.index]['tackles'] = deepcopy(self.last_tackle_results)
            infos[agent.index]['foul'] = deepcopy(self.last_foul)
            infos[agent.index]['fouls'] = deepcopy(self.last_fouls)
            infos[agent.index]['restart'] = deepcopy(self.last_restart)
            infos[agent.index]['red_card'] = deepcopy(self.last_red_card)
            infos[agent.index]['dives'] = deepcopy(self.last_dive_results)
            infos[agent.index]['loose_ball_contest'] = deepcopy(self.last_loose_ball_contest)
            infos[agent.index]['red_cards'] = deepcopy(self.last_red_cards)
            infos[agent.index].update(
                match_over=self.match_over, winner=self.winner, match_result=self.match_result,
            )
            if self.last_goal is not None:
                infos[agent.index]['goal_scored_by'] = deepcopy(self.last_goal)
        self.match_log.transition(self, submitted, before)
        return observations, rewards, terminations, truncations, infos

    def _finish_transition(self):
        """Finalize a scored goal/time limit before producing next observations."""
        reached_target = max(self.team_scores.values()) >= 3
        if reached_target or self.step_count >= self.max_steps:
            self.match_over = True
            left, right = self.team_scores[0], self.team_scores[1]
            self.winner = None if left == right else (0 if left > right else 1)
            self.match_result = 'draw' if self.winner is None else ('left_win' if self.winner == 0 else 'right_win')
            if reached_target:
                for agent in self.agents:
                    agent.state.terminated = True
            self.pending_goal = None
            for team in (0, 1):
                event = (
                    SoccerEvent.MATCH_DRAW if self.winner is None else
                    SoccerEvent.MATCH_WIN if team == self.winner else SoccerEvent.MATCH_LOSS
                )
                self.reward_events.emit(event, team_id=team, step=self.step_count)
        elif self.pending_goal is not None:
            conceded = self.pending_goal['conceding_team']
            self._reset_formation()
            self.ball = CustomBall(position=(7 if conceded == 0 else 8, 5))
            self.grid.set(*self.ball.position, self.ball)
            self.pending_goal = None

    def _check_goal(self):
        """Score before ordinary reception; current defending GKs block their cell."""
        ball = self.ball
        if ball is None or self.pending_goal is not None:
            return False
        for defending_team, cells in self.goal_cells.items():
            if ball.position not in cells:
                continue
            protected = any(
                agent.team_index == defending_team
                and agent.current_role is SoccerAgentRole.GOALKEEPER
                and tuple(agent.state.pos) == ball.position
                for agent in self.active_players
            )
            if protected:
                return False
            scoring_team = 1 - defending_team
            self._award_goal(scoring_team, tuple(ball.position))
            return True
        return False

    def _award_goal(self, scoring_team, position, reason='ball_reached_goal'):
        if self.pending_goal is not None:
            return
        self.team_scores[scoring_team] += 1
        self.last_goal = {
            'team': scoring_team, 'conceding_team': 1 - scoring_team,
            'position': tuple(position), 'transition': self.step_count, 'reason': reason,
        }
        self.pending_goal = deepcopy(self.last_goal)
        self.reward_events.emit(SoccerEvent.GOAL_SCORED, team_id=scoring_team, step=self.step_count)
        self.reward_events.emit(SoccerEvent.GOAL_CONCEDED, team_id=1 - scoring_team, step=self.step_count)
        if self.ball is not None:
            self.ball.position = tuple(position)
            self.ball.remaining_trajectory.clear()
            self.ball.speed = 0
            self.ball.carrier_id = None
            self.ball.kicker_id = None

    def _transition_rewards(self):
        for result in self.last_dive_results.values():
            if result['status'] == 'ready':
                result['status'] = 'not_triggered'
        self._finish_transition()
        self.last_reward_result = self.reward_calculator.calculate_rewards(
            self.reward_events.events(), {a.index: a.team_index for a in self.agents},
        )
        self.last_team_rewards = {
            team: self.last_reward_result.team_total(team) for team in (0, 1)
        }

        if self.config.rewards in SHAPED_PRESETS:
            for team in (0, 1):
                after = team_potential(self, team)
                self.last_team_rewards[team] += approach_shaping(
                    self.potentials[team], after, self.match_over, self.config.shaping_gamma)
                self.potentials[team] = after
                
        # Repeat each team's single aggregate through the per-agent API.
        return {a.index: self.last_team_rewards[a.team_index] for a in self.agents}

    def handle_actions(self, actions):
        self.last_goal = None
        self.reward_events.clear()
        # Decode all actions before changing any live state. Legacy movement
        # enums/indices remain accepted for existing runners and tests.
        movements, kick_requests, tackle_requests, dive_requests = {}, {}, {}, {}
        contest_intensities = {}
        self.last_dive_results = {}
        self._dive_targets = {}
        self.last_tackle_results = {}
        self._aggressive_tacklers = set()
        self.last_foul = None
        self.last_fouls = []
        self.last_restart = None
        self.last_red_card = None
        self.last_red_cards = []
        self.last_loose_ball_contest = None
        active_ids = {a.index for a in self.active_players}
        if not active_ids.issubset(actions) or not set(actions).issubset({a.index for a in self.agents}):
            raise ValueError('Provide one action for every active agent')
        for agent in self.agents:
            if agent.suspended:
                continue
            action = actions[agent.index]
            if isinstance(action, dict):
                action = {'contest_intensity': 0, **action}
                # Accept pre-tackle action dictionaries for existing callers.
                if 'target_player_id' not in action and action.get('type') not in (
                    SoccerActionType.TACKLE, SoccerActionType.AGGRESSIVE_TACKLE,
                ):
                    action = {**action, 'target_player_id': 0}
                if not agent.action_space.contains(action):
                    raise ValueError(f'Invalid soccer action for agent {agent.index}')
                action_type = SoccerActionType(action['type'])
                if action_type == SoccerActionType.MOVE:
                    contest_intensities[agent.index] = int(action['contest_intensity'])
                movements[agent.index] = (
                    self.movement_actions[action['movement']]
                    if action_type == SoccerActionType.MOVE else MovementActions.DO_NOTHING
                )
                if action_type == SoccerActionType.KICK:
                    kick_requests[agent.index] = {
                        'displacement': tuple(int(value) - 3 for value in action['target']),
                        'strength': int(action['strength']) + 1,
                    }
                elif action_type in (SoccerActionType.TACKLE, SoccerActionType.AGGRESSIVE_TACKLE):
                    tackle_requests[agent.index] = int(action['target_player_id'])
                    if action_type == SoccerActionType.AGGRESSIVE_TACKLE:
                        self._aggressive_tacklers.add(agent.index)
                elif action_type == SoccerActionType.DIVE:
                    dive_requests[agent.index] = tuple(int(v) - 3 for v in action['target'])
            elif isinstance(action, MovementActions):
                movements[agent.index] = action
            else:
                if not spaces.Discrete(len(self.movement_actions)).contains(action):
                    raise ValueError(f'Invalid movement action for agent {agent.index}')
                movements[agent.index] = self.movement_actions[action]
        if self.free_kick_taker is not None:
            # The taker must launch a legal kick. All selected movement,
            # tackling and diving actions are held until the next transition.
            movements = {i: MovementActions.DO_NOTHING for i in active_ids}
            kick_requests = {i: request for i, request in kick_requests.items() if i == self.free_kick_taker}
            tackle_requests.clear()
            dive_requests.clear()
            contest_intensities.clear()
            self._aggressive_tacklers.clear()
        valid_tacklers = self._validate_tackles(tackle_requests)
        self._validate_dives(dive_requests)
        self._prepare_aggressive_tackles()
        self.last_kick_requests = kick_requests
        self.last_kick_results = {}
        self.last_pass_completed = None
        # Possession challenges use occupancy before anyone moves.
        agents_by_id = {agent.index: agent for agent in self.agents}
        starting_occupants = {tuple(agent.state.pos): agent for agent in self.active_players}
        carrier_id = self.ball.carrier_id if self.ball is not None else None
        old_ball_position = self.ball.position if self.ball is not None else None
        was_in_flight = self.ball is not None and self.ball.flight_state == 'in_flight'
        self.last_possession_transfer = None
        self.last_movement_results = {}
        # A foul interrupts movement, tackles, kicks and ball flight together.
        if self._resolve_foul_restart():
            self.last_kick_results = {i: 'interrupted' for i in kick_requests}
            return self._transition_rewards()
        intentions = build_movement_intentions(self.active_players, movements)
        validated = validate_proposed_moves(
            intentions, agents_by_id, self.grid, self.width, self.height, self.is_move_allowed,
        )
        contests = []
        loose_position = self.ball.position if self.ball is not None and self.ball.flight_state == 'loose' else None
        resolved = resolve_simultaneous_conflicts(
            validated, agents_by_id, self.np_random, loose_ball_position=loose_position,
            contest_intensities=contest_intensities, contest_events=contests,
        )
        if contests:
            self.last_loose_ball_contest = contests[0]
            if contests[0]['red_cards']:
                self._resolve_contest_cards(contests[0])
                self.last_kick_results = {i: 'interrupted' for i in kick_requests}
                return self._transition_rewards()
        self.last_movement_results = commit_resolved_positions(agents_by_id, resolved)
        if self.ball is not None:
            # Remove the old loose/in-flight marker before updating ball state.
            cell = self.grid.get(*old_ball_position)
            if isinstance(cell, CustomBall):
                self.grid.set(*old_ball_position, None)
            # Only a legal dribble blocked by starting occupancy can transfer
            # possession; waits and invalid moves cannot challenge a player.
            if carrier_id is not None:
                carrier = agents_by_id[carrier_id]
                movement = self.last_movement_results[carrier_id]
                opponent = starting_occupants.get(tuple(movement['intended_position']))
                if (
                    movement['status'] == 'blocked_occupied_cell'
                    and opponent is not None
                    and opponent.team_index != carrier.team_index
                ):
                    self.last_possession_transfer = {
                        'from_agent': carrier_id,
                        'to_agent': opponent.index,
                        'from_team': carrier.team_index,
                        'to_team': opponent.team_index,
                    }
                    self.ball.carrier_id = opponent.index
            cancelled_kickers = self._resolve_tackles(valid_tacklers, agents_by_id)
            for agent_id, request in kick_requests.items():
                if agent_id in cancelled_kickers:
                    self.last_kick_results[agent_id] = 'cancelled_by_tackle'
                elif agent_id != carrier_id:
                    self.last_kick_results[agent_id] = 'not_carrier'
                else:
                    self.last_kick_results[agent_id] = self._launch_kick(agents_by_id[agent_id], request)
            self.ball.sync_with_carrier(agents_by_id)
            if (
                carrier_id is not None and self.ball.carrier_id == carrier_id
                and carrier_id not in cancelled_kickers
                and self.last_movement_results[carrier_id]['status'] == 'moved'
            ):
                self._try_dives(self.ball.position, agents_by_id, dribbler_id=carrier_id)
            # Detect dribbled goals before collection; new kicks are checked
            # as they traverse cells, not at their launch position.
            if self.ball.flight_state != 'in_flight':
                self._check_goal()
            if self.pending_goal is None and (self.ball.flight_state == 'loose' or was_in_flight):
                for agent in self.active_players:
                    if tuple(agent.state.pos) == self.ball.position:
                        self._receive_ball(agent, agents_by_id)
                        break
            if self.pending_goal is None and self.ball.flight_state == 'in_flight':
                self._advance_ball(agents_by_id)
            self.ball.sync_with_carrier(agents_by_id)
            if self.ball.carrier_id is None:
                self.grid.set(*self.ball.position, self.ball)
            # Mosaic's renderer reads carrying from AgentState, not carrier_id.
            for agent in self.agents:
                agent.state.carrying = self.ball if agent.index == self.ball.carrier_id else None
        else:
            self.last_kick_results = {i: 'not_carrier' for i in kick_requests}
        return self._transition_rewards()


# ---------------------------------------------------------------------------
# Factories used by the gymnasium registration in soccer/__init__.py, so the
# env can be created by id exactly like the original Mosaic one:
#     import soccer, gymnasium as gym
#     env = gym.make("EthicsSoccer-3v3-v0", render_mode="human")
# ---------------------------------------------------------------------------
def make_3v3(render_mode=None, **kwargs):
    return SoccerEnv(SoccerEnvConfig(n_players_per_team=3), render_mode=render_mode, **kwargs)


def make_2v2(render_mode=None, **kwargs):
    return SoccerEnv(SoccerEnvConfig(n_players_per_team=2), render_mode=render_mode, **kwargs)
