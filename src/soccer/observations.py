"""Complete structured soccer observations, independent of rendered images."""

from copy import deepcopy

from gymnasium import spaces
from shared.observations import ObservationProvider


class SoccerObservationProvider(ObservationProvider):
    """Expose the complete implemented game state to every roster member."""

    def build_space(self, env):
        position_space = spaces.MultiDiscrete([env.width, env.height])
        return spaces.Dict({
            'self_id': spaces.Discrete(env.num_agents),
            'self_team': spaces.Discrete(2),
            'agent_positions': spaces.Dict({a.index: spaces.MultiDiscrete(
                [env.width + 1, env.height + 1], start=[-1, -1],
            ) for a in env.agents}),
            'suspended': spaces.Dict({a.index: spaces.Discrete(2) for a in env.agents}),
            'agent_teams': spaces.Dict({a.index: spaces.Discrete(2) for a in env.agents}),
            'original_roles': spaces.Dict({a.index: spaces.Discrete(2) for a in env.agents}),
            'current_roles': spaces.Dict({a.index: spaces.Discrete(2) for a in env.agents}),
            'ball_present': spaces.Discrete(2),
            # -1 denotes no ball, carrier or previous possessing team.
            'ball_position': spaces.MultiDiscrete([env.width + 1, env.height + 1], start=[-1, -1]),
            'ball_carrier': spaces.Discrete(env.num_agents + 1, start=-1),
            'ball_kicker': spaces.Discrete(env.num_agents + 1, start=-1),
            'last_possessing_team': spaces.Discrete(3, start=-1),
            'ball_state': spaces.Discrete(4),  # absent, loose, possessed, in flight
            'ball_trajectory': spaces.Sequence(position_space),
            'ball_speed': spaces.Discrete(3),
            'steps_left': spaces.Discrete(env.max_steps + 1),
            'score': spaces.Dict({team: spaces.Discrete(env.max_steps + 1) for team in (0, 1)}),
            'goal_pending': spaces.Discrete(2),
            'match_result': spaces.Discrete(4),  # ongoing, left win, right win, draw
            'free_kick_taker': spaces.Discrete(env.num_agents + 1, start=-1),
        })

    def get_observations(self, env):
        ball = env.ball
        state = {
            'agent_positions': {a.index: (-1, -1) if a.suspended else tuple(a.state.pos) for a in env.agents},
            'suspended': {a.index: int(a.suspended) for a in env.agents},
            'agent_teams': {a.index: a.team_index for a in env.agents},
            'original_roles': {a.index: a.original_role.value for a in env.agents},
            'current_roles': {a.index: a.current_role.value for a in env.agents},
            'ball_present': int(ball is not None),
            'ball_position': tuple(ball.position) if ball is not None else (-1, -1),
            'ball_carrier': ball.carrier_id if ball is not None and ball.carrier_id is not None else -1,
            'ball_kicker': ball.kicker_id if ball is not None and ball.kicker_id is not None else -1,
            'last_possessing_team': ball.last_possessing_team if ball is not None and ball.last_possessing_team is not None else -1,
            'ball_state': {'loose': 1, 'possessed': 2, 'in_flight': 3}[ball.flight_state] if ball is not None else 0,
            'ball_trajectory': tuple(tuple(p) for p in ball.remaining_trajectory) if ball is not None else (),
            'ball_speed': ball.speed if ball is not None else 0,
            'steps_left': max(0, env.max_steps - env.step_count),
            'score': env.team_scores.copy(),
            'goal_pending': int(env.pending_goal is not None),
            'match_result': {'ongoing': 0, 'left_win': 1, 'right_win': 2, 'draw': 3}[env.match_result],
            'free_kick_taker': env.free_kick_taker if env.free_kick_taker is not None else -1,
        }
        return {
            agent.index: dict(deepcopy(state), self_id=agent.index, self_team=agent.team_index)
            for agent in env.agents
        }
