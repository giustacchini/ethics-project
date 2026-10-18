"""Fixed-size, complete soccer encodings for actors and centralized critics."""

import numpy as np

ENCODING_VERSION = 1


def _encode(obs, order, team, width, height, goals_to_win, max_steps):
    if min(width, height) < 2 or goals_to_win <= 0 or max_steps <= 0:
        raise ValueError('Encoding dimensions and normalization limits must be positive')
    n_players = len(order)
    def position(cell):
        return (-1.0, -1.0) if tuple(cell) == (-1, -1) else (cell[0] / (width - 1), cell[1] / (height - 1))
    numbers = [float(team)]  # preserves absolute attacking direction
    for player in order:
        numbers.extend((player / max(1, n_players - 1), *position(obs['agent_positions'][player]),
                        float(obs['agent_teams'][player] != team),
                        float(obs['original_roles'][player]), float(obs['current_roles'][player]),
                        float(obs['suspended'][player]), float(obs['ball_carrier'] == player),
                        float(obs['ball_kicker'] == player), float(obs['free_kick_taker'] == player)))
    numbers.extend((float(obs['ball_present']), *position(obs['ball_position'])))
    numbers.extend(float(obs['ball_state'] == state) for state in range(4))
    numbers.append(obs['ball_speed'] / 2)
    previous = obs['last_possessing_team']
    numbers.extend((float(previous == -1), float(previous == team), float(previous not in (-1, team))))
    numbers.extend((obs['score'][team] / goals_to_win, obs['score'][1 - team] / goals_to_win,
                    obs['steps_left'] / max_steps, float(obs['goal_pending'])))
    result = obs['match_result']
    numbers.extend(float(result == outcome) for outcome in (0, team + 1, 2 - team, 3))
    trajectory = obs['ball_trajectory']
    capacity = max(width, height) - 1
    if len(trajectory) > capacity:
        raise ValueError('Ball trajectory exceeds fixed pitch-based encoding capacity')
    numbers.append(len(trajectory) / capacity)
    for index in range(capacity):
        numbers.extend((1.0, *position(trajectory[index])) if index < len(trajectory) else (0.0, -1.0, -1.0))
    return np.asarray(numbers, dtype=np.float32)


def obs_to_vector(obs, width, height, goals_to_win, max_steps):
    """Actor input: self, teammates, opponents; retain every structured field."""
    me, team = obs['self_id'], obs['self_team']
    teammates = sorted(p for p, t in obs['agent_teams'].items() if t == team and p != me)
    opponents = sorted(p for p, t in obs['agent_teams'].items() if t != team)
    return _encode(obs, [me, *teammates, *opponents], team, width, height, goals_to_win, max_steps)


def state_to_vector(obs, team, width, height, goals_to_win, max_steps):
    """One canonical team-perspective critic input, independent of observer ID."""
    if team not in (0, 1):
        raise ValueError('Soccer team must be 0 or 1')
    order = sorted(p for p, t in obs['agent_teams'].items() if t == team)
    order += sorted(p for p, t in obs['agent_teams'].items() if t != team)
    return _encode(obs, order, team, width, height, goals_to_win, max_steps)


def encode_team(observations, team, width=16, height=11, goals_to_win=3, max_steps=1000):
    """Return IDs, actor matrix, centralized critic vector and active-player mask."""
    ids = sorted(i for i, obs in observations.items() if obs['self_team'] == team)
    if not ids:
        raise ValueError('No observations for requested team')
    actors = np.stack([obs_to_vector(observations[i], width, height, goals_to_win, max_steps) for i in ids])
    state = state_to_vector(observations[ids[0]], team, width, height, goals_to_win, max_steps)
    active = np.asarray([not observations[i]['suspended'][i] for i in ids], dtype=np.bool_)
    return ids, actors, state, active
