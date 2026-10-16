import numpy as np


def obs_to_vector(obs: dict, width: int, height: int,
                  goals_to_win: int, max_steps: int) -> np.ndarray:
    """Turn one agent's observation dict into a flat list of numbers for MAPPO.

    Agent order is always: me, my teammates, then opponents.
    All numbers are scaled to roughly 0..1.
    """
    me = obs["self_id"]
    my_team = obs["self_team"]
    teams = obs["agent_teams"]

    # Order: me first, then teammates, then opponents (sorted by id)
    teammates = sorted(a for a, t in teams.items() if t == my_team and a != me)
    opponents = sorted(a for a, t in teams.items() if t != my_team)
    order = [me] + teammates + opponents

    numbers = []

    # Per agent: x, y, and whether it carries the ball (3 numbers each)
    for a in order:
        x, y = obs["agent_positions"][a]
        numbers += [x / (width - 1), y / (height - 1), float(obs["ball_carrier"] == a)]

    # Ball on the ground: x, y  (or -1, -1 if someone carries it)
    if obs["ball_position"] is not None:
        bx, by = obs["ball_position"]
        numbers += [bx / (width - 1), by / (height - 1)]
    else:
        numbers += [-1.0, -1.0]

    # Score (mine, theirs) and time left
    my_goals = obs["score"].get(my_team, 0)
    their_goals = sum(s for t, s in obs["score"].items() if t != my_team)
    numbers += [my_goals / goals_to_win, their_goals / goals_to_win,
                obs["steps_left"] / max_steps]

    return np.array(numbers, dtype=np.float32)