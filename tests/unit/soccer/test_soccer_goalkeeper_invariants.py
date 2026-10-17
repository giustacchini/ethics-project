from itertools import product

import pytest

from soccer.agent import SoccerAgentRole
from soccer.env import SoccerEnv, SoccerEnvConfig


@pytest.mark.parametrize('size', [2, 3])
def test_initial_goalkeeper_invariants(size):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    env.reset(seed=0)
    env.check_goalkeeper_invariants()


def test_temporary_goalkeeper_ignores_suspended_original():
    env = SoccerEnv()
    env.reset(seed=0)
    env._apply_red_card(0)
    env.check_goalkeeper_invariants()
    assert env.agents[0].suspended
    assert env.agents[1].current_role is SoccerAgentRole.GOALKEEPER


def test_duplicate_active_goalkeeper_is_invalid():
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[1].current_role = SoccerAgentRole.GOALKEEPER
    env.agents[1].state.pos = (1, 5)
    with pytest.raises(AssertionError, match='multiple active goalkeepers'):
        env.check_goalkeeper_invariants()


@pytest.mark.parametrize('position', [(3, 5), (1, 2), (1, 8), (14, 5)])
def test_keeper_outside_own_area_is_invalid(position):
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[0].state.pos = position
    with pytest.raises(AssertionError, match='outside its team area'):
        env.check_goalkeeper_invariants()


def test_every_opposing_keeper_position_pair_has_disjoint_dive_targets():
    env = SoccerEnv()
    env.reset(seed=0)
    left, right = env.agents[0], env.agents[3]
    def targets(keeper, position):
        return {
            (position[0] + dx, position[1] + dy)
            for dx, dy in product((-1, 0, 1), repeat=2)
            if (dx, dy) != (0, 0) and env.is_move_allowed(
                env.grid, keeper, position, (position[0] + dx, position[1] + dy))
        }
    left_positions = list(product(range(3), range(3, 8)))
    right_positions = list(product(range(13, 16), range(3, 8)))
    for lpos, rpos in product(left_positions, right_positions):
        assert targets(left, lpos).isdisjoint(targets(right, rpos))


def test_shared_dive_target_raises_instead_of_silently_skipping_save():
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[1].current_role = SoccerAgentRole.GOALKEEPER
    env.agents[1].state.pos = (2, 5)
    env._dive_targets = {0: (1, 5), 1: (1, 5)}
    env.last_dive_results = {0: dict(status='ready'), 1: dict(status='ready')}
    with pytest.raises(AssertionError, match='share dive target'):
        env._try_dives((1, 5), {a.index: a for a in env.agents})
