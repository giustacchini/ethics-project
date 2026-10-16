import pytest

from soccer.env import SoccerEnv
from soccer.validate_matches import check_state, run_match


@pytest.mark.parametrize('size', [2, 3])
def test_short_match_validation_and_repeatability(size):
    first, trace = run_match(size, seed=7, max_steps=20)
    repeated, repeated_trace = run_match(size, seed=7, max_steps=20)
    assert first == repeated
    assert trace == repeated_trace
    assert first['transitions'] == 20
    assert len(trace) == 21


def test_validator_detects_player_overlap():
    env = SoccerEnv()
    env.reset(seed=0)
    env.agents[2].state.pos = env.agents[1].state.pos
    with pytest.raises(AssertionError, match='overlap'):
        check_state(env, env.gen_obs())


def test_validator_detects_missing_loose_ball_marker():
    env = SoccerEnv()
    obs, _ = env.reset(seed=0)
    env.grid.set(*env.ball.position, None)
    with pytest.raises(AssertionError, match='Grid ball marker'):
        check_state(env, obs)
