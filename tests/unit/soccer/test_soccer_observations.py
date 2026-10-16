import pytest

from enums import MovementActions
from soccer.env import SoccerEnv, SoccerEnvConfig


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('position', [None, (4, 3)])
def test_full_state_matches_declared_spaces(size, position):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size, initial_ball_position=position))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs[0]['agent_positions'][size] == (15, 5)
    assert len(obs[0]['agent_positions']) == size * 2
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert obs[0]['steps_left'] == 999


def test_collection_and_flight_state_are_observed():
    env = SoccerEnv(SoccerEnvConfig(initial_ball_position=(4, 3)))
    env.reset(seed=0)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[1] = MovementActions.RIGHT
    obs, *_ = env.step(actions)
    for observation in obs.values():
        assert observation['ball_carrier'] == 1
        assert observation['last_possessing_team'] == 0
        assert observation['ball_state'] == 2
        assert observation['ball_position'] == (4, 3)
    env.ball.carrier_id = None
    env.ball.remaining_trajectory = [(5, 3), (6, 3)]
    env.ball.speed = 2
    obs = env.gen_obs()
    assert obs[0]['ball_state'] == 3
    assert obs[0]['ball_trajectory'] == ((5, 3), (6, 3))
    assert env.observation_space.contains(obs)


def test_observations_do_not_alias_state_or_other_agents():
    env = SoccerEnv(SoccerEnvConfig(initial_ball_position=(7, 5)))
    obs, _ = env.reset(seed=0)
    obs[0]['agent_positions'][1] = (9, 9)
    obs[0]['current_roles'][1] = 0
    assert env.agents[1].state.pos == (3, 3)
    assert obs[1]['agent_positions'][1] == (3, 3)
    assert obs[1]['current_roles'][1] == 1
    frame = env.get_full_render(highlight=True, tile_size=16)
    assert frame.shape == (176, 256, 3)
