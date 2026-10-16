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


@pytest.mark.parametrize('size', [2, 3])
def test_structured_observations_do_not_generate_local_images(size, monkeypatch):
    from mosaic_multigrid.base import MultiGridEnv

    def local_images_must_not_run(*args, **kwargs):
        raise AssertionError('Soccer must not generate local image observations')

    monkeypatch.setattr(MultiGridEnv, 'gen_obs', local_images_must_not_run)
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size), max_steps=1)
    obs, _ = env.reset(seed=0)
    for observation in obs.values():
        assert not {'image', 'direction', 'mission', 'global_rgb'} & observation.keys()
        assert len(observation['agent_positions']) == 2 * size
    common = [{k: v for k, v in observation.items() if k not in ('self_id', 'self_team')}
              for observation in obs.values()]
    assert all(state == common[0] for state in common)
    assert env.observation_space.contains(obs)
    obs, _, _, truncs, _ = env.step({a.index: MovementActions.DO_NOTHING for a in env.agents})
    assert all(truncs.values())
    assert env.observation_space.contains(obs)
    assert len(obs) == 2 * size  # final observations remain available to everyone


def test_gui_frame_does_not_depend_on_observations_or_local_highlights(monkeypatch):
    import numpy as np

    env = SoccerEnv()
    env.reset(seed=0)
    def observations_must_not_run():
        raise AssertionError('Rendering must read state directly')
    monkeypatch.setattr(env, 'gen_obs', observations_must_not_run)
    normal = env.get_frame(highlight=False, tile_size=16)
    highlighted = env.get_frame(highlight=True, tile_size=16)
    assert normal.shape == (176, 256, 3)
    assert np.array_equal(normal, highlighted)
    assert env.highlight is False
