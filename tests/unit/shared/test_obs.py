from types import SimpleNamespace

import numpy as np
import pytest

from shared.observations import (
    FullObservationProvider,
    LocalObservationProvider,
)


def make_env():
    agents = [
        SimpleNamespace(
            index=0,
            state=SimpleNamespace(
                pos=(1, 1),
                terminated=False,
            ),
        ),
        SimpleNamespace(
            index=1,
            state=SimpleNamespace(
                pos=(3, 2),
                terminated=False,
            ),
        ),
    ]

    grid = SimpleNamespace(state=np.zeros((5, 5, 3)))

    return SimpleNamespace(
        agents=agents,
        grid=grid,
        get_observable_grid=lambda: grid.state.copy(),
    )


def test_full_observation_for_active_agents():
    env = make_env()
    provider = FullObservationProvider()

    observations = provider.get_observations(env)

    assert 0 in observations
    assert 1 in observations

    assert observations[0]["self_position"] == (1, 1)
    assert observations[1]["self_position"] == (3, 2)

    assert observations[0]["agent_positions"] == {
        0: (1, 1),
        1: (3, 2),
    }


def test_terminated_agent_is_excluded():
    env = make_env()

    env.agents[1].state.terminated = True

    provider = FullObservationProvider()
    observations = provider.get_observations(env)

    assert 0 in observations
    assert 1 not in observations


def test_local_observation_view_size_validation():
    LocalObservationProvider(3)
    LocalObservationProvider(5)

    with pytest.raises(ValueError):
        LocalObservationProvider(4)


def test_observation_grid_is_a_copy():
    env = make_env()
    obs = FullObservationProvider().get_observations(env)

    obs[0]["grid"][0, 0, 0] = 99

    assert env.grid.state[0, 0, 0] == 0
