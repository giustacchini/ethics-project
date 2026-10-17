import pytest

from enums import SoccerActionType
from soccer.env import SoccerEnv


def action(kind, movement=0, target=(3, 3), strength=0):
    return dict(type=int(kind), movement=movement, target=target, strength=strength,
                target_player_id=0, contest_intensity=0)


def test_structured_movement_wait_and_kick_decoding():
    env = SoccerEnv()
    env.reset(seed=0)
    actions = {a.index: action(SoccerActionType.DO_NOTHING, movement=3) for a in env.agents}
    actions[1] = action(SoccerActionType.MOVE, movement=3)
    actions[2] = action(SoccerActionType.KICK, movement=3, target=(0, 6), strength=1)
    assert env.action_space.contains(actions)
    env.step(actions)
    assert env.agents[0].state.pos == (0, 5)
    assert env.agents[1].state.pos == (4, 3)
    assert env.agents[2].state.pos == (3, 7)
    assert env.last_kick_requests == {2: dict(displacement=(-3, 3), strength=2)}
    env.reset(seed=0)
    assert env.last_kick_requests == {}


def test_sampled_actions_run_with_matching_observations():
    env = SoccerEnv(max_steps=10)
    env.reset(seed=0)
    env.action_space.seed(3)
    for _ in range(10):
        actions = env.action_space.sample()
        assert env.action_space.contains(actions)
        obs, _, _, truncations, _ = env.step(actions)
        assert env.observation_space.contains(obs)
    assert all(truncations.values())


def test_invalid_structured_action_rejected_before_movement():
    env = SoccerEnv()
    env.reset(seed=0)
    actions = {a.index: action(SoccerActionType.MOVE, movement=3) for a in env.agents}
    actions[5]['target'] = (7, 3)
    with pytest.raises(ValueError, match='Invalid soccer action'):
        env.step(actions)
    assert env.agents[1].state.pos == (3, 3)
