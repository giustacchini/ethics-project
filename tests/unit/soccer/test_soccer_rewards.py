from enums import SoccerEvent
from shared.events import Event
from soccer.rules.rewards import make_soccer_reward_calculator

# Agents 0, 1, 2 are in team 1; agents 3, 4, 5 are in team 2
TEAMS_3V3 = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}


def test_goals_have_no_reward():
    calculator = make_soccer_reward_calculator()
    events = [
        Event(type=SoccerEvent.GOAL_SCORED, team_id=1),
        Event(type=SoccerEvent.GOAL_CONCEDED, team_id=2),
    ]

    result = calculator.calculate_rewards(events, TEAMS_3V3)

    assert result.team_total(1) == 0.0
    assert result.team_total(2) == 0.0


def test_draw_is_punished_for_both_teams():
    calculator = make_soccer_reward_calculator()
    events = [
        Event(type=SoccerEvent.MATCH_DRAW, team_id=1),
        Event(type=SoccerEvent.MATCH_DRAW, team_id=2),
    ]

    result = calculator.calculate_rewards(events, TEAMS_3V3)

    assert result.team_total(1) == -1000.0
    assert result.team_total(2) == -1000.0


import pytest

from enums import MovementActions
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('winner', [0, 1, None])
def test_time_limit_rewards_belong_to_teams_once(size, winner):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size), max_steps=3)
    env.reset(seed=0)
    waits = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    _, rewards, *_ = env.step(waits)
    assert all(r == 0 for r in rewards.values())
    assert env.reward_events.events() == []
    if winner is not None:
        env.team_scores[winner] = 1
    env.step(waits)
    _, rewards, _, truncs, infos = env.step(waits)
    assert all(truncs.values())
    assert len(env.reward_events.events()) == 2
    assert all(event.agent_id is None for event in env.reward_events.events())
    assert all(r == 0 for r in env.last_reward_result.individual.values())
    for team in (0, 1):
        expected = 1000.0 if team == winner else -1000.0
        assert env.last_reward_result.team_total(team) == expected
        assert env.last_reward_result.shared_by_team[team] == expected
        assert env.last_team_rewards[team] == expected
        for agent in env.agents:
            if agent.team_index == team:
                assert rewards[agent.index] == expected
                assert infos[agent.index]['team_rewards'][team] == expected
    with pytest.raises(RuntimeError):
        env.step(waits)
    env.reset(seed=0)
    assert env.last_team_rewards == {0: 0, 1: 0}
    assert env.reward_events.events() == []


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('winner', [0, 1])
def test_goals_reward_only_the_match_winning_transition(size, winner):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    env.reset(seed=0)
    for goal in range(3):
        scorer = env.agents[winner * size + 1]
        scorer.state.pos = (14 if winner == 0 else 1, 4)
        env.ball = CustomBall(position=scorer.state.pos, carrier_id=scorer.index,
                              last_possessing_team=winner)
        actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
        actions[scorer.index] = MovementActions.RIGHT if winner == 0 else MovementActions.LEFT
        _, rewards, terms, _, _ = env.step(actions)
        for agent in env.agents:
            expected = (1000 if agent.team_index == winner else -1000) if goal == 2 else 0
            assert rewards[agent.index] == expected

        types = [event.type for event in env.reward_events.events()]
        assert types.count(SoccerEvent.GOAL_SCORED) == types.count(SoccerEvent.GOAL_CONCEDED) == 1
        match_events = {SoccerEvent.MATCH_WIN, SoccerEvent.MATCH_LOSS, SoccerEvent.MATCH_DRAW}
        assert sum(t in match_events for t in types) == (2 if goal == 2 else 0)

def test_v2_rewards_goals_for_both_teams():
    env = SoccerEnv(SoccerEnvConfig(rewards='v2'))
    env.reset(seed=0)
    env.reward_events.clear()
    env._award_goal(0, (15, 5))
    env._transition_rewards()
    assert env.last_team_rewards == {0: 100.0, 1: -100.0}