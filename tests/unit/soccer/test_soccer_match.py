import pytest

from enums import MovementActions
from soccer.agent import SoccerAgentRole
from soccer.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def waits(env):
    return {a.index: MovementActions.DO_NOTHING for a in env.agents}


def arrange_goal(env, scoring_team):
    size = env.config.n_players_per_team
    scorer = env.agents[scoring_team * size + 1]
    scorer.state.pos = (14 if scoring_team == 0 else 1, 4)
    env.ball = CustomBall(position=scorer.state.pos, carrier_id=scorer.index,
                          last_possessing_team=scoring_team)
    actions = waits(env)
    actions[scorer.index] = MovementActions.RIGHT if scoring_team == 0 else MovementActions.LEFT
    return actions


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('scoring_team', [0, 1])
def test_goal_restart_preserves_match_and_restores_roles(size, scoring_team):
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size))
    env.reset(seed=0)
    expected = [a.state.pos for a in env.agents]
    actions = arrange_goal(env, scoring_team)
    env.agents[0].current_role = SoccerAgentRole.FIELD_PLAYER
    obs, _, terms, truncs, infos = env.step(actions)
    assert not any(terms.values()) and not any(truncs.values())
    assert [a.state.pos for a in env.agents] == expected
    assert all(a.current_role == a.original_role for a in env.agents)
    assert all(a.state.carrying is None for a in env.agents)
    assert env.ball.position == ((8, 5) if scoring_team == 0 else (7, 5))
    assert env.ball.carrier_id is None and env.ball.last_possessing_team is None
    assert env.step_count == 1
    assert obs[0]['score'][scoring_team] == 1
    assert infos[0]['goal_scored_by']['team'] == scoring_team
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize('team', [0, 1])
def test_three_goals_end_match(team):
    env = SoccerEnv()
    env.reset(seed=0)
    for goal in range(3):
        obs, _, terms, truncs, infos = env.step(arrange_goal(env, team))
        assert all(terms.values()) == (goal == 2)
        assert not any(truncs.values())
    assert env.winner == team and env.match_over
    assert obs[0]['score'][team] == 3
    assert obs[0]['match_result'] == team + 1
    assert infos[0]['winner'] == team
    with pytest.raises(RuntimeError, match='reset'):
        env.step(waits(env))
    env.reset(seed=0)
    assert env.team_scores == {0: 0, 1: 0}
    assert not env.match_over


@pytest.mark.parametrize('score,winner,result', [
    ({0: 2, 1: 1}, 0, 'left_win'),
    ({0: 0, 1: 1}, 1, 'right_win'),
    ({0: 1, 1: 1}, None, 'draw'),
])
def test_time_limit_outcomes(score, winner, result):
    env = SoccerEnv(max_steps=1)
    env.reset(seed=0)
    env.team_scores = score
    _, _, terms, truncs, infos = env.step(waits(env))
    assert all(truncs.values()) and not any(terms.values())
    assert env.winner == winner and env.match_result == result
    assert infos[0]['match_over']


def test_last_transition_goal_is_counted_before_outcome():
    env = SoccerEnv(max_steps=1)
    env.reset(seed=0)
    env.team_scores = {0: 0, 1: 1}
    _, _, terms, truncs, _ = env.step(arrange_goal(env, 0))
    assert env.match_result == 'draw'
    assert env.ball.position == (15, 4)  # no kickoff after a match-ending goal
    assert all(truncs.values()) and not any(terms.values())


def test_third_goal_at_time_limit_is_termination():
    env = SoccerEnv(max_steps=3)
    env.reset(seed=0)
    env.team_scores = {0: 2, 1: 0}
    env.step_count = 2
    _, _, terms, truncs, _ = env.step(arrange_goal(env, 0))
    assert all(terms.values()) and not any(truncs.values())
    assert env.winner == 0
