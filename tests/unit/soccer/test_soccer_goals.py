import pytest

from enums import MovementActions, SoccerActionType
from soccer.entities.agent import SoccerAgentRole
from soccer.entities.ball import CustomBall
from soccer.env import SoccerEnv, SoccerEnvConfig


def shot(defending_team, row, occupant=None):
    env = SoccerEnv()
    env.reset(seed=0)
    shooter = env.agents[1 if defending_team == 1 else 4]
    goal_x = 15 if defending_team == 1 else 0
    shooter.state.pos = (goal_x - 2 if goal_x else 2, row)
    keeper = env.agents[3 if defending_team == 1 else 0]
    keeper.state.pos = (14 if goal_x else 1, 7)
    if occupant == 'keeper':
        keeper.state.pos = (goal_x, row)
    elif occupant == 'temporary_keeper':
        replacement = env.agents[4 if defending_team == 1 else 1]
        replacement.current_role = SoccerAgentRole.GOALKEEPER
        replacement.state.pos = (goal_x, row)
    elif occupant == 'field':
        env.agents[4 if defending_team == 1 else 1].state.pos = (goal_x, row)
    env.ball = CustomBall(position=shooter.state.pos, carrier_id=shooter.index,
                          last_possessing_team=shooter.team_index)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[shooter.index] = dict(type=int(SoccerActionType.KICK), movement=0,
                                 target=(5 if goal_x else 1, 3), strength=1)
    return env, actions


@pytest.mark.parametrize('defending_team', [0, 1])
@pytest.mark.parametrize('row', [4, 5, 6])
def test_all_goal_cells_score_once(defending_team, row):
    env, actions = shot(defending_team, row)
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert env.team_scores[1 - defending_team] == 1
    assert env.team_scores[defending_team] == 0
    assert env.ball.speed == 0
    assert env.ball.remaining_trajectory == []
    assert env.observation_space.contains(obs)
    assert all(i['goal_scored_by']['team'] == 1 - defending_team for i in infos.values())
    _, _, _, _, next_infos = env.step(actions)
    assert env.team_scores[1 - defending_team] == 1
    assert all('goal_scored_by' not in i for i in next_infos.values())
    env.reset(seed=0)
    assert env.team_scores == {0: 0, 1: 0}
    assert env.pending_goal is None


@pytest.mark.parametrize('occupant', ['keeper', 'temporary_keeper'])
def test_current_goalkeeper_catches_at_occupied_goal(occupant):
    env, actions = shot(1, 5, occupant)
    env.step(actions)
    assert env.team_scores == {0: 0, 1: 0}
    assert env.ball.carrier_id == (3 if occupant == 'keeper' else 4)


def test_field_player_cannot_block_goal_by_occupying_cell():
    env, actions = shot(1, 5, 'field')
    env.step(actions)
    assert env.team_scores[0] == 1
    assert env.last_pass_completed is None


@pytest.mark.parametrize('defending_team', [0, 1])
def test_dribbled_goal(defending_team):
    env, _ = shot(defending_team, 4)
    carrier = env.agents[1 if defending_team == 1 else 4]
    carrier.state.pos = (14 if defending_team else 1, 4)
    env.ball.position = tuple(carrier.state.pos)
    actions = {a.index: MovementActions.DO_NOTHING for a in env.agents}
    actions[carrier.index] = MovementActions.RIGHT if defending_team else MovementActions.LEFT
    env.step(actions)
    assert env.team_scores[1 - defending_team] == 1
    assert carrier.state.carrying is None


def test_goal_in_intermediate_cell_stops_remaining_travel():
    env, _ = shot(1, 4)
    env.ball.carrier_id = None
    env.ball.position = (15, 3)
    env.ball.remaining_trajectory = [(15, 4), (15, 5)]
    env.ball.speed = 2
    env.step({a.index: MovementActions.DO_NOTHING for a in env.agents})
    assert env.last_goal['position'] == (15, 4)
    assert env.ball.position == (8, 5)
    assert env.team_scores[0] == 1


def test_baseline_single_cell_goal_can_be_configured():
    env = SoccerEnv(SoccerEnvConfig(goal_rows=(5,)))
    env.reset(seed=0)
    env.ball = CustomBall(position=(15, 4))
    env.step({a.index: MovementActions.DO_NOTHING for a in env.agents})
    assert env.team_scores == {0: 0, 1: 0}
