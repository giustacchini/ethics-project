from Enums import SoccerEvent
from shared.events import Event
from soccer.rewards import make_soccer_reward_calculator

# Agents 0, 1, 2 are in team 1; agents 3, 4, 5 are in team 2
TEAMS_3V3 = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}


def test_goal_is_counted_once_for_the_team():
    calculator = make_soccer_reward_calculator()
    events = [
        Event(type=SoccerEvent.GOAL_SCORED, team_id=1),
        Event(type=SoccerEvent.GOAL_CONCEDED, team_id=2),
    ]

    result = calculator.calculate_rewards(events, TEAMS_3V3)

    # +100 for the team, not +300 for three players
    assert result.team_total(1) == 100.0
    assert result.team_total(2) == -100.0


def test_draw_is_punished_for_both_teams():
    calculator = make_soccer_reward_calculator()
    events = [
        Event(type=SoccerEvent.MATCH_DRAW, team_id=1),
        Event(type=SoccerEvent.MATCH_DRAW, team_id=2),
    ]

    result = calculator.calculate_rewards(events, TEAMS_3V3)

    assert result.team_total(1) == -1000.0
    assert result.team_total(2) == -1000.0