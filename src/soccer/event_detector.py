"""Turn what happened in one MOSAIC soccer step into our own soccer events.

Temporary bridge: MOSAIC only reports goals, passes and steals.
"""

from Enums import SoccerEvent
from shared.events import EventLog


def detect_events(env, infos: dict, match_over: bool) -> list:
    log = EventLog()
    step = env.step_count
    info = infos[0]  # MOSAIC copies the same event info to every agent

    def other(team):
        return 2 if team == 1 else 1

    # A goal: +100 for the scorers, -100 for the other team
    if "goal_scored_by" in info:
        team = info["goal_scored_by"]["team"]
        log.emit(SoccerEvent.GOAL_SCORED, team_id=team, step=step)
        log.emit(SoccerEvent.GOAL_CONCEDED, team_id=other(team), step=step)

    # A pass to a teammate: +2
    if "pass_completed" in info:
        team = info["pass_completed"]["team"]
        log.emit(SoccerEvent.PASS_COMPLETED, team_id=team, step=step)

    # A steal: the ball changes team (+5 / -5)
    if "steal_completed" in info:
        team = info["steal_completed"]["team"]
        log.emit(SoccerEvent.POSSESSION_RECOVERED, team_id=team, step=step)
        log.emit(SoccerEvent.POSSESSION_LOST, team_id=other(team), step=step)

    # End of match: win, loss or draw
    if match_over:
        score = env.team_scores
        if score[1] == score[2]:
            log.emit(SoccerEvent.MATCH_DRAW, team_id=1, step=step)
            log.emit(SoccerEvent.MATCH_DRAW, team_id=2, step=step)
        else:
            winner = 1 if score[1] > score[2] else 2
            log.emit(SoccerEvent.MATCH_WIN, team_id=winner, step=step)
            log.emit(SoccerEvent.MATCH_LOSS, team_id=other(winner), step=step)

    return log.events()