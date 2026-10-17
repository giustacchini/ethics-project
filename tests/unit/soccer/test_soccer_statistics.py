import json

from soccer.statistics import summarize
from soccer.env import SoccerEnv
from enums import MovementActions


def record(kind, step=0, **data):
    return dict(schema_version=1, match=1, step=step, type=kind, **data)


def test_statistics_intent_consequences_denominators_and_simultaneous_cards():
    rows = [record('match_start', players=[dict(id=0, team=0), dict(id=1, team=1)]),
            record('actions', 1, players_before=[dict(id=0, team=0, suspended=False, decision_eligible=True),
                                                dict(id=1, team=1, suspended=False, decision_eligible=False)],
                   actions={'0': dict(type=4), '1': dict(type=4)}),
            record('foul', 1, details=dict(offending_team=0)),
            record('foul', 1, details=dict(offending_team=1)),
            record('red_card', 1, details=dict(team=0)),
            record('red_card', 1, details=dict(team=1)),
            record('tackle', 1, player=0, details=dict(status='foul', intensity='aggressive')),
            record('loose_ball_contest', 1, details=dict(players=[0, 1], intensities={'0': 1, '1': 0})),
            record('restart', 1, details=dict(type='penalty', team=1)),
            record('goal', 1, details=dict(reason='one_active_player', conceding_team=0)),
            record('match_end', 1, score={'0': 0, '1': 1}, winner=1)]
    report = summarize(rows)
    left, right = report['teams'][0], report['teams'][1]
    assert left['aggressive_tackle_requests'] == 1
    assert right['aggressive_tackle_requests'] == 0
    assert right['decision_opportunities'] == 0
    assert left['red_cards'] == right['red_cards'] == 1
    assert left['resolved_challenges'] == left['resolved_aggressive_challenges'] == 1
    assert right['resolved_challenges'] == 1
    assert left['fouls_per_1000_active_player_transitions'] == 1000
    assert left['penalties_conceded'] == left['automatic_goals_conceded'] == 1
    assert left['losses'] == right['wins'] == 1
    assert left['goals_conceded'] == right['goals_scored'] == 1


def test_incomplete_matches_excluded_and_old_trace_context_unknown():
    start = record('match_start', players=[dict(id=0, team=0)])
    action = record('actions', 1, players_before=[dict(id=0, team=0, suspended=False)], actions={'0': [0, 0]})
    end = record('match_end', 1, score={'0': 0, '1': 0}, winner=None)
    report = summarize([start, action, start, action, end, start])
    assert report['completed_matches'] == 1
    assert report['incomplete_matches'] == 2
    assert report['teams'][0]['active_player_transitions'] == 1
    assert report['teams'][0]['decision_opportunities'] is None
    assert report['teams'][1]['fouls_per_1000_active_player_transitions'] is None
    assert summarize([start])['teams'][0]['fouls_per_match'] is None


def test_saved_trace_and_memory_statistics_agree(tmp_path):
    path = tmp_path / 'match.jsonl'
    env = SoccerEnv(max_steps=2, event_log_path=path)
    env.reset(seed=4)
    for _ in range(2):
        env.step({a.index: MovementActions.DO_NOTHING for a in env.agents})
    with path.open() as stream:
        saved = summarize(json.loads(line) for line in stream)
    assert saved == summarize(env.match_log.records)
    assert saved['teams'][0]['active_player_transitions'] == 6
    assert saved['teams'][0]['draws'] == 1
    env.close()
