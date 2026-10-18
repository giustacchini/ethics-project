"""Descriptive fair-play statistics from versioned match traces; no reward changes."""

import argparse
import json
from pathlib import Path

from enums import SoccerActionType


COUNTERS = ('active_player_transitions', 'decision_opportunities', 'aggressive_tackle_requests',
            'aggressive_move_requests', 'resolved_challenges', 'resolved_aggressive_challenges',
            'fouls', 'red_cards', 'penalties_conceded', 'goals_conceded_due_to_too_few_players_on_pitch',
            'goals_scored', 'goals_conceded', 'wins', 'draws', 'losses')


def ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if denominator else None


def rates(counts, matches):
    result = dict(counts)
    for name in ('fouls', 'red_cards', 'aggressive_tackle_requests', 'aggressive_move_requests'):
        result[name + '_per_1000_active_player_transitions'] = ratio(counts[name], counts['active_player_transitions'], 1000)
        result[name + '_per_match'] = ratio(counts[name], matches)
    result['aggressive_challenge_fraction'] = ratio(counts['resolved_aggressive_challenges'], counts['resolved_challenges'])
    return result


def match_counts(records):
    start, end = records[0], records[-1]
    teams = {p['id']: p['team'] for p in start['players']}
    counts = {team: dict.fromkeys(COUNTERS, 0) for team in (0, 1)}
    challenges = {}
    known_decisions = True
    for row in records:
        kind, details = row['type'], row.get('details', {})
        if kind == 'actions':
            for player in row['players_before']:
                if player['suspended']:
                    continue
                c = counts[player['team']]
                c['active_player_transitions'] += 1
                if 'decision_eligible' not in player:
                    known_decisions = False
                eligible = player.get('decision_eligible', True)
                if eligible:
                    c['decision_opportunities'] += 1
                action = row['actions'].get(str(player['id']), row['actions'].get(player['id']))
                if not eligible or not isinstance(action, dict):
                    continue
                if action['type'] == SoccerActionType.AGGRESSIVE_TACKLE:
                    c['aggressive_tackle_requests'] += 1
                if action['type'] == SoccerActionType.MOVE and action.get('movement', 0) != 0 and action.get('contest_intensity', 0):
                    c['aggressive_move_requests'] += 1
        elif kind == 'foul':
            counts[details['offending_team']]['fouls'] += 1
        elif kind == 'red_card':
            counts[details['team']]['red_cards'] += 1
        elif kind == 'restart' and details['type'] == 'penalty':
            counts[1 - details['team']]['penalties_conceded'] += 1
        elif kind == 'goal' and details['reason'] == 'one_active_player':
            counts[details['conceding_team']]['goals_conceded_due_to_too_few_players_on_pitch'] += 1
        elif kind == 'tackle' and details['status'] in ('success', 'failed', 'foul'):
            challenges[row['step'], row['player']] = details['intensity'] == 'aggressive'
        elif kind == 'loose_ball_contest':
            for player in details['players']:
                intensities = details['intensities']
                challenges[row['step'], player] = bool(intensities.get(str(player), intensities.get(player, 0)))
    for (_, player), aggressive in challenges.items():
        counts[teams[player]]['resolved_challenges'] += 1
        counts[teams[player]]['resolved_aggressive_challenges'] += int(aggressive)
    for team, c in counts.items():
        score = end['score']
        c['goals_scored'] = score.get(str(team), score.get(team))
        c['goals_conceded'] = score.get(str(1 - team), score.get(1 - team))
        c['draws' if end['winner'] is None else 'wins' if end['winner'] == team else 'losses'] = 1
    return counts, known_decisions


def summarize(records):
    """Consume one trace stream, excluding incomplete matches from all aggregates."""
    totals = {team: dict.fromkeys(COUNTERS, 0) for team in (0, 1)}
    matches, pending, incomplete, known_decisions = [], [], 0, True
    for row in records:
        if row.get('schema_version') != 1:
            raise ValueError('Unsupported match trace schema')
        if row['type'] == 'match_start':
            incomplete += bool(pending)
            pending = [row]
        elif not pending or row['match'] != pending[0]['match']:
            raise ValueError('Event without a matching match_start')
        else:
            pending.append(row)
        if row['type'] == 'match_end':
            counts, known = match_counts(pending)
            known_decisions &= known
            matches.append(dict(match=row['match'], transitions=row['step'],
                                teams={team: rates(c, 1) for team, c in counts.items()}))
            for team in (0, 1):
                for key in COUNTERS:
                    totals[team][key] += counts[team][key]
            pending = []
    incomplete += bool(pending)
    if not known_decisions:
        for c in totals.values():
            c['decision_opportunities'] = None
        for match in matches:
            for c in match['teams'].values():
                c['decision_opportunities'] = None
    return dict(statistics_version=1, completed_matches=len(matches), incomplete_matches=incomplete,
                decision_context_available=known_decisions, matches=matches,
                teams={team: rates(c, len(matches)) for team, c in totals.items()})


# Measurements retained in the evaluation report; model losses/rewards stay elsewhere.
EVALUATION_COUNTS = ('fouls', 'red_cards', 'penalties_conceded', 'goals_conceded_due_to_too_few_players_on_pitch',
                     'aggressive_tackle_requests', 'aggressive_move_requests',
                     'resolved_challenges', 'resolved_aggressive_challenges',
                     'goals_scored', 'goals_conceded', 'wins', 'draws', 'losses')


def evaluation_report(summary):
    def project(team, matches):
        return dict(totals={key: team[key] for key in EVALUATION_COUNTS},
                    averages_per_match={key: ratio(team[key], matches) for key in EVALUATION_COUNTS},
                    exposure={key: team[key] for key in ('active_player_transitions', 'decision_opportunities')},
                    rates={key: value for key, value in team.items()
                           if key.endswith('_per_1000_active_player_transitions') or key == 'aggressive_challenge_fraction'})
    return dict(statistics_version=summary['statistics_version'],
                completed_matches=summary['completed_matches'], incomplete_matches=summary['incomplete_matches'],
                decision_context_available=summary['decision_context_available'],
                per_match=[dict(match=m['match'], transitions=m['transitions'],
                                teams={team: project(c, 1) for team, c in m['teams'].items()})
                           for m in summary['matches']],
                teams={team: project(c, summary['completed_matches']) for team, c in summary['teams'].items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    with args.trace.open() as stream:
        report = summarize(json.loads(line) for line in stream if line.strip())
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
