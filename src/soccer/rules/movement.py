"""Soccer collision rules composed with the shared movement stages."""

from collections import defaultdict


def resolve_simultaneous_conflicts(
    validated_moves, agents_by_id, rng, *, loose_ball_position=None,
    contest_intensities=None, contest_events=None,
):
    """Block duplicate teammates, then resolve eligible opposing contenders.

    Loose-ball cards are reported before any live positions are committed.
    """
    resolved = {agent_id: record.copy() for agent_id, record in validated_moves.items()}
    occupied = {tuple(record['current_position']) for record in validated_moves.values()}
    contenders = defaultdict(list)

    for agent_id, record in resolved.items():
        record['resolved_position'] = record['current_position'].copy()
        if record['status'] != 'valid':
            continue
        current = tuple(record['current_position'])
        target = tuple(record['intended_position'])
        if target == current:
            record['status'] = 'waited'
        elif target in occupied:
            record['status'] = 'blocked_occupied_cell'
        else:
            contenders[target].append(agent_id)

    for target, agent_ids in sorted(contenders.items()):
        agent_ids = sorted(agent_ids)
        teams = defaultdict(list)
        for agent_id in agent_ids:
            teams[agents_by_id[agent_id].team_index].append(agent_id)
        eligible_ids = sorted(members[0] for members in teams.values() if len(members) == 1)
        if len(eligible_ids) <= 1:
            winners = eligible_ids
        elif len(eligible_ids) == 2:
            if target == loose_ball_position:
                intensities = {i: (contest_intensities or {}).get(i, 0) for i in eligible_ids}
                successes = {i: bool(rng.random() < (0.8 if intensities[i] else 0.5)) for i in eligible_ids}
                cards = [i for i in eligible_ids if intensities[i] and rng.random() < 0.5]
                successful = [i for i in eligible_ids if successes[i]]
                winner = None
                if not cards and successful:
                    strongest = max(intensities[i] for i in successful)
                    eligible = [i for i in successful if intensities[i] == strongest]
                    winner = eligible[0] if len(eligible) == 1 else eligible[int(rng.integers(2))]
                event = dict(position=target, players=eligible_ids, intensities=intensities,
                             successes=successes, red_cards=cards, winner=winner)
                if contest_events is not None:
                    contest_events.append(event)
                winners = [] if winner is None else [winner]
            else:
                winners = [eligible_ids[int(rng.integers(2))]]
        else:
            raise ValueError('Soccer movement supports at most two competing teams')

        for agent_id in agent_ids:
            record = resolved[agent_id]
            if agent_id in winners:
                record['resolved_position'] = record['intended_position'].copy()
                record['status'] = 'moved'
            else:
                record['status'] = 'blocked_same_destination'

    return resolved
