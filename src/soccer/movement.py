"""Soccer collision rules composed with the shared movement stages."""

from collections import defaultdict

from shared.movement import (
    build_movement_intentions,
    validate_proposed_moves,
    commit_resolved_positions,
)


def resolve_simultaneous_conflicts(validated_moves, agents_by_id, rng):
    """Resolve start-occupied cells and two-player empty-cell contests.

    Multi-player contests need an agreed rule and are rejected explicitly.
    Ball possession and loose-ball contests are handled in a later stage.
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

    for agent_ids in contenders.values():
        agent_ids = sorted(agent_ids)
        if len(agent_ids) == 1:
            winners = agent_ids
        elif len({agents_by_id[i].team_index for i in agent_ids}) == 1:
            winners = []
        elif len(agent_ids) == 2:
            winners = [agent_ids[int(rng.integers(2))]]
        else:
            raise NotImplementedError('Mixed-team contests with more than two players need a rule')

        for agent_id in agent_ids:
            record = resolved[agent_id]
            if agent_id in winners:
                record['resolved_position'] = record['intended_position'].copy()
                record['status'] = 'moved'
            else:
                record['status'] = 'blocked_same_destination'

    return resolved


def handle_simultaneous_movement(
    agents, grid, actions, width, height, rng, is_move_allowed=None,
):
    """Validate and resolve the joint action before committing any position."""
    agents_by_id = {agent.index: agent for agent in agents}
    intentions = build_movement_intentions(agents, actions)
    validated = validate_proposed_moves(
        intentions, agents_by_id, grid, width, height, is_move_allowed,
    )
    resolved = resolve_simultaneous_conflicts(validated, agents_by_id, rng)
    return commit_resolved_positions(agents_by_id, resolved)
