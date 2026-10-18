"""Domain-independent movement stages: build intentions, validate, commit.

The middle stage (conflict resolution) is soccer-specific and lives in
``soccer/rules/movement.py``. Cleanup note: the generic ``resolve_simultaneous_conflicts``
and the ``handle_simultaneous_movement`` wrapper were removed because only the
old snowplow environment used them.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from mosaic_multigrid.base import AgentID

from enums import MovementActions
from shared.grid import in_bounds, is_statically_blocked

# Type alias: (grid, agent, current_pos, intended_pos) -> bool
MoveAllowedFn = Callable[[object, object, tuple, tuple], bool]


def build_movement_intentions(agents, actions: dict[AgentID, MovementActions]) -> dict:
    """
    Convert each agent's raw action into a movement record. No live
    agent state is modified here.

    Parameters
    ----------
    agents : list of Agent (mosaic_multigrid.core.agent.Agent)
    actions : {agent.index: raw_action}
    """
    movement_intentions = {}
    for agent in agents:
        raw_action: MovementActions = actions[agent.index]
        # agent.state.pos returns a plain tuple for a single agent's
        # AgentState slice (not an ndarray), so wrap it before .copy()/
        # arithmetic rather than relying on a numpy-only API.
        current_position = np.array(agent.state.pos, dtype=int)

        if raw_action.name in MovementActions.__members__:
            chosen_action = raw_action
            action_valid = True
        else:
            chosen_action = MovementActions.DO_NOTHING
            action_valid = False

        movement_intentions[agent.index] = {
            "raw_action": raw_action,
            "chosen_action": chosen_action,
            "action_valid": action_valid,
            "current_position": current_position,
            "intended_position": current_position + MovementActions(chosen_action).value,
        }
    return movement_intentions


def validate_proposed_moves(
    movement_intentions: dict, agents_by_id: dict, grid, width: int, height: int, is_move_allowed: Optional[MoveAllowedFn] = None
) -> dict:
    """
    Reject movements that are illegal before any conflict resolution:
    invalid raw actions, out-of-bounds targets, statically blocked
    targets (walls / forbidden cells), and anything the domain-specific
    `is_move_allowed` hook rejects (e.g. wrong-direction lane).
    """
    validated_moves = {agent_id: record.copy() for agent_id, record in movement_intentions.items()}

    for agent_id, record in validated_moves.items():
        record["status"] = "valid"
        current_pos = tuple(record["current_position"])
        intended_pos = tuple(record["intended_position"])

        blocked = not record["action_valid"] or not in_bounds(width, height, intended_pos) or is_statically_blocked(grid, intended_pos)

        if not blocked and current_pos != intended_pos and is_move_allowed is not None:
            agent = agents_by_id[agent_id]
            if not is_move_allowed(grid, agent, current_pos, intended_pos):
                blocked = True

        if blocked:
            record["status"] = "blocked_invalid_action"
            record["intended_position"] = record["current_position"]

    return validated_moves


def commit_resolved_positions(agents_by_id: dict, moves_after_conflicts: dict) -> dict:
    """The only stage that writes resolved positions to live agent state."""
    for agent_id, record in moves_after_conflicts.items():
        agents_by_id[agent_id].state.pos = record["resolved_position"].copy()
    return moves_after_conflicts
