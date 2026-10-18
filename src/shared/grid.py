"""Small grid helpers used by the shared movement pipeline (shared/movement.py).

Cleanup note: ``agent_at`` and ``is_cell_free`` were removed. They were written
for the earlier snowplow environment and nothing in soccer (or the tests)
used them.
"""

from __future__ import annotations


def in_bounds(width: int, height: int, pos) -> bool:
    """True if `pos` = (x, y) lies inside a `width` x `height` grid."""
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def is_statically_blocked(grid, pos) -> bool:
    """True if the grid's static layer (walls, forbidden cells, ...) blocks this cell."""
    cell = grid.get(*pos)
    return cell is not None and not cell.can_overlap()
