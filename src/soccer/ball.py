"""Soccer ball state and integer-grid kick trajectories."""

from mosaic_multigrid.core.world_object import Ball


def build_trajectory(start, target):
    """Bresenham cells, excluding the starting cell and including the target."""
    x, y = start
    end_x, end_y = target
    dx, dy = abs(end_x - x), -abs(end_y - y)
    step_x = 1 if x < end_x else -1
    step_y = 1 if y < end_y else -1
    error = dx + dy
    cells = []
    while (x, y) != (end_x, end_y):
        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x += step_x
        if doubled_error <= dx:
            error += dx
            y += step_y
        cells.append((x, y))
    return cells


class CustomBall(Ball):
    # Mosaic constructs its array-backed objects in __new__. Keep soccer
    # arguments out of Ball.__new__, which accepts color/index/reward only.
    def __new__(cls, position, **kwargs):
        return super().__new__(cls)

    def __init__(
        self, position, *, carrier_id=None, last_possessing_team=None,
        remaining_trajectory=None, speed=0,
    ):
        self.position = tuple(position)
        self.carrier_id = carrier_id
        self.last_possessing_team = last_possessing_team
        self.remaining_trajectory = list(remaining_trajectory or [])
        self.speed = speed
        self.kicker_id = None

    def can_overlap(self):
        return True

    @property
    def flight_state(self):
        if self.carrier_id is not None:
            return 'possessed'
        return 'in_flight' if self.remaining_trajectory else 'loose'

    def sync_with_carrier(self, agents_by_id):
        """Follow the committed position, including when movement was blocked."""
        if self.carrier_id is not None:
            carrier = agents_by_id[self.carrier_id]
            self.position = tuple(carrier.state.pos)
            self.last_possessing_team = carrier.team_index
            self.remaining_trajectory.clear()
            self.speed = 0
