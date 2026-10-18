"""BallRules: cohesive rules operating on the shared SoccerEnv state."""

from soccer.entities.agent import SoccerAgentRole
from soccer.entities.ball import build_trajectory


class BallRules:
    def _launch_kick(self, agent, request):
        """Invalid game choices are no-ops, with a reason for diagnostics."""
        ball = self.ball
        if ball is None or ball.carrier_id != agent.index:
            return 'not_carrier'
        dx, dy = request['displacement']
        limit = 3 if agent.current_role is SoccerAgentRole.GOALKEEPER else 2
        if max(abs(dx), abs(dy)) > limit:
            return 'out_of_range'
        if dx == dy == 0:
            return 'zero_displacement'
        x, y = agent.state.pos
        target = (int(x) + dx, int(y) + dy)
        ball.position = (int(x), int(y))
        # Preserve the aimed line; stop at the last on-pitch trajectory cell.
        # Clamping the target coordinates first would change diagonal angles.
        ball.remaining_trajectory = []
        for cell in build_trajectory(ball.position, target):
            if not (0 <= cell[0] < self.width and 0 <= cell[1] < self.height):
                break
            ball.remaining_trajectory.append(cell)
        ball.speed = request['strength'] if ball.remaining_trajectory else 0
        ball.kicker_id = agent.index
        ball.last_possessing_team = agent.team_index
        ball.carrier_id = None
        if self.free_kick_taker == agent.index:
            self.free_kick_taker = None
        return 'launched'

    def _receive_ball(self, receiver, agents_by_id):
        ball = self.ball
        previous_team = ball.last_possessing_team
        kicker_id = ball.kicker_id
        if previous_team is not None and previous_team != receiver.team_index:
            self.last_possession_transfer = {
                'from_agent': kicker_id,
                'to_agent': receiver.index,
                'from_team': previous_team,
                'to_team': receiver.team_index,
            }
        if (
            kicker_id is not None and kicker_id != receiver.index
            and agents_by_id[kicker_id].team_index == receiver.team_index
        ):
            self.last_pass_completed = {
                'from_agent': kicker_id, 'to_agent': receiver.index,
                'team': receiver.team_index,
            }
        ball.carrier_id = receiver.index
        ball.last_possessing_team = receiver.team_index
        ball.remaining_trajectory.clear()
        ball.speed = 0
        ball.kicker_id = None

    def _advance_ball(self, agents_by_id):
        """Check every traversed cell against resolved player positions."""
        ball = self.ball
        occupants = {tuple(a.state.pos): a for a in self.active_players}
        for _ in range(ball.speed):
            if not ball.remaining_trajectory:
                break
            ball.position = ball.remaining_trajectory.pop(0)
            if self._try_dives(ball.position, agents_by_id):
                break
            if self._check_goal():
                break
            receiver = occupants.get(ball.position)
            if receiver is not None:
                self._receive_ball(receiver, agents_by_id)
                break
        if not ball.remaining_trajectory:
            ball.speed = 0

    def _validate_dives(self, requests):
        self._dive_start_occupied = {tuple(a.state.pos) for a in self.active_players}
        for agent_id, displacement in requests.items():
            agent = self.agents[agent_id]
            result = {'status': 'ready', 'success': None}
            self.last_dive_results[agent_id] = result
            dx, dy = displacement
            if agent.current_role is not SoccerAgentRole.GOALKEEPER:
                result['status'] = 'invalid_role'
                continue
            if max(abs(dx), abs(dy)) != 1:
                result['status'] = 'invalid_displacement'
                continue
            x, y = agent.state.pos
            target = (int(x) + dx, int(y) + dy)
            result['target'] = target
            if not self.is_move_allowed(self.grid, agent, agent.state.pos, target):
                result['status'] = 'invalid_target'
                continue
            self._dive_targets[agent_id] = target

    def _try_dives(self, position, agents_by_id, dribbler_id=None):
        """One dive attempt: regular GK 70%, temporary GK 30%."""
        candidates = [i for i, target in self._dive_targets.items()
                      if target == position and self.last_dive_results[i]['status'] == 'ready'
                      and (dribbler_id is None or agents_by_id[i].team_index != agents_by_id[dribbler_id].team_index)]
        if not candidates:
            return False
        if len(candidates) > 1:
            raise AssertionError(
                f'Invalid goalkeeper state: players {candidates} share dive target {position}'
            )
        occupied = {tuple(a.state.pos) for a in self.active_players if a.index != dribbler_id}
        if position in self._dive_start_occupied or position in occupied:
            for i in candidates:
                self.last_dive_results[i]['status'] = 'blocked_occupied_cell'
            return False
        agent_id = candidates[0]
        keeper = agents_by_id[agent_id]
        probability = 0.7 if keeper.original_role is SoccerAgentRole.GOALKEEPER else 0.3
        success = bool(self.np_random.random() < probability)
        self.last_dive_results[agent_id].update(
            status='saved' if success else 'failed', success=success, save_probability=probability,
        )
        if not success:
            return False
        if dribbler_id is not None:
            dribbler = agents_by_id[dribbler_id]
            movement = self.last_movement_results[dribbler_id]
            dribbler.state.pos = tuple(movement['current_position'])
            movement['resolved_position'] = movement['current_position'].copy()
            movement['status'] = 'blocked_by_dive'
        keeper.state.pos = position
        if agent_id in self.last_movement_results:
            movement = self.last_movement_results[agent_id]
            movement['resolved_position'] = movement['current_position'].copy()
            movement['resolved_position'][:] = position
            movement['status'] = 'dived'
        self._receive_ball(keeper, agents_by_id)
        return True

