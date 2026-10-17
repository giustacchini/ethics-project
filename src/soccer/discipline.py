"""DisciplineRules: cohesive rules operating on the shared SoccerEnv state."""

from copy import deepcopy

from soccer.agent import SoccerAgentRole
from soccer.ball import CustomBall
from soccer.grid import SoccerGrid


class DisciplineRules:
    def _apply_red_card(self, offender_id, defer_concession=False):
        """Apply a confirmed card; return whether it awards an automatic goal."""
        offender = self.agents[offender_id]
        if offender.suspended:
            raise ValueError('Player is already suspended')
        was_keeper = offender.current_role is SoccerAgentRole.GOALKEEPER
        offender.suspended = True
        offender.state.carrying = None
        if self.ball is not None and self.ball.carrier_id == offender_id:
            self.ball.carrier_id = None
        teammates = [a for a in self.active_players if a.team_index == offender.team_index]
        self.last_red_card = {'player': offender_id, 'team': offender.team_index, 'replacement': None}
        self.last_red_cards.append(self.last_red_card)
        if len(teammates) <= 1:
            if defer_concession:
                return False
            self._automatic_concession(offender.team_index)
            return True
        if was_keeper:
            goal = (0 if offender.team_index == 0 else 15, 5)
            candidates = [a for a in teammates if a.current_role is SoccerAgentRole.FIELD_PLAYER]
            # Deterministic implementation choice: Manhattan distance, then ID.
            replacement = min(candidates, key=lambda a: (self._manhattan(a.state.pos, goal), a.index))
            replacement.current_role = SoccerAgentRole.GOALKEEPER
            occupied = {tuple(a.state.pos) for a in self.active_players if a.index != replacement.index}
            cells = [(x, y) for x in range(self.width) for y in range(self.height)
                     if (x, y) not in occupied and self.is_move_allowed(
                         self.grid, replacement, replacement.state.pos, (x, y))]
            replacement.state.pos = min(cells, key=lambda p: (self._manhattan(p, goal), *p))
            self.last_red_card['replacement'] = replacement.index
        return False

    def _validate_tackles(self, requests):
        """Validate from the same pre-transition state without changing it."""
        valid = []
        carrier_id = self.ball.carrier_id if self.ball is not None else None
        for tackler_id, target_id in requests.items():
            tackler, target = self.agents[tackler_id], self.agents[target_id]
            record = {
                'target': target_id, 'status': 'attempted', 'success': None,
                'intensity': 'aggressive' if tackler_id in self._aggressive_tacklers else 'normal',
                'red_card': False,
            }
            self.last_tackle_results[tackler_id] = record
            if tackler.suspended or target.suspended:
                record['status'] = 'invalid_suspended_player'
                continue
            if tackler.team_index == target.team_index:
                record['status'] = 'invalid_teammate'
                continue
            distance = max(abs(int(a) - int(b)) for a, b in zip(tackler.state.pos, target.state.pos))
            if distance != 1:
                record['status'] = 'invalid_distance'
            elif target_id != carrier_id:
                record['status'] = 'off_ball_foul'
            else:
                valid.append(tackler_id)
        return valid

    def _prepare_aggressive_tackles(self):
        """Independent draws; a card interrupts before any physical outcome."""
        for tackler_id in sorted(self._aggressive_tacklers):
            record = self.last_tackle_results[tackler_id]
            if record['status'] not in ('attempted', 'off_ball_foul'):
                continue
            if record['status'] == 'attempted':
                record['sampled_success'] = bool(self.np_random.random() < 0.8)
            record['red_card'] = bool(self.np_random.random() < 0.5)
            if record['red_card']:
                record['status'] = 'off_ball_foul'

    def _resolve_tackles(self, valid_tacklers, agents_by_id):
        """Resolve non-carded carrier tackles after movement, before ball travel."""
        cancelled_kickers = set()
        successful_tacklers = []
        for tackler_id in sorted(valid_tacklers):
            record = self.last_tackle_results[tackler_id]
            if record['red_card'] or record['status'] != 'attempted':
                continue
            target_id = record['target']
            # A movement collision may already have transferred possession.
            if self.ball.carrier_id != target_id:
                record['status'] = 'cancelled_possession_changed'
                record['new_carrier'] = self.ball.carrier_id
                continue
            success = (
                record['sampled_success'] if record['intensity'] == 'aggressive'
                else bool(self.np_random.random() < 0.5)
            )
            record.update(status='success' if success else 'failed', success=success)
            if not success:
                continue
            cancelled_kickers.add(target_id)
            successful_tacklers.append(tackler_id)
        # All challenges see the same carrier. Only then resolve their shared
        # physical effect, so one success cannot suppress another's draw.
        if successful_tacklers:
            target_id = self.last_tackle_results[successful_tacklers[0]]['target']
            x, y = agents_by_id[target_id].state.pos
            cells = [
                (int(cx), int(cy))
                for cx in range(max(0, int(x) - 1), min(self.width, int(x) + 2))
                for cy in range(max(0, int(y) - 1), min(self.height, int(y) + 2))
            ]
            self.ball.carrier_id = None
            self.ball.kicker_id = None
            self.ball.remaining_trajectory.clear()
            self.ball.speed = 0
            self.ball.position = cells[int(self.np_random.integers(len(cells)))]
            for tackler_id in successful_tacklers:
                self.last_tackle_results[tackler_id]['ball_position'] = self.ball.position
            occupant = next((a for a in self.active_players if tuple(a.state.pos) == self.ball.position), None)
            if occupant is not None:
                self._receive_ball(occupant, agents_by_id)
        return cancelled_kickers

    @staticmethod
    def _manhattan(first, second):
        return sum(abs(int(a) - int(b)) for a, b in zip(first, second))

    def _restart_positions(self, taker, spot, keeper=None, keeper_spot=None):
        """Plan placement without modifying players; preserve legal distant cells."""
        positions = {taker.index: spot}
        if keeper is not None:
            positions[keeper.index] = keeper_spot
        occupied = set(positions.values())
        displaced = []
        for agent in self.active_players:
            if agent.index in positions:
                continue
            current = tuple(agent.state.pos)
            if current not in occupied and self._manhattan(current, spot) >= 3:
                positions[agent.index] = current
                occupied.add(current)
            else:
                displaced.append(agent)
        for agent in displaced:
            candidates = [
                (x, y) for x in range(self.width) for y in range(self.height)
                if (x, y) not in occupied and self._manhattan((x, y), spot) >= 3
                and self.is_move_allowed(self.grid, agent, agent.state.pos, (x, y))
            ]
            if not candidates:
                raise RuntimeError('No legal restart placement for player')
            destination = min(candidates, key=lambda p: (self._manhattan(agent.state.pos, p), *p))
            positions[agent.index] = destination
            occupied.add(destination)
        return positions

    def _neutral_restart(self, restore_suspended=False):
        self._reset_formation(restore_suspended=restore_suspended)
        self.ball = CustomBall(position=(7 + int(self.np_random.integers(2)), 5))
        self.grid.set(*self.ball.position, self.ball)
        self.last_restart = {'type': 'neutral', 'position': self.ball.position}

    def _automatic_concession(self, team):
        """Award one goal with a consistent final ball state and no restart."""
        self.last_restart = {'type': 'automatic_goal', 'team': 1 - team}
        self.grid = SoccerGrid(self.width, self.height, goal_rows=self.config.goal_rows)
        for agent in self.agents:
            agent.state.carrying = None
        goal = (0 if team == 0 else 15, 5)
        if self.ball is None:
            self.ball = CustomBall(position=goal)
        self._award_goal(1 - team, goal, 'one_active_player')
        self.grid.set(*self.ball.position, self.ball)

    def _card_roster_restart(self):
        """Handle low rosters only after all simultaneous cards are applied."""
        remaining = {team: sum(a.team_index == team for a in self.active_players) for team in (0, 1)}
        if all(count <= 1 for count in remaining.values()):
            self._neutral_restart(restore_suspended=True)
            return True
        if min(remaining.values()) <= 1:
            team = min(remaining, key=remaining.get)
            self._automatic_concession(team)
            return True
        return False

    @staticmethod
    def _is_penalty_foul(team, position):
        x, y = position
        return (0 <= x <= 2 if team == 0 else 13 <= x <= 15) and 3 <= y <= 7

    def _resolve_foul_restart(self, foul_position=None):
        fouls = sorted(i for i, result in self.last_tackle_results.items()
                       if result['status'] in ('off_ball_foul', 'contest_disciplinary_foul'))
        if not fouls:
            return False
        # Snapshot locations before any card relocates a replacement keeper.
        self.last_fouls = []
        for player_id in fouls:
            offender = self.agents[player_id]
            taker = self.agents[self.last_tackle_results[player_id]['target']]
            position = tuple(int(v) for v in (taker.state.pos if foul_position is None else foul_position))
            self.last_fouls.append(dict(
                offender=player_id, fouled_player=taker.index,
                offending_team=offender.team_index, position=position,
            ))
            self.last_tackle_results[player_id]['status'] = 'foul'
        self.last_foul = deepcopy(self.last_fouls[0])
        if len(fouls) > 1:
            self.last_foul['offenders'] = fouls
        cards = [i for i in fouls if self.last_tackle_results[i].get('red_card', False)]
        for player_id in cards:
            self._apply_red_card(player_id, defer_concession=True)
        if len(cards) > 1:
            self.last_red_card = None  # use plural red-card metadata
        if cards and self._card_roster_restart():
            return True
        teams = {foul['offending_team'] for foul in self.last_fouls}
        if len(teams) == 2:
            self._neutral_restart()
            return True
        penalties = [foul for foul in self.last_fouls
                     if self._is_penalty_foul(foul['offending_team'], foul['position'])]
        candidates = penalties or self.last_fouls
        chosen = candidates[0] if len(candidates) == 1 else candidates[int(self.np_random.integers(len(candidates)))]
        self.last_foul = deepcopy(chosen)
        if len(fouls) > 1:
            self.last_foul['offenders'] = fouls
        offender = self.agents[chosen['offender']]
        taker = self.agents[chosen['fouled_player']]
        foul_spot = chosen['position']
        inside = bool(penalties)
        spot = ((2 if offender.team_index == 0 else 13), 5) if inside else foul_spot
        keeper = None
        keeper_spot = None
        if inside:
            keeper = next(a for a in self.active_players if a.team_index == offender.team_index
                          and a.current_role is SoccerAgentRole.GOALKEEPER)
            keeper_spot = (0 if offender.team_index == 0 else 15,
                           4 if int(self.np_random.integers(2)) == 0 else 6)
        positions = self._restart_positions(taker, spot, keeper, keeper_spot)
        self.last_restart = {
            'type': 'penalty' if inside else 'free_kick',
            'taker': taker.index, 'team': taker.team_index, 'position': spot,
        }
        self.free_kick_taker = None if inside else taker.index
        # Interrupt the entire joint action; discard any old loose/flight marker.
        self.grid = SoccerGrid(self.width, self.height, goal_rows=self.config.goal_rows)
        for agent in self.active_players:
            agent.state.pos = positions[agent.index]
            agent.state.carrying = None
        self.ball = CustomBall(position=spot, carrier_id=taker.index,
                               last_possessing_team=taker.team_index)
        if inside:
            temporary = keeper.original_role is not SoccerAgentRole.GOALKEEPER
            probability = 0.3 if temporary else 0.7
            saved = bool(self.np_random.random() < probability)
            self.last_restart.update(saved=saved, save_probability=probability)
            if saved:
                self._receive_ball(keeper, {a.index: a for a in self.agents})
                self.ball.sync_with_carrier({a.index: a for a in self.agents})
            else:
                self.ball.position = (0 if offender.team_index == 0 else 15, 5)
                self.ball.carrier_id = None
                self._check_goal()
                self.grid.set(*self.ball.position, self.ball)
        for agent in self.agents:
            agent.state.carrying = self.ball if self.ball.carrier_id == agent.index else None
        return True

    def _resolve_contest_cards(self, contest):
        cards = contest['red_cards']
        if len(cards) == 2:
            for player_id in cards:
                self._apply_red_card(player_id, defer_concession=True)
            self.last_red_card = None
            if not self._card_roster_restart():
                self._neutral_restart()
            return
        player_id = cards[0]
        target_id = next(i for i in contest['players'] if i != player_id)
        self.last_tackle_results[player_id] = dict(
            target=target_id, status='contest_disciplinary_foul', success=None,
            intensity='aggressive', red_card=True,
        )
        self._resolve_foul_restart(foul_position=contest['position'])

