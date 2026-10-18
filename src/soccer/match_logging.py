"""Versioned match traces, independent of observations and reward calculation."""

from copy import deepcopy
from enum import Enum
import json
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic, Enum)):
        return json_value(value.value if isinstance(value, Enum) else value.item())
    return value


class MatchLog:
    """Keep current-match history; optionally stream every record to JSONL."""

    def __init__(self, path=None, show_events=False, append=False):
        self.show_events = show_events
        self.path = None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                # Exclusive creation prevents mixing independent runs.
                with self.path.open('x'):
                    pass
        self.match = self._last_match() if append and self.path is not None else 0
        self.records = []

    def _last_match(self):
        try:
            with self.path.open() as stream:
                last = None
                for line in stream:
                    if line.strip():
                        last = json.loads(line)
            return last.get('match', 0) if last else 0
        except FileNotFoundError:
            return 0

    def emit(self, kind, step, **data):
        record = json_value(dict(schema_version=1, match=self.match, step=step, type=kind, **deepcopy(data)))
        self.records.append(record)
        if self.show_events and kind not in ('actions', 'state'):
            print(json.dumps(record, allow_nan=False), flush=True)
        if self.path is not None:
            with self.path.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')

    def start(self, env):
        self.match += 1
        self.records = []
        self.emit('match_start', 0, seed=env.np_random_seed, team_size=env.config.n_players_per_team,
                  max_steps=env.max_steps, players=[dict(id=a.index, team=a.team_index,
                  original_role=a.original_role.value) for a in env.agents], ball_position=env.ball.position if env.ball else None)

    def transition(self, env, actions, before):
        self.emit('actions', env.step_count, actions=actions, players_before=before)
        for kind, entries in (('foul', env.last_fouls), ('red_card', env.last_red_cards)):
            for entry in entries:
                self.emit(kind, env.step_count, details=entry)
        for kind, attr in (('tackle', 'last_tackle_results'), ('kick', 'last_kick_results'),
                           ('dive', 'last_dive_results')):
            for player, result in getattr(env, attr).items():
                self.emit(kind, env.step_count, player=player, team=env.agents[player].team_index, details=result)
        for kind, attr in (('loose_ball_contest', 'last_loose_ball_contest'),
                           ('possession_transfer', 'last_possession_transfer'), ('pass_completed', 'last_pass_completed'),
                           ('restart', 'last_restart'), ('goal', 'last_goal')):
            value = getattr(env, attr)
            if value is not None:
                self.emit(kind, env.step_count, details=value)
        if env.last_goal is not None and not env.match_over:
            self.emit('restart', env.step_count, details=dict(type='kickoff',
                      team=env.last_goal['conceding_team'], position=env.ball.position))
        self.emit('state', env.step_count, score=env.team_scores, ball_position=env.ball.position if env.ball else None,
                  players=[dict(id=a.index, position=a.state.pos, suspended=a.suspended,
                                role=a.current_role.value) for a in env.agents])
        if env.match_over:
            self.emit('match_end', env.step_count, score=env.team_scores, winner=env.winner,
                      result=env.match_result, team_rewards=env.last_team_rewards)
