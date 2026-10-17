import json
from unittest.mock import Mock

import pytest

from enums import MovementActions, SoccerActionType
from soccer.env import SoccerEnv


def wait(env):
    return {a.index: MovementActions.DO_NOTHING for a in env.agents}


def test_trace_persists_across_resets_and_marks_terminal(tmp_path):
    path = tmp_path / 'events.jsonl'
    env = SoccerEnv(max_steps=1, event_log_path=path)
    env.reset(seed=12)
    actions = wait(env)
    env.step(actions)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['type'] for r in records] == ['match_start', 'actions', 'state', 'match_end']
    assert records[0]['seed'] == 12
    assert records[-1]['step'] == 1
    assert records[-1]['team_rewards'] == {'0': -1000.0, '1': -1000.0}
    actions[0] = 4
    assert env.match_log.records[1]['actions']['0'] == [0, 0]
    env.reset(seed=13)
    assert len(env.match_log.records) == 1
    assert env.match_log.records[0]['match'] == 2
    assert len(path.read_text().splitlines()) == 5
    with pytest.raises(FileExistsError):
        SoccerEnv(event_log_path=path)
    env.close()


def test_simultaneous_cards_and_fouls_logged_individually(tmp_path):
    env = SoccerEnv(event_log_path=tmp_path / 'events.jsonl')
    env.reset(seed=2)
    env.agents[1].state.pos = (4, 5)
    env.agents[4].state.pos = (5, 5)
    env.agents[2].state.pos = (6, 7)
    env.agents[5].state.pos = (7, 7)
    env.np_random = Mock(random=Mock(return_value=0.1), integers=Mock(return_value=0))
    actions = wait(env)
    for player, target in ((1, 4), (5, 2)):
        actions[player] = dict(type=int(SoccerActionType.AGGRESSIVE_TACKLE),
                               movement=0, target=(3, 3), strength=0, target_player_id=target)
    env.step(actions)
    records = env.match_log.records
    assert len([r for r in records if r['type'] == 'foul']) == 2
    assert {r['details']['player'] for r in records if r['type'] == 'red_card'} == {1, 5}
    assert [r['details']['type'] for r in records if r['type'] == 'restart'] == ['neutral']
    env.step(wait(env))
    assert len([r for r in records if r['type'] == 'red_card']) == 2
    env.close()


def test_goal_and_kickoff_recorded():
    env = SoccerEnv()
    env.reset(seed=1)
    env.ball.position = (15, 4)
    env.step(wait(env))
    goals = [r for r in env.match_log.records if r['type'] == 'goal']
    assert len(goals) == 1
    assert goals[0]['details']['team'] == 0
    assert any(r['type'] == 'restart' and r['details']['type'] == 'kickoff'
               for r in env.match_log.records)
    env.close()


def test_console_events_and_validation_trace(tmp_path, capsys):
    from soccer.validate_matches import run_match

    path = tmp_path / 'validation.jsonl'
    summary, trace = run_match(2, 0, max_steps=2, event_log_path=path, show_events=True)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert printed == [r for r in saved if r['type'] not in ('actions', 'state')]
    assert printed[0]['type'] == 'match_start'
    assert printed[-1]['type'] == 'match_end'
    repeated_summary, repeated_trace = run_match(2, 0, max_steps=2)
    assert not capsys.readouterr().out
    assert summary == repeated_summary
    assert trace == repeated_trace
