from types import SimpleNamespace

import numpy as np
import pytest

from soccer.movement import resolve_simultaneous_conflicts


def resolve(positions, targets, teams, seed=7, reverse=False):
    records = {
        i: dict(current_position=np.array(p), intended_position=np.array(t), status='valid')
        for i, (p, t) in enumerate(zip(positions, targets))
    }
    if reverse:
        records = dict(reversed(list(records.items())))
    agents = {i: SimpleNamespace(team_index=team) for i, team in enumerate(teams)}
    return resolve_simultaneous_conflicts(records, agents, np.random.default_rng(seed))


@pytest.mark.parametrize('targets', [[(4, 3), (5, 3)], [(4, 3), (3, 3)]])
def test_start_occupied_cell_is_blocked(targets):
    results = resolve([(3, 3), (4, 3)], targets, [0, 1])
    assert tuple(results[0]['resolved_position']) == (3, 3)
    assert results[0]['status'] == 'blocked_occupied_cell'
    assert tuple(results[1]['resolved_position']) == ((5, 3) if targets[1] == (5, 3) else (4, 3))


def test_teammates_are_both_blocked():
    results = resolve([(3, 3), (5, 3)], [(4, 3), (4, 3)], [0, 0])
    assert all(r['status'] == 'blocked_same_destination' for r in results.values())


def test_opposing_contest_is_seeded_and_has_one_winner():
    winners = set()
    for seed in range(20):
        args = ([(3, 3), (5, 3)], [(4, 3), (4, 3)], [0, 1])
        results = resolve(*args, seed=seed)
        reordered = resolve(*args, seed=seed, reverse=True)
        moved = [i for i, r in results.items() if r['status'] == 'moved']
        assert len(moved) == 1
        assert moved == [i for i, r in reordered.items() if r['status'] == 'moved']
        winners.update(moved)
    assert winners == {0, 1}


def test_mixed_multi_player_contest_requires_rule():
    with pytest.raises(NotImplementedError):
        resolve([(3, 3), (5, 3), (4, 2)], [(4, 3)] * 3, [0, 0, 1])
