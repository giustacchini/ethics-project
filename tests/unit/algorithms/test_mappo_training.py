import numpy as np
import pytest
import torch

from algorithms.mappo.compute_gae import compute_advantages
from algorithms.mappo.device import resolve_device, seed_everything
from algorithms.mappo.experience_collector import ExperienceCollector
from algorithms.mappo.mappo_update import ppo_update
from algorithms.mappo.teams import create_teams
from mappo_train import TrainingConfig, train
from soccer.config import SoccerEnvConfig
from soccer.observation.encoding import encode_team
from soccer.env import SoccerEnv
from watch import watch


def setup(size=3, max_steps=3, training=(0, 1), device='cpu'):
    seed_everything(4)
    env = SoccerEnv(SoccerEnvConfig(n_players_per_team=size, max_steps=max_steps))
    obs, _ = env.reset(seed=4)
    _, matrix, state, _ = encode_team(obs, 0, max_steps=max_steps)
    teams = create_teams(matrix.shape[-1], len(state), size * 2, hidden=16, device=device)
    return env, teams, ExperienceCollector(env, teams, training, seed=4)


def test_gae_stops_at_terminal_and_time_limit_outcomes():
    rewards = torch.tensor([0., 1000., 0., -1000.])
    values = torch.zeros(4)
    advantages, returns = compute_advantages(rewards, values, torch.tensor([False, True, False, True]),
                                              torch.tensor(99999.), gamma=1, lam=1)
    assert returns.tolist() == [1000., 1000., -1000., -1000.]
    assert torch.equal(advantages, returns)
    _, bootstrapped = compute_advantages(torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                                         torch.tensor(5.), gamma=1, lam=1)
    assert bootstrapped.tolist() == [5., 5.]


@pytest.mark.parametrize('size', [2, 3])
def test_collection_preserves_chunk_boundaries_and_team_reward_once(size):
    env, teams, collector = setup(size=size, max_steps=3)
    buffers, _, completed = collector.collect(2)
    assert completed == []
    buffers, bootstrap, completed = collector.collect(4)
    assert len(completed) == 2
    assert completed[0]['episode'] == 1 and completed[1]['episode'] == 2
    assert collector.episodes == 2
    for team in (0, 1):
        data = buffers[team].as_tensors()
        assert data['obs'].shape[:2] == (4, size)
        assert data['states'].ndim == 2 and data['values'].shape == (4,)
        assert data['actions']['target'].shape == (4, size, 2)
        assert data['active_masks'].shape == (4, size)
        assert data['dones'].tolist() == [True, False, False, True]
        assert data['rewards'][0].item() in (-1000, 1000)
        assert data['rewards'][1:3].tolist() == [0, 0]
        assert bootstrap[team].shape == ()
        assert not data['log_probs'].requires_grad
    assert env.step_count == 0  # collector explicitly reset after the final match


def test_random_opponent_is_not_updated_and_critic_has_one_sample_per_transition():
    _, teams, collector = setup(max_steps=2, training=(0,))
    opponent = [p.detach().clone() for p in teams[1].actor.parameters()]
    buffers, bootstrap, _ = collector.collect(5)
    data = buffers[0].as_tensors()
    advantages, returns = compute_advantages(data['rewards'], data['values'], data['dones'], bootstrap[0])
    pair = teams[0]
    stats = ppo_update(pair.actor, pair.critic, pair.actor_optimizer, pair.critic_optimizer,
                       data, advantages, returns, epochs=1, n_minibatches=20)
    assert all(np.isfinite(value) for value in stats.values())
    assert stats['critic_samples'] == 5
    assert stats['actor_samples'] <= 15
    assert all(torch.equal(old, new) for old, new in zip(opponent, teams[1].actor.parameters()))


def test_all_inactive_rollout_skips_actor_but_trains_critic():
    _, teams, collector = setup(max_steps=2, training=(0,))
    buffers, bootstrap, _ = collector.collect(3)
    data = buffers[0].as_tensors()
    data['active_masks'][:] = False
    pair = teams[0]
    before = [p.detach().clone() for p in pair.actor.parameters()]
    advantages, returns = compute_advantages(data['rewards'], data['values'], data['dones'], bootstrap[0])
    stats = ppo_update(pair.actor, pair.critic, pair.actor_optimizer, pair.critic_optimizer,
                       data, advantages, returns, epochs=1)
    assert stats['actor_samples'] == 0 and stats['actor_loss'] == 0
    assert stats['critic_samples'] == 3
    assert all(torch.equal(old, new) for old, new in zip(before, pair.actor.parameters()))


def test_free_kick_masks_forced_waiting_players():
    env, _, collector = setup()
    env.free_kick_taker = 1
    collector.observations = env.gen_obs()
    assert collector.encode(0)[3].tolist() == [False, True, False]
    assert collector.encode(1)[3].tolist() == [False, False, False]


def test_device_selection_and_unavailable_requests(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert resolve_device('auto') == torch.device('cpu')
    with pytest.raises(RuntimeError, match='CUDA requested'):
        resolve_device('cuda')
    with pytest.raises(RuntimeError, match='MPS requested'):
        resolve_device('mps')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    assert resolve_device('auto') == torch.device('cuda')
    with pytest.raises(RuntimeError, match='index'):
        resolve_device('cuda:1')


@pytest.mark.parametrize('self_play,size', [(False, 2), (True, 3)])
def test_training_checkpoint_and_headless_watch(tmp_path, self_play, size):
    config = TrainingConfig(team_size=size, updates=1, rollout_steps=4, max_steps=2,
                            epochs=1, minibatches=2, hidden=16, device='cpu',
                            self_play=self_play, output=str(tmp_path / 'run'))
    result = train(config)
    assert result['checkpoint'].exists()
    assert (tmp_path / 'run' / 'metrics.jsonl').exists()
    assert len(result['metrics'][0]['completed_matches']) == 2
    results = watch(result['checkpoint'], matches=2, device='cpu', headless=True, self_play=self_play,
                    output_root=tmp_path / 'evaluations')
    import json
    directory, = (tmp_path / 'evaluations').iterdir()
    report = json.loads((directory / 'statistics.json').read_text())
    assert report['completed_matches'] == 2
    assert len(report['per_match']) == 2
    assert report['teams']['0']['totals']['draws'] == 2
    assert report['teams']['0']['averages_per_match']['draws'] == 1
    assert (directory / 'events.jsonl').exists()
    assert (directory / 'evaluation.json').exists()
    assert len(results) == 2
    assert all(match['transitions'] == 2 for match in results)
    with pytest.raises(FileExistsError):
        train(config)


def test_checkpoint_interval_and_resume(tmp_path):
    output = tmp_path / 'run'
    config = TrainingConfig(team_size=2, updates=3, rollout_steps=2, max_steps=2,
                            epochs=1, minibatches=1, hidden=8, device='cpu',
                            checkpoint_interval=2, output=str(output))
    result = train(config)
    assert result['checkpoint'].name == 'update_0003.pt'
    assert (output / 'update_0002.pt').exists()
    assert not (output / 'update_0001.pt').exists()

    resumed = train(TrainingConfig(updates=1, rollout_steps=2, max_steps=2, epochs=1,
                                   minibatches=1, hidden=8, device='cpu',
                                   checkpoint_interval=1, resume=str(result['checkpoint'])))
    assert resumed['checkpoint'].name == 'update_0004.pt'
    assert len(resumed['metrics']) == 1


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
                         + (['mps'] if torch.backends.mps.is_available() else []))
def test_rollout_and_update_tensors_stay_on_selected_device(device):
    _, teams, collector = setup(max_steps=2, training=(0,), device=device)
    buffers, bootstrap, _ = collector.collect(3)
    data = buffers[0].as_tensors()
    assert all(tensor.device.type == device for key, tensor in data.items() if key != 'actions')
    assert all(tensor.device.type == device for tensor in data['actions'].values())
    advantages, returns = compute_advantages(data['rewards'], data['values'], data['dones'], bootstrap[0])
    pair = teams[0]
    stats = ppo_update(pair.actor, pair.critic, pair.actor_optimizer, pair.critic_optimizer,
                       data, advantages, returns, epochs=1)
    assert np.isfinite(stats['critic_loss'])
