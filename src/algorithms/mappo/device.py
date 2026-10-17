"""Device selection shared by training, rollout collection and evaluation."""

import random

import numpy as np
import torch


def resolve_device(requested='auto'):
    name = str(requested)
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Supported devices: auto, cpu, cuda[:index], mps')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; check GPU hardware, driver and your PyTorch build')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device index {device.index} is unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable; use cpu or a supported Apple GPU/PyTorch build')
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
