"""
src/repro.py

Reproducibility helpers shared by every run_clean_* script.
"""

import os
import sys
import random
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Seed python, numpy and torch (CPU + CUDA)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """Give every DataLoader worker its own numpy / python stream.

    Datasets that hold a numpy Generator (``rng`` attribute) get it re-seeded per
    worker; without this each forked worker replays an identical jitter stream.
    """
    seed = torch.initial_seed() % (2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    # eval datasets keep rng=None (clips are seeded from the event timestamps)
    if info is not None and getattr(info.dataset, 'rng', None) is not None:
        info.dataset.rng = np.random.default_rng(seed)


def make_generator(seed: int = 42) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _git(*args) -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.check_output(['git', *args], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def env_snapshot() -> dict:
    """Software / hardware record written next to every result set."""
    snap = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
    }
    for pkg in ('numpy', 'pandas', 'sklearn', 'lightgbm', 'optuna', 'transformers',
                'librosa', 'shap'):
        try:
            snap[pkg] = __import__(pkg).__version__
        except Exception:
            snap[pkg] = None
    return snap
