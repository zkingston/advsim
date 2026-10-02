"""Skips for what a test needs (markers `oracle` and `built`, pyproject.toml), and
helpers for the tests that set a position by hand rather than replaying one."""
import shutil

import numpy as np
import pytest
import warp as wp

from advsim import fileio
from advsim.engine import kernels
from advsim.env import Gen3Env


BUILT = ('dex.npz', 'ids.json', 'layout.json', 'pool.npz', 'setdist.npz')


def pytest_collection_modifyitems(config, items):
    missing = [f for f in BUILT if not (fileio.ARTIFACTS / f).exists()]
    node = shutil.which('node') and (fileio.ROOT / 'showdown' / 'node_modules').exists()
    for item in items:
        if ('built' in item.keywords or 'oracle' in item.keywords) and missing:
            item.add_marker(pytest.mark.skip(reason=f'artifacts/{missing[0]} missing; run `advsim build`'))
        elif 'oracle' in item.keywords and not node:
            item.add_marker(pytest.mark.skip(reason='needs node and `npm ci` in showdown/'))


def make_env(batch: int, seed: int, rng_mode: str = 'train') -> Gen3Env:
    if not (fileio.ARTIFACTS / 'pool.npz').exists():
        pytest.skip('artifacts/pool.npz missing; run `advsim build`')
    e = Gen3Env(batch=batch, device='cpu', rng_mode=rng_mode)
    e.reset(seed=seed)
    return e


def poke(env: Gen3Env, name: str, index, value) -> None:
    host = env.arrays[name].numpy().copy()
    host[index] = value
    env.arrays[name].assign(host)


def step(env: Gen3Env, actions) -> None:
    log = wp.zeros((env.batch, 1), dtype=wp.uint32, device='cpu')
    wp.launch(kernels.step_idx, dim=env.batch, device='cpu', inputs=[
        env.state, env.dex, wp.array(np.array(actions, np.int32), dtype=wp.int32, device='cpu'), log, env.every])
