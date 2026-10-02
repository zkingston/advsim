"""The engine's baseline policies: maxdamage picks what play.js picked at every
decision of battles the oracle played with it (its moves and its Emerald
replacements are both deterministic), and random play is even."""
import pytest
import numpy as np
import warp as wp

from advsim import oracle
from advsim.engine import kernels_search, policy_heuristic as ph
from advsim.env import Gen3Env
from conftest import make_env


def picks(states: list[dict], policy: int) -> np.ndarray:
    n = len(states)
    env = Gen3Env(batch=n, device='cpu', rng_mode='replay')
    env.reset(seed=0)
    env.load_words(states)
    out = wp.zeros((n, 2), dtype=wp.int32, device='cpu')
    slots = np.repeat(np.arange(n, dtype=np.int32), 2)
    sides = np.tile(np.array([0, 1], dtype=np.int32), n)
    arr = lambda x: wp.array(x, dtype=wp.int32, device='cpu')
    wp.launch(kernels_search.heuristic_actions, dim=2 * n, device='cpu', inputs=[
        env.state, env.dex, arr(slots), arr(sides), arr(np.full(2 * n, policy, np.int32)), wp.uint32(1), 0, out])
    return out.numpy().copy()


@pytest.mark.oracle
def test_maxdamage_matches_play_js():
    cases = oracle.turn_cases(120, seed=31, turns=100, policy='maxdamage')
    states, want = [], []
    for c in cases:
        before = [c['before']] + [seg['after'] for seg in c['segments'][:-1]]
        for state, seg in zip(before, c['segments']):
            states.append(state)
            want.append(seg['choices'])
    got = picks(states, ph.MAXDAMAGE)
    want = np.array(want)
    wrong = np.nonzero((got != want).any(1))[0]
    assert len(states) > 1000
    assert not len(wrong), f'{len(wrong)} of {len(states)} differ; first {[(int(i), got[i].tolist(), want[i].tolist()) for i in wrong[:5]]}'


def test_every_pick_is_legal():
    env = make_env(64, seed=5)
    from test_env_api import play
    play(env, 20)
    mask = np.asarray(env.observe()[1].cpu() if hasattr(env.observe()[1], 'cpu') else env.observe()[1])
    for policy in range(5):
        got = picks_env(env, policy)
        assert all((mask[b, s] >> got[b, s]) & 1 for b in range(64) for s in range(2)), policy


def picks_env(env, policy):
    n = env.batch
    out = wp.zeros((n, 2), dtype=wp.int32, device='cpu')
    arr = lambda x: wp.array(np.asarray(x, np.int32), dtype=wp.int32, device='cpu')
    wp.launch(kernels_search.heuristic_actions, dim=2 * n, device='cpu', inputs=[
        env.state, env.dex, arr(np.repeat(np.arange(n), 2)), arr(np.tile([0, 1], n)), arr(np.full(2 * n, policy)),
        wp.uint32(3), 7, out])
    return out.numpy().copy()
