"""The gate's benchmark is only meaningful if its three implementations agree.

This is the fast version of that check: Warp against JAX at a small batch. The
raw-CUDA implementation is covered by `bench/gate.py`, which needs nvcc.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'bench' / 'microbench'))

BATCH, STEPS = 256, 20


@pytest.fixture(scope='module')
def impls():
    warp = pytest.importorskip('warp')
    pytest.importorskip('jax')
    import jax_bench
    import warp_bench
    if not warp.get_device().is_cuda:
        pytest.skip('gate parity needs a CUDA device')
    return warp, warp_bench, jax_bench


def test_warp_and_jax_reach_the_same_state(impls):
    wp, w, j = impls
    dev = 'cuda:0'
    s = w.make_state(BATCH, dev)
    st, lanes = j.make_state(BATCH)
    for _ in range(STEPS):
        wp.launch(w.k_step, dim=BATCH, inputs=[s], device=dev)
        st = j.step_batch(st, lanes)
    wp.synchronize()
    assert w.checksum(s, BATCH, dev) == int(j.checksum(st))


def test_index_map_matches_the_dense_step(impls):
    """Stepping every slot through an index map must equal stepping them densely."""
    import numpy as np
    wp, w, j = impls
    dev = 'cuda:0'
    idx = wp.array(np.arange(BATCH, dtype=np.int32), dtype=wp.int32, device=dev)
    dense, mapped = w.make_state(BATCH, dev), w.make_state(BATCH, dev)
    for _ in range(STEPS):
        wp.launch(w.k_step, dim=BATCH, inputs=[dense], device=dev)
        wp.launch(w.k_step_idx, dim=BATCH, inputs=[mapped, idx], device=dev)
    wp.synchronize()
    assert w.checksum(dense, BATCH, dev) == w.checksum(mapped, BATCH, dev)
