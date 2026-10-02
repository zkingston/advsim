"""The draw mappings must match sim/prng.js exactly; replay parity rests on them."""
import numpy as np
import pytest
import warp as wp

from advsim import oracle
from advsim.engine import kernels


@pytest.fixture(scope='module')
def cases():
    try:
        return oracle.prng_cases(2000, seed=99)
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_mappings_match_showdown(cases):
    col = lambda k, dt=np.int32: wp.array(np.array([c[k] for c in cases], dtype=dt), device='cpu')
    n = len(cases)
    out = [wp.zeros(n, dtype=wp.int32, device='cpu') for _ in range(3)]
    wp.launch(kernels.prng_mappings, dim=n, device='cpu', inputs=[
        col('raw', np.uint32), col('n'), col('m'), col('hi'), col('num'), col('den'), *out])
    assert list(out[0].numpy()) == [c['random_n'] for c in cases]
    assert list(out[1].numpy()) == [c['random_range'] for c in cases]
    assert list(out[2].numpy()) == [int(c['chance']) for c in cases]


def test_cases_are_not_degenerate(cases):
    assert len({c['raw'] for c in cases}) > len(cases) // 2
    assert any(c['chance'] for c in cases) and not all(c['chance'] for c in cases)
