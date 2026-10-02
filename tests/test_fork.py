"""`fork`: a child is its parent in every field but the RNG key, and a perft
position replayed through the engine's own fork matches Showdown leaf by leaf."""
import pytest
import numpy as np

from advsim import oracle
from conftest import make_env
from advsim.replay import replay
from test_env_api import play


def test_a_child_is_its_parent_but_for_the_key():
    env = make_env(8, seed=5)
    play(env, 30)
    env.fork([0, 1, 2, 3], [4, 5, 6, 7], salt=9)
    words = env.words()
    for name, arr in words.items():
        if name != 'rng_key':
            assert np.array_equal(arr[:4], arr[4:]), name
    assert (words['rng_key'][:4] != words['rng_key'][4:]).all()
    h = env.hash(position=True)
    assert (h[:4] == h[4:]).all()


@pytest.mark.oracle
def test_perft_through_fork():
    leaves = 0
    for cases in oracle.perft(7000003, turns=4, depth=1, budget=5000):
        if isinstance(cases, int):
            assert cases > 0, 'the position ended before perft reached it'
            continue
        leaves += len(cases)
        for index, b, problems in replay(cases, fork=True):
            assert not problems, (b, problems)
    assert leaves > 1
