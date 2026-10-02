"""The packed pool is what `reset` draws from, so its invariants are load-bearing."""
import numpy as np
import pytest

from advsim import artifacts, fileio
from advsim.build import build_pool

LIMIT = 2000


@pytest.fixture(scope='module')
def teams():
    if not (fileio.ARTIFACTS / 'pool.jsonl').exists():
        pytest.skip('artifacts/pool.jsonl missing; run showdown/gen_pool.js')
    return build_pool.build(limit=LIMIT, write=False)['teams']


def test_shape_and_dtype(teams):
    assert teams.shape == (LIMIT, 6, 20)
    assert teams.dtype == np.int16


def test_values_stay_inside_the_vocabulary(teams):
    build_pool.validate(teams, artifacts.load_ids())  # raises on anything out of range


def test_every_pokemon_has_a_first_move_and_matching_pp(teams):
    moves, pp = teams[:, :, 12:16], teams[:, :, 16:20]
    assert (moves[:, :, 0] > 0).all()
    assert np.array_equal(moves == 0, pp == 0), 'move and PP padding must agree'


def test_short_movesets_are_zero_padded(teams):
    moves = teams[:, :, 12:16]
    short = (moves == 0).any(axis=2)
    assert short.any(), 'the pool should contain sets with fewer than four moves'
    for slot in np.argwhere(short)[:20]:
        row = moves[tuple(slot)]
        used = (row != 0).sum()
        assert (row[:used] != 0).all() and (row[used:] == 0).all(), 'padding must be trailing'


def test_genderless_is_zero_not_missing(teams):
    """Showdown stores '' for genderless, so 0 here is a value, not an absence."""
    assert (teams[:, :, 2] == 0).any()
    assert set(np.unique(teams[:, :, 2])) <= {0, 1, 2, 3}


def test_packing_is_deterministic(teams):
    again = build_pool.build(limit=LIMIT, write=False)['teams']
    assert fileio.hash_arrays({'teams': again}) == fileio.hash_arrays({'teams': teams})
