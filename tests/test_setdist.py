"""The set table: every pool Pokemon is one of its species' candidates, and
the counts are the pool's."""
import numpy as np
import pytest

from advsim import fileio
from advsim.build import setdist


def test_the_set_table_is_the_pool():
    path = fileio.ARTIFACTS / 'setdist.npz'
    if not path.exists():
        pytest.skip('artifacts/setdist.npz missing; run `advsim build`')
    t = fileio.read_npz(path)
    sets, start, count = t['sets'], t['start'], t['count']
    assert count.max() <= 40, 'SPEC §Determinization: at most 40 candidates per species'
    assert (np.bincount(sets[:, 0], weights=t['weight'], minlength=len(count)) == t['species_weight']).all()
    assert (t['gender'].sum(1) == t['species_weight']).all()
    for sp in np.nonzero(count)[0]:
        assert (sets[start[sp]:start[sp] + count[sp], 0] == sp).all()
    moves = sets[:, setdist.MOVES:setdist.MOVES + 4]
    assert (np.diff(moves, axis=1) >= 0).all(), 'moves sorted by id'
    rows = setdist.canonical(fileio.read_npz(fileio.ARTIFACTS / 'pool.npz')['teams'][:1000].reshape(-1, 20))
    rows[:, setdist.GENDER] = 0
    known = {r.tobytes() for r in sets}
    assert all(r.tobytes() in known for r in rows)
