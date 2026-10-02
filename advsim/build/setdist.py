"""The set-distribution table `determinize` samples hidden sets from.

Every distinct Pokemon row in the pool, counted. A row is a pool row (same 20
columns) with its moves sorted by id: the protocol never shows a move's slot,
so slot order is not part of what a set is. Gender is zeroed: the generator
draws it apart from the set, so it has its own table, `gender[sp, g]`. Rows are
grouped by species, so a species' candidates are one contiguous run:
`start[sp]`, `count[sp]`. Over 1M teams that is 1,346 rows and at most 40 per
species, few enough to enumerate rather than sample.
"""
from __future__ import annotations

import numpy as np

from advsim import fileio
from advsim.build.build_pool import COLUMNS

MOVES = COLUMNS.index('move1')
PP = COLUMNS.index('pp1')
GENDER = COLUMNS.index('gender')


def canonical(rows: np.ndarray) -> np.ndarray:
    """Pool rows with each Pokemon's moves (and their PP) sorted by move id."""
    order = np.argsort(rows[:, MOVES:MOVES + 4], axis=1, kind='stable')
    out = rows.copy()
    out[:, MOVES:MOVES + 4] = np.take_along_axis(rows[:, MOVES:MOVES + 4], order, 1)
    out[:, PP:PP + 4] = np.take_along_axis(rows[:, PP:PP + 4], order, 1)
    return out


def table(teams: np.ndarray, n_species: int) -> dict[str, np.ndarray]:
    rows = canonical(teams.reshape(-1, teams.shape[-1]))
    rows = rows[rows[:, 0] != 0]
    gender = np.zeros((n_species, 4), dtype=np.int32)
    np.add.at(gender, (rows[:, 0], rows[:, GENDER]), 1)
    rows[:, GENDER] = 0
    sets, weight = np.unique(rows, axis=0, return_counts=True)  # sorted, so species first
    count = np.bincount(sets[:, 0], minlength=n_species).astype(np.int32)
    start = (np.cumsum(count) - count).astype(np.int32)
    return {'sets': sets.astype(np.int16), 'weight': weight.astype(np.int32), 'start': start, 'count': count,
            'gender': gender,
            'species_weight': np.bincount(rows[:, 0], minlength=n_species).astype(np.int32)}


def build(teams: np.ndarray, n_species: int, write: bool = True) -> dict[str, np.ndarray]:
    out = table(teams, n_species)
    if write:
        fileio.write_npz(fileio.ARTIFACTS / 'setdist.npz', out)
    return out
