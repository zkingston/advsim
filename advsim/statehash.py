"""The NumPy mirror of the canonical hash.

`engine/hash.py` runs on the device over engine state; this runs on the host
over a dict of arrays, which is what `export_state.js` hands back for a
Showdown battle. Both walk the word order in `engine/layout.py`, so a field
added there is hashed by both or neither.
"""
from __future__ import annotations

import numpy as np

from advsim.engine import layout

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX1 = np.uint64(0xBF58476D1CE4E5B9)
MIX2 = np.uint64(0x94D049BB133111EB)


def splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over='ignore'):
        z = x + GOLDEN
        z = (z ^ (z >> np.uint64(30))) * MIX1
        z = (z ^ (z >> np.uint64(27))) * MIX2
    return z ^ (z >> np.uint64(31))


def state_hash(words: dict[str, np.ndarray], position: bool = False) -> np.ndarray:
    """uint64[B] over `{field name: array with a leading batch axis}`.

    Missing fields hash as zero, which is what a partial export means: the
    exporter left the field at its default, not that the word is skipped.
    """
    fields = layout.POSITION if position else layout.FULL
    offsets = layout.word_offsets()
    batch = max(np.asarray(a).shape[0] for a in words.values())
    h = np.zeros(batch, dtype=np.uint64)
    for f in fields:
        arr = words.get(f.name)
        a = (np.zeros((batch, f.elems()), dtype=np.uint64) if arr is None
             else np.asarray(arr).astype(np.uint64).reshape(batch, f.elems()))
        idx = (np.arange(f.elems(), dtype=np.uint64) + np.uint64(offsets[f.name])) << np.uint64(32)
        h ^= np.bitwise_xor.reduce(splitmix64(idx[None, :] | a), axis=1)
    return h
