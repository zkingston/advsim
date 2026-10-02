"""The Emerald cartridge AI's tables, from `showdown/lib/emerald.json`.

The policy in `showdown/lib/policy.js` reads the same file, so the engine's
chooser and the oracle's cannot drift apart on the table.
"""
import numpy as np

from advsim import fileio

SOURCE = fileio.ROOT / 'showdown' / 'lib' / 'emerald.json'


def tables(types: tuple[str, ...], moves: list[str]) -> dict[str, np.ndarray]:
    data = fileio.read_json(SOURCE)
    order = np.array([[types.index(a), types.index(d), m] for a, d, m in data['type_order']], dtype=np.int32)
    skip = np.array([1 if m in data['nonstandard'] else 0 for m in moves], dtype=np.int32)
    return {'type_ai_order': order, 'move_ai_skip': skip}
