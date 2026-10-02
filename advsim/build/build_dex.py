"""Turn the Showdown dump plus the team pool into packed dex tables:
`vocab.py` decides and checks what is covered, `tables.py` builds the columns.
"""
from __future__ import annotations

import pathlib

import numpy as np

from advsim import artifacts, fileio
from advsim.build import emerald
from advsim.build import families as F
from advsim.build import tables as T
from advsim.build import vocab as V


def build(pool_path=None, limit: int | None = None, write: bool = True) -> dict:
    raw = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')
    pool_path = pathlib.Path(pool_path) if pool_path else fileio.ARTIFACTS / 'pool.jsonl'
    vocab = V.vocabulary(raw)
    conditions = V.referenced_conditions(raw, vocab['moves'], vocab['abilities'])
    problems = V.check_coverage(raw, vocab, conditions) + V.generator_assumptions(raw, vocab)
    if pool_path.exists():
        problems += V.cross_check_pool(raw, vocab, pool_path, limit)
    if problems:
        raise SystemExit('build failed:\n  ' + '\n  '.join(problems))

    ids = T.Ids(vocab, conditions)
    flags = T.flag_bits(raw, vocab['moves'])
    arrays: dict[str, np.ndarray] = {}
    arrays.update(T.move_tables(raw, ids, flags))
    arrays.update(T.simple_tables(raw, ids, 'abilities', F.ABILITIES, F.ABILITY_FAMILIES, 'ability'))
    arrays.update(T.accuracy_tables(ids))
    arrays.update(T.simple_tables(raw, ids, 'items', F.ITEMS, F.ITEM_FAMILIES, 'item'))
    arrays.update(T.species_tables(raw, ids))
    arrays['type_chart'] = T.type_chart(raw)
    arrays['type_status_immune'] = T.type_immunities(raw, ids)
    arrays['type_is_special'] = T.special_types(raw)
    arrays.update(emerald.tables(T.TYPES, ids.tables['moves']))

    ids_json = {**ids.tables, 'types': list(T.TYPES), 'stats': list(T.STATS), 'weather': list(T.WEATHER),
                'genders': list(T.GENDERS),
                'categories': list(T.CATEGORIES), 'targets': list(T.TARGETS),
                'move_flags': [k for k, _ in sorted(flags.items(), key=lambda kv: kv[1])],
                'move_families': list(F.MOVE_FAMILIES) + T.one_offs(F.MOVE_FAMILIES),
                'ability_families': list(F.ABILITY_FAMILIES) + T.one_offs(F.ABILITY_FAMILIES),
                'item_families': list(F.ITEM_FAMILIES)}
    if write:  # tests build from a pool slice and must not clobber the real artifacts
        fileio.write_npz(fileio.ARTIFACTS / 'dex.npz', arrays)
        fileio.write_json(fileio.ARTIFACTS / 'ids.json', ids_json)
        artifacts.write_manifest(
            {'dex': {'sha256': fileio.hash_arrays(arrays), 'arrays': len(arrays)},
             'ids': {'sha256': fileio.hash_json(ids_json), **{k: len(v) for k, v in ids.tables.items()}}},
            showdown=raw['meta']['showdown'])
    return {'ids': ids_json, 'arrays': arrays, 'dead': sorted(F.DEAD)}
