"""What the static page reads: the vocabulary the observation needs, and
PolicyTF checkpoints as JSON (float16 arrays in base64: half the download of
float32, and tests/test_web.py bounds what the rounding moves).

    uv run --group rl python tools/export_web.py dmg46

writes web/data/vocab.json, web/data/setdist.json and web/models/<tag>.json per tag
(artifacts/ppo_<tag>.npz), plus web/models/index.json listing them.
"""
from __future__ import annotations

import argparse
import base64

import numpy as np

from advsim import artifacts, fileio, net
from advsim.build.build_pool import COLUMNS
from advsim.engine import obs_layout as L

WEB = fileio.ROOT / 'web'
TABLES = ('species', 'moves', 'abilities', 'items', 'conditions', 'types')


def vocab() -> dict:
    ids, dex = artifacts.load_ids(), artifacts.load_dex()
    out = {t: ids[t] for t in TABLES}
    for k in ('species_type1', 'species_type2', 'move_type', 'move_category', 'move_type_from_mon', 'move_priority'):
        out[k] = dex[k].astype(int).tolist()
    out['type_chart'] = dex['type_chart'].astype(int).tolist()
    out['layout'] = {'MON': L.MON, 'ACTIVE': L.ACTIVE, 'FIELD': L.FIELD, 'MATCHUP': L.MATCHUP,
                     'MON_BASE': L.MON_BASE, 'ACTIVE_BASE': L.ACTIVE_BASE, 'FIELD_BASE': L.FIELD_BASE,
                     'MATCHUP_BASE': L.MATCHUP_BASE, 'MASK_BASE': L.MASK_BASE, 'HISTORY_BASE': L.HISTORY_BASE,
                     'STATS': L.STATS, 'STATS_BASE': L.STATS_BASE, 'ORDER_BASE': L.ORDER_BASE, 'OBS_DIM': L.OBS_DIM}
    return out


def setdist() -> dict:
    """The set table determinize samples from (build/setdist.py), for the page's MCTS:
    rows keep the pool's engine ids, which vocab.json names."""
    d = fileio.read_npz(fileio.ARTIFACTS / 'setdist.npz')
    return {k: d[k].astype(int).tolist() for k in ('sets', 'weight', 'start', 'count', 'gender', 'species_weight')} \
        | {'genders': artifacts.load_ids()['genders'], 'columns': list(COLUMNS)}


def damage_tables() -> dict:
    """What advsim/damage.py's formula reads, for web/js/damage.js: its tables and the ids it names."""
    from advsim import damage
    from advsim.engine._generated import ids
    t = {k: v.tolist() for k, v in damage.tables('cpu').items()}
    names = ('MOVE_HIDDENPOWER', 'MOVE_RETURN', 'MOVE_FLAIL', 'MOVE_REVERSAL', 'MOVE_FACADE', 'TYPE_NORMAL', 'TYPE_FIRE',
             'TYPE_ICE', 'TYPE_WATER', 'TYPE_ELECTRIC', 'TYPE_GROUND', 'ABILITY_HUGEPOWER', 'ABILITY_PUREPOWER',
             'ABILITY_HUSTLE', 'ABILITY_GUTS', 'ABILITY_THICKFAT', 'ABILITY_LEVITATE', 'ABILITY_FLASHFIRE',
             'ABILITY_WATERABSORB', 'ABILITY_VOLTABSORB', 'ABILITY_WONDERGUARD', 'ITEM_CHOICEBAND', 'COND_BRN', 'COND_PAR',
             'COND_PSN', 'COND_TOX', 'WEATHER_RAIN', 'WEATHER_SUN', 'CATEGORY_STATUS')
    return t | {'ids': {n: int(getattr(ids, n)) for n in names}}


def weights(tag: str) -> dict:
    arrays = fileio.read_npz(fileio.ARTIFACTS / f'ppo_{tag}.npz')
    arch = int(arrays['arch'][0]) if 'arch' in arrays else 1
    config = [int(c) for c in arrays['config']] if 'config' in arrays else []
    # web/js/policy.js has compact PolicyTF with at most the damage inputs among its options
    # (config: d, layers, heads, react, compact, damage, then the rest).
    if not (arch == 3 and config[4] == 1 and not any(config[6:])):
        raise SystemExit(f'{tag}: the page runs compact PolicyTF with no option but --damage')
    features = {'mon_numeric': net.MON_NUMERIC, 'active_flags': net.ACTIVE_FLAGS,
                'active_boosts': net.ACTIVE_BOOSTS, 'field_numeric': net.FIELD_NUMERIC}
    weights = {k: v for k, v in arrays.items() if k not in ('arch', 'config')}
    if not all(np.isfinite(v.astype('<f2')).all() for v in weights.values()):
        raise SystemExit(f'{tag}: a weight is out of float16 range')
    extra = {'damage': damage_tables()} if arch == 3 and config[5:6] == [1] else {}
    return {'features': features, 'dtype': 'f16', 'arch': arch, 'config': config, **extra,
            'weights': {k: {'shape': list(v.shape), 'data': base64.b64encode(np.ascontiguousarray(v, dtype='<f2').tobytes()).decode()}
                        for k, v in weights.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tags', nargs='+', help='checkpoint tags: artifacts/ppo_<tag>.npz')
    args = ap.parse_args()
    fileio.write_json(WEB / 'data' / 'vocab.json', vocab())
    fileio.write_json(WEB / 'data' / 'setdist.json', setdist())
    for tag in args.tags:
        fileio.write_json(WEB / 'models' / f'{tag}.json', weights(tag))
    fileio.write_json(WEB / 'models' / 'index.json', sorted(args.tags))


if __name__ == '__main__':
    main()
