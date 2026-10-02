"""Pack the generated team pool into the array `reset` draws from.

`gen_pool.js` writes one JSON team per line with Showdown's own stats; this
turns names into dex IDs and lays them out as int16 [N, 6, 20]. Uncompressed:
1M teams is 240 MB and loads in under a second, where the same pool as JSONL
takes tens of seconds to parse.
"""
from __future__ import annotations

import itertools

import numpy as np

from advsim import artifacts, fileio

# Column order inside a Pokemon's 20 int16 slots.
COLUMNS = ('species', 'level', 'gender', 'ability', 'item', 'hp_type', 'maxhp',
           'atk', 'def', 'spa', 'spd', 'spe',
           'move1', 'move2', 'move3', 'move4', 'pp1', 'pp2', 'pp3', 'pp4')
WIDTH = len(COLUMNS)
PARTY = 6
LIMITS = {'stat': 506, 'maxhp': 506, 'pp': 64}  # measured over 18K Pokemon; int16 has room


def pack(pool_path=None, limit: int | None = None, ids: dict | None = None) -> np.ndarray:
    ids = ids or artifacts.load_ids()
    index = {table: {name: i for i, name in enumerate(ids[table])}
             for table in ('species', 'moves', 'abilities', 'items')}
    types = {name: i for i, name in enumerate(ids['types'])}
    genders = {name: i for i, name in enumerate(ids['genders'])}
    rows = fileio.iter_jsonl(pool_path or fileio.ARTIFACTS / 'pool.jsonl')
    teams: list[np.ndarray] = []
    for team in (itertools.islice(rows, limit) if limit else rows):
        block = np.zeros((PARTY, WIDTH), dtype=np.int16)
        for slot, mon in enumerate(team['mons']):
            if slot >= PARTY:
                raise ValueError(f'team {len(teams)} has more than {PARTY} Pokemon')
            c = COLUMNS.index
            block[slot, c('species')] = index['species'][mon['species']]
            block[slot, c('level')] = mon['level']
            block[slot, c('gender')] = genders[mon['gender']]
            block[slot, c('ability')] = index['abilities'][mon['ability']]
            block[slot, c('item')] = index['items'][mon['item']] if mon['item'] else 0
            block[slot, c('hp_type')] = types[mon['hpType']]
            block[slot, c('maxhp')] = mon['maxhp']
            block[slot, c('atk'):c('spe') + 1] = mon['stats']
            for j, move in enumerate(mon['moves']):  # 0 pads a set with fewer than four
                block[slot, c('move1') + j] = index['moves'][move]
                block[slot, c('pp1') + j] = mon['maxpp'][j]
        teams.append(block)
    return np.stack(teams)


def validate(teams: np.ndarray, ids: dict) -> None:
    if teams.dtype != np.int16:
        raise ValueError(f'pool must be int16, got {teams.dtype}')
    if (teams < 0).any():
        raise ValueError('negative value in the pool; every field is an ID or a count')
    # gender 0 is genderless, which is what Showdown stores for Lunatone and friends,
    # not a missing value; Cute Charm and Attract treat it as neither M nor F.
    checks = {'species': (0, len(ids['species']) - 1), 'level': (1, 100), 'gender': (0, 3),
              'ability': (1, len(ids['abilities']) - 1), 'item': (0, len(ids['items']) - 1),
              'hp_type': (0, len(ids['types']) - 1), 'maxhp': (1, LIMITS['maxhp'])}
    for name, (lo, hi) in checks.items():
        col = teams[:, :, COLUMNS.index(name)]
        if col.min() < lo or col.max() > hi:
            raise ValueError(f'{name} out of range: [{col.min()}, {col.max()}] not within [{lo}, {hi}]')
    stats = teams[:, :, 7:12]
    if stats.max() > LIMITS['stat']:
        raise ValueError(f'stat {stats.max()} above the measured maximum {LIMITS["stat"]}')
    pp = teams[:, :, 16:20]
    if pp.max() > LIMITS['pp']:
        raise ValueError(f'max PP {pp.max()} above the measured maximum {LIMITS["pp"]}')
    moves = teams[:, :, 12:16]
    if ((moves[:, :, 0] == 0) & (teams[:, :, 0] != 0)).any():
        raise ValueError('a Pokemon with no first move')
    if (((moves == 0) & (pp != 0)) | ((moves != 0) & (pp == 0))).any():
        raise ValueError('move and PP padding disagree')


def build(pool_path=None, limit: int | None = None, write: bool = True) -> dict:
    ids = artifacts.load_ids()
    teams = pack(pool_path, limit, ids)
    validate(teams, ids)
    arrays = {'teams': teams}
    if write:
        fileio.write_npz(fileio.ARTIFACTS / 'pool.npz', arrays)
        artifacts.write_manifest({'pool': {'sha256': fileio.hash_arrays(arrays), 'teams': int(teams.shape[0])}},
                                 showdown=artifacts.read_manifest()['showdown'])
    return {'teams': teams}
