"""Dense IDs and the dex tables: one column per field the engine reads, built
from the dump and `families.py`."""
from __future__ import annotations

from typing import Any

import numpy as np

from advsim.build import families as F


TYPES = ('???', 'Bug', 'Dark', 'Dragon', 'Electric', 'Fighting', 'Fire', 'Flying', 'Ghost',
         'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water')


STATS = ('atk', 'def', 'spa', 'spd', 'spe', 'acc', 'eva')


WEATHER = ('none', 'sun', 'rain', 'sand')


CATEGORIES = ('Physical', 'Special', 'Status')


GENDERS = ('', 'M', 'F', 'N')


# Exactly the targets this vocabulary uses; an unlisted one must fail the build
# rather than silently take some other move's meaning.
TARGETS = ('all', 'allAdjacent', 'allAdjacentFoes', 'allyTeam', 'any', 'foeSide', 'normal',
           'randomNormal', 'scripted', 'self')


SENTINELS = {'': 0, 'spore': -1, 'recoil': -2, 'all': -3, 'crit': -4, 'status': -5, 'boosts': -6}


TYPE_CODE = {0: 0, 1: 1, 2: -1, 3: -8}  # Showdown damageTaken -> chart code; -8 means immune


LEVEL_DAMAGE = 255


class Ids:
    """Dense IDs per category, alphabetical, with 0 reserved for 'none'."""

    def __init__(self, vocab: dict[str, list[str]], conditions: list[str]):
        self.tables = {k: [''] + list(v) for k, v in vocab.items()}
        self.tables['conditions'] = [''] + conditions
        self.index = {k: {n: i for i, n in enumerate(v)} for k, v in self.tables.items()}

    def get(self, table: str, name: str | None) -> int:
        if not name:
            return 0
        idx = self.index[table].get(name if table == 'species' else str(name).lower())
        if idx is None:
            raise KeyError(f'{name!r} is outside the {table} vocabulary')
        return idx

    def resolve(self, sym: Any) -> int:
        """Symbolic family parameters: types, stats, weather, conditions, species, sentinels.

        Two forms carry structure. 'stats:spa+spd' is a bitmask over the stats,
        because an item can raise more than one. 'x11/10' is a multiplier, stored
        the way Showdown stores one: truncated to 4096ths, so 1.1 is 4505.
        """
        if isinstance(sym, int):
            return sym
        if isinstance(sym, str) and sym.startswith('stats:'):
            mask = 0
            for name in sym[len('stats:'):].split('+'):
                mask |= 1 << STATS.index(name)
            return mask
        if isinstance(sym, str) and sym.startswith('x') and '/' in sym:
            num, den = sym[1:].split('/')
            return int(num) * 4096 // int(den)
        if sym in TYPES:
            return TYPES.index(sym)
        if sym in CATEGORIES:
            return CATEGORIES.index(sym)
        if sym in STATS:
            return STATS.index(sym)
        if sym in WEATHER:
            return WEATHER.index(sym)
        if sym in self.index['conditions']:
            return self.index['conditions'][sym]
        if sym in self.index['species']:
            return self.index['species'][sym]
        if sym in SENTINELS:
            return SENTINELS[sym]
        raise KeyError(f'family parameter {sym!r} resolves to nothing')


def pack_boosts(boosts: dict | None) -> int:
    """Seven signed nibbles, two's complement, so 'no boosts' is exactly zero."""
    if not boosts:
        return 0
    word = 0
    for i, stat in enumerate(STATS):
        word |= (int(boosts.get(stat, 0)) & 0xF) << (4 * i)
    return word


def flag_bits(raw: dict, moves: list[str]) -> dict[str, int]:
    """Only the flags families.py keeps; each must still exist in the dump."""
    present = {f for m in moves for f in raw['moves'][m].get('flags', {})}
    unknown = sorted(set(F.MOVE_FLAGS) - present)
    if unknown:
        raise ValueError(f'kept move flags that no move in the vocabulary has: {unknown}')
    names = sorted(F.MOVE_FLAGS)
    if len(names) > 16:
        raise ValueError(f'{len(names)} move flags do not fit in u16')
    return {n: 1 << i for i, n in enumerate(names)}


def special_types(raw: dict) -> np.ndarray:
    """Gen 3 derives a move's category from its type, and the gen3 `init` script
    holds the list. Hidden Power needs it at use time, because its type comes
    from the Pokemon and its category follows."""
    source = raw['scripts']['init']
    names = ('Fire', 'Water', 'Grass', 'Ice', 'Electric', 'Dark', 'Psychic', 'Dragon')
    for name in names:
        assert f'"{name}"' in source, f'gen3 init no longer lists {name} as special'
    return np.array([1 if t in names else 0 for t in TYPES], dtype=np.int32)


def type_immunities(raw: dict, ids: Ids) -> np.ndarray:
    """Per type, a bitmask of the statuses and weathers it cannot receive.

    Showdown keeps these in the same damageTaken table as type effectiveness,
    keyed by condition id, so Fire's burn immunity and Steel's sandstorm
    immunity come from exactly the same place as Water resisting Fire.
    """
    mask = np.zeros(len(TYPES), dtype=np.int32)
    for i, name in enumerate(TYPES):
        entry = raw['typechart'].get(name.lower())
        if not entry:
            continue
        for key, code in entry['damageTaken'].items():
            if key[:1].islower() and code == 3:
                cond = ids.index['conditions'].get(key)
                if cond is not None:
                    mask[i] |= 1 << cond
    return mask


def type_chart(raw: dict) -> np.ndarray:
    chart = np.zeros((len(TYPES), len(TYPES)), dtype=np.int8)
    for ai, att in enumerate(TYPES):
        for di, dfn in enumerate(TYPES):
            entry = raw['typechart'].get(dfn.lower())
            if entry:
                chart[ai, di] = TYPE_CODE[entry['damageTaken'].get(att, 0)]
    return chart


def move_tables(raw: dict, ids: Ids, flags: dict[str, int]) -> dict[str, np.ndarray]:
    names = ids.tables['moves']
    cols: dict[str, list[int]] = {c: [0] * len(names) for c in (
        'power', 'type', 'category', 'accuracy', 'priority', 'target', 'flags', 'crit_stage',
        'multihit_min', 'recoil_num', 'recoil_den', 'drain_num', 'drain_den',
        'heal_num', 'heal_den', 'fixed_damage', 'status', 'volatile', 'boosts', 'self_boosts',
        'sec_chance', 'sec_status', 'sec_volatile', 'sec_boosts', 'sec_self_boosts',
        'weather', 'switch_mode', 'type_from_mon', 'selfdestruct', 'ignore_immunity', 'pp',
        'family', 'p0')}
    for i, name in enumerate(names[1:], start=1):
        m = raw['moves'][name]
        put = lambda c, v: cols[c].__setitem__(i, int(v))
        put('power', resolve_power(raw, name, m))
        # Struggle's onModifyMove makes it typeless, which costs it STAB and
        # every type interaction, so the table carries ??? rather than Normal.
        put('type', 0 if name == 'struggle' else (TYPES.index(m['type']) if m['type'] in TYPES else 0))
        put('category', CATEGORIES.index(m['category']))
        put('accuracy', 255 if m['accuracy'] is True else m['accuracy'])
        put('priority', m.get('priority', 0))
        if m['target'] not in TARGETS:
            raise ValueError(f'{name}: target {m["target"]!r} is outside the known set {TARGETS}')
        put('target', TARGETS.index(m['target']))
        put('flags', sum(bit for f, bit in flags.items() if f in m.get('flags', {})))
        put('crit_stage', m.get('critRatio', 1) - 1)
        multihit = m.get('multihit')
        if multihit:
            lo = multihit if isinstance(multihit, int) else multihit[0]
            put('multihit_min', lo)
        for field, prefix in (('recoil', 'recoil'), ('drain', 'drain')):
            if m.get(field):
                put(f'{prefix}_num', m[field][0])
                put(f'{prefix}_den', m[field][1])
        heal = m.get('heal')
        if heal:
            put('heal_num', heal[0])
            put('heal_den', heal[1])
        if m.get('damage'):
            put('fixed_damage', LEVEL_DAMAGE if m['damage'] == 'level' else m['damage'])
        put('status', ids.get('conditions', m.get('status')))
        put('volatile', ids.get('conditions', m.get('volatileStatus')))
        put('boosts', pack_boosts(m.get('boosts')))
        put('self_boosts', pack_boosts((m.get('self') or {}).get('boosts')))
        secs = m.get('secondaries') or []
        if secs:
            sec = secs[0]
            put('sec_chance', sec.get('chance', 0))
            put('sec_status', ids.get('conditions', sec.get('status')))
            put('sec_volatile', ids.get('conditions', sec.get('volatileStatus')))
            put('sec_boosts', pack_boosts(sec.get('boosts')))
            put('sec_self_boosts', pack_boosts((sec.get('self') or {}).get('boosts')))
        put('weather', WEATHER.index({'sunnyday': 'sun', 'raindance': 'rain', 'sandstorm': 'sand'}[m['weather'].lower()]) if m.get('weather') else 0)
        put('switch_mode', 1 if m.get('forceSwitch') else (2 if m.get('selfSwitch') == 'copyvolatile' else (3 if m.get('selfSwitch') else 0)))
        put('type_from_mon', 1 if name == 'hiddenpower' else 0)
        # Explosion and Self-Destruct: the user faints, and gen 3 halves the
        # defender's Def in getDamage before the base-damage formula.
        put('selfdestruct', {'always': 1, 'ifHit': 2}.get(m.get('selfdestruct'), 0))
        # Showdown defaults ignoreImmunity to "status moves ignore it", but Thunder
        # Wave and Glare set it false, so Ground still blocks Thunder Wave.
        ignore = m.get('ignoreImmunity')
        put('ignore_immunity', 1 if (m['category'] == 'Status' if ignore is None else ignore) else 0)
        put('pp', m.get('pp', 0))
        fam, params = F.MOVES.get(name, ('', ()))
        put('family', family_id(F.MOVE_FAMILIES, fam, name))
        assert len(params) <= 1, f'{name}: a move family takes one parameter, move_p0'
        if params:
            put('p0', ids.resolve(params[0]))
    return {f'move_{k}': np.array(v, dtype=np.int32) for k, v in cols.items()}


def resolve_power(raw: dict, name: str, m: dict) -> int:
    """Two moves get their power from a callback; both are constant in this format."""
    if name == 'return':
        assert 'happiness' in m['callbacks'].get('basePowerCallback', ''), 'Return no longer reads happiness'
        return 255 * 10 // 25
    if name == 'hiddenpower':
        powers = {raw['moves'][k]['basePower'] for k in raw['moves'] if k.startswith('hiddenpower') and k != 'hiddenpower'}
        assert powers == {70}, f'Hidden Power variants have powers {powers}, expected 70'
        return 70
    return m['basePower']


def family_id(order: tuple[str, ...], fam: str, name: str) -> int:
    """0 = data only. Families come first, then one-offs, one ID each."""
    if not fam:
        return 0
    if fam == F.ONE_OFF:
        return len(order) + 1 + sorted(one_offs(order)).index(name)
    return order.index(fam) + 1


def one_offs(order: tuple[str, ...]) -> list[str]:
    """The one-off entries of the table whose families are `order`."""
    for families, src in ((F.MOVE_FAMILIES, F.MOVES), (F.ABILITY_FAMILIES, F.ABILITIES), (F.ITEM_FAMILIES, F.ITEMS)):
        if order is families:
            return sorted(n for n, (f, _) in src.items() if f == F.ONE_OFF)
    raise ValueError('not a family order')


def simple_tables(raw: dict, ids: Ids, table: str, mapping: dict, order: tuple[str, ...], prefix: str) -> dict[str, np.ndarray]:
    names = ids.tables[table]
    cols = {c: [0] * len(names) for c in ('family', 'p0', 'p1', 'p2', 'p3')}
    for i, name in enumerate(names[1:], start=1):
        fam, params = mapping.get(name, ('', ()))
        cols['family'][i] = family_id(order, fam, name)
        for j, sym in enumerate(params):
            cols[f'p{j}'][i] = ids.resolve(sym)
    return {f'{prefix}_{k}': np.array(v, dtype=np.int32) for k, v in cols.items()}


def accuracy_tables(ids: Ids) -> dict[str, np.ndarray]:
    """The accuracy modifiers, which sit beside an ability's family rather than
    inside it: Hustle carries one as well as its Attack multiplier."""
    names = ids.tables['abilities']
    cols = {c: [0] * len(names) for c in ('acc_mod', 'acc_whose', 'acc_when', 'acc_prio')}
    for i, name in enumerate(names[1:], start=1):
        if name not in F.ACCURACY:
            continue
        for key, sym in zip(('acc_mod', 'acc_whose', 'acc_when', 'acc_prio'), F.ACCURACY[name]):
            cols[key][i] = ids.resolve(sym)
    return {f'ability_{k}': np.array(v, dtype=np.int32) for k, v in cols.items()}


def species_tables(raw: dict, ids: Ids) -> dict[str, np.ndarray]:
    names = ids.tables['species']
    type1, type2 = [0] * len(names), [0] * len(names)
    nums = [0] * len(names)  # national dex number, which is how items filter species
    base = np.zeros((len(names), 6), dtype=np.int32)
    for i, name in enumerate(names[1:], start=1):
        s = raw['species'][name]
        nums[i] = s['num']
        type1[i] = TYPES.index(s['types'][0])
        type2[i] = TYPES.index(s['types'][1]) if len(s['types']) > 1 else type1[i]
        base[i] = [s['baseStats'][k] for k in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')]
    return {'species_type1': np.array(type1, dtype=np.int32),
            'species_type2': np.array(type2, dtype=np.int32),
            'species_num': np.array(nums, dtype=np.int32),
            'species_base_stats': base}
