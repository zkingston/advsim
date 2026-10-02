"""obs_layout's vector from an InfoState: the live half of the observation,
which must equal the engine's `obs` kernel wherever the player could know.

The one gap is the legal mask under a hidden trap: Arena Trap and Magnet Pull
trap without saying so, the request says only `maybeTrapped`, and the player
cannot know what the engine does. The parity check leaves those masks out.
"""
from __future__ import annotations

import numpy as np

from advsim import artifacts
from advsim.engine import obs_layout as L
from advsim.live.infostate import BOOSTS, InfoState

IMMUNE = -8
PASS, FORCED = 10, 11


class Vocab:
    """Names to the engine's ids, and the tables the matchup reads."""

    def __init__(self) -> None:
        ids, dex = artifacts.load_ids(), artifacts.load_dex()
        self.index = {t: {n: i for i, n in enumerate(ids[t])} for t in ('species', 'moves', 'abilities', 'items', 'conditions', 'types')}
        types = ids['types']
        self.species_types = {n: (types[dex['species_type1'][i]], types[dex['species_type2'][i]])
                              for i, n in enumerate(ids['species']) if n}
        self.chart = dex['type_chart'].astype(int)
        self.move_type = dex['move_type']
        self.move_cat = dex['move_category']
        self.from_mon = dex['move_type_from_mon']
        self.priority = dex['move_priority']

    def id(self, table: str, name: str) -> int:
        return self.index[table].get(name, 0) if name else 0

    def type_id(self, name: str) -> int:
        return self.index['types'].get(name, 0)

    def effectiveness(self, t: int, d1: int, d2: int) -> int:
        m = self.chart[t, d1] + (self.chart[t, d2] if d2 != d1 else 0)
        return IMMUNE if m < -6 else max(-6, min(6, m))


def hp_percent(hp: int, maxhp: int) -> int:
    if hp <= 0 or maxhp <= 0:
        return 0
    pct = (100 * hp + maxhp - 1) // maxhp
    return 99 if pct == 100 and hp < maxhp else pct


def observe(st: InfoState, v: Vocab) -> np.ndarray:
    out = np.zeros(L.OBS_DIM, dtype=np.int16)
    me, foe = st.sides[st.me], st.sides[st.foe]
    M = len(L.MON)

    def put_mon(token: int, mon, own: bool, active: bool, pp):
        base = token * M
        row = {'present': 1, 'species': v.id('species', mon.species), 'level': mon.level,
               'hp_pct': mon.hp if not own else hp_percent(mon.hp, mon.maxhp),
               'status': 0 if mon.fainted else v.id('conditions', mon.status),
               'toxic_stage': mon.toxic_stage if mon.status == 'tox' else 0,
               'fainted': int(mon.fainted), 'active': int(active)}
        if own:
            row.update(hp=mon.hp, maxhp=mon.maxhp, ability=v.id('abilities', mon.ability), ability_known=1,
                       item=v.id('items', mon.item), item_known=1)
            for i, m in enumerate(mon.req_moves[:4]):
                row[f'move{i}'] = v.id('moves', m)
                if active and pp is not None:
                    row[f'pp{i}'] = pp[i]
        else:
            if mon.ability_known:
                row.update(ability=v.id('abilities', mon.ability), ability_known=1)
            if mon.item_known:
                row.update(item=v.id('items', mon.item), item_known=1)
            for i, m in enumerate(sorted(v.id('moves', m) for m in mon.moves)[:4]):
                row[f'move{i}'] = m
        for name, value in row.items():
            out[base + L.MON.index(name)] = value

    for i, name in enumerate(me.order):
        put_mon(i, me.mons[name], True, name == me.active, getattr(me, 'pp', None))
    seen = sorted((v.id('species', m.species), n) for n, m in foe.mons.items() if m.species)
    for rank, (_, name) in enumerate(seen):
        put_mon(6 + rank, foe.mons[name], False, name == foe.active, None)

    for token, side in enumerate((me, foe)):
        base = L.ACTIVE_BASE + token * len(L.ACTIVE)
        row = {f'boost_{k[:3] if k in ("accuracy", "evasion") else k}': side.boosts.get(k, 0) for k in BOOSTS}
        t0, t1 = side.types if side.types else ('???', '???')
        row.update(type0=v.type_id(t0), type1=v.type_id(t1), **{k: n for k, n in side.vol.items()})
        if token == 0 and side.choice_move:
            row['choice_move'] = v.id('moves', side.choice_move)
        for name, value in row.items():
            out[base + L.ACTIVE.index(name)] = value

    field = {'weather': st.weather, 'weather_turns': st.weather_turns, 'turn': min(st.turn, 1000),
             'spikes_own': me.spikes, 'spikes_foe': foe.spikes,
             'wish_own': int(me.wish > 0), 'wish_foe': int(foe.wish > 0)}
    for name, value in field.items():
        out[L.FIELD_BASE + L.FIELD.index(name)] = value

    for token, (side, other) in enumerate(((me, foe), (foe, me))):
        mon = side.mons.get(side.active)
        if mon is None or not other.types:
            continue
        if token == 0:
            moves = [(v.id('moves', m), mon.hp_type) for m in mon.req_moves[:4]]
        else:
            moves = [(m, None) for m in sorted(v.id('moves', m) for m in mon.moves)[:4]]
        d1, d2 = v.type_id(other.types[0]), v.type_id(other.types[1])
        own_types = {v.type_id(t) for t in side.types}
        for i, (m, hp_type) in enumerate(moves):
            if m == 0 or v.move_cat[m] == 2 or (v.from_mon[m] and hp_type is None):
                continue
            t = v.type_id(hp_type.capitalize()) if v.from_mon[m] else int(v.move_type[m])
            out[L.MATCHUP_BASE + token * 8 + 2 * i] = v.effectiveness(t, d1, d2)
            out[L.MATCHUP_BASE + token * 8 + 2 * i + 1] = int(t in own_types)

    out[L.MASK_BASE:L.MASK_BASE + 12] = (legal_mask(st) >> np.arange(12)) & 1
    out[L.HISTORY_BASE] = v.id('moves', me.last_move)
    out[L.HISTORY_BASE + 1] = v.id('moves', foe.last_move)
    # The turn in progress once anything has happened in it, the last one until then: as the engine
    # clears its record when a turn starts running, not at the `turn` line before the choice.
    movers = st.movers if st.mid_turn else st.last_movers
    order = list(movers)
    if len(order) == 2 and len({int(v.priority[v.id('moves', m)]) for m in movers.values()}) == 1:
        out[L.ORDER_BASE] = 1 if order[0] == st.me else -1
    for i, name in enumerate(me.order):
        for k, stat in enumerate(L.STATS):
            out[L.STATS_BASE + 5 * i + k] = me.mons[name].stats.get(stat, 0)
    return out


def legal_mask(st: InfoState) -> int:
    """What the request offers, as the engine's action bitmask."""
    req = st.request
    if not req or req.get('wait') or st.ended:
        return 1 << PASS  # a finished battle keeps its last request; it is stale
    me = st.sides[st.me]
    switches = 0
    for i, name in enumerate(me.order):
        mon = me.mons[name]
        if not mon.fainted and name != me.active:
            switches |= 1 << (4 + i)
    if req.get('forceSwitch'):
        return switches or 1 << PASS
    active = req['active'][0]
    trapped = active.get('trapped', False)
    moves = active['moves']
    # One move offered is a forced action (a lock or Struggle) unless it is a one-move set's own move.
    lone = len(moves) == 1 and (moves[0]['id'] == 'struggle' or len(me.mons[me.active].req_moves) > 1)
    if lone and not moves[0].get('disabled'):
        return (1 << FORCED) | (0 if trapped else switches)
    bits = sum(1 << i for i, m in enumerate(moves) if not m.get('disabled') and m.get('pp', 1) > 0)
    return (bits or 1 << FORCED) | (0 if trapped else switches)
