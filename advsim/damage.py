"""Damage ranges a player can work out from its observation: network inputs.

For each move that matters this turn, the least and most one use of it would
take off its target, as a fraction of the target's max HP: the own active's
four moves against the foe active, the foe active's revealed moves against
the own active, and for each own Pokemon the most it could deal to the foe
active and the most the foe active's revealed moves could deal to it.

Everything comes from the observation vector (obs_layout version 3), so the
engine's training, search and live play all compute the same thing, and
nothing hidden can leak in. The foe's stats are estimated from species and
level as Random Battle builds them (IV 31, EV 85: exact for three stats in
four; special attackers' Attack is overestimated). The Gen 3 formula keeps the
parts that move a range: level, power, the attack and defence stats with their
boosts, burn, weather, Flash Fire, STAB, type effectiveness, the 85-100% roll,
multi-hit counts, fixed damage, Flail and Reversal, Choice Band, Huge Power, Pure
Power, Hustle, Guts, Thick Fat, and the absorbing abilities when they are
known. Critical hits are left out. It is a feature, not the engine's damage:
the network reads it next to everything else.

A network holds the tables as a `Features` module's buffers. Loading them
inside a compiled forward (`tables`, behind its cache) split PolicyTF's graph
in ten and cost 14-15% of training throughput.
"""
from __future__ import annotations

import functools

import torch
from torch import nn

from advsim import artifacts, fileio
from advsim.engine import obs_layout as L
from advsim.engine._generated import ids

M, A = len(L.MON), len(L.ACTIVE)
mcol = L.MON.index
acol = L.ACTIVE.index
BOOSTED = (1, 2, 3, 4)  # atk, def, spa, spd among the stats' order (hp, atk, def, spa, spd, spe)
STATUSED = (ids.COND_BRN, ids.COND_PAR, ids.COND_PSN, ids.COND_TOX)


@functools.lru_cache(maxsize=None)
def tables(device: str) -> dict:
    """The dex columns the formula reads, as tensors on `device`, with the
    move data the dex arrays drop (the most hits, OHKO) and the formula's constants."""
    dex = artifacts.load_dex()
    names = artifacts.load_ids()['moves']
    raw = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')['moves']
    most, ohko = [0] * len(names), [0] * len(names)
    for i, name in enumerate(names):
        m = raw.get(name) or {}
        hits = m.get('multihit')
        most[i] = hits[1] if isinstance(hits, list) else (hits or 0)
        ohko[i] = int(bool(m.get('ohko')))
    t = lambda a: torch.as_tensor(a, device=device)
    return {'base': t(dex['species_base_stats']).float(), 'type1': t(dex['species_type1']).long(),
            'type2': t(dex['species_type2']).long(), 'power': t(dex['move_power']).float(),
            'mtype': t(dex['move_type']).long(), 'cat': t(dex['move_category']).long(),
            'fixed': t(dex['move_fixed_damage']).long(), 'least': t(dex['move_multihit_min']).float(),
            'most': t(most).float(), 'ohko': t(ohko).bool(), 'selfdestruct': t(dex['move_selfdestruct']).bool(),
            'special': t(dex['type_is_special']).bool(), 'chart': t(dex['type_chart']).float(),
            'statused': t(STATUSED), 'flail_at': t([2., 5., 10., 17., 33.]),
            'flail_power': t([200., 150., 100., 80., 40., 20.])}


def stage(s: torch.Tensor) -> torch.Tensor:
    """A boost stage's multiplier: (2 + s) / 2 up, 2 / (2 - s) down."""
    return torch.where(s >= 0, (2 + s) / 2, 2 / (2 - s))


def tokens(obs: torch.Tensor, T: dict) -> dict:
    """Per Pokemon token [n, 12, ...]: what the formula needs about it."""
    n = obs.shape[0]
    x = obs.long()
    mon = x[:, L.MON_BASE:L.MON_BASE + 12 * M].view(n, 12, M)
    act = x[:, L.ACTIVE_BASE:L.ACTIVE_BASE + 2 * A].view(n, 2, A)
    species, level = mon[..., mcol('species')], mon[..., mcol('level')].float()
    active = (mon[..., mcol('active')] * mon[..., mcol('present')]).bool()
    base = T['base'][species]  # [n, 12, 6]
    est = torch.floor((2 * base + 52) * level[..., None] / 100)
    est = torch.cat([est[..., :1] + level[..., None] + 10, est[..., 1:] + 5], -1)  # hp, atk, def, spa, spd, spe
    own = torch.arange(12, device=obs.device) < 6
    mine = x[:, L.STATS_BASE:L.STATS_BASE + 30].view(n, 6, 5).float()
    stats = torch.cat([torch.cat([est[:, :6, :1], mine], -1), est[:, 6:]], 1)  # own: the request's; foe: estimated
    maxhp = torch.where(own, mon[..., mcol('maxhp')].float(), est[..., 0])
    hpfrac = torch.where(own, mon[..., mcol('hp')].float() / maxhp.clamp(min=1), mon[..., mcol('hp_pct')].float() / 100)
    types = torch.stack([T['type1'][species], T['type2'][species]], -1)
    # The actives: current types, boosts and Flash Fire from their tokens (own 0, foe 1).
    side = torch.where(own, 0, 1)[None].expand(n, -1)
    cur = act.gather(1, side[..., None].expand(-1, -1, A))  # [n, 12, A] each token's side's active row
    types = torch.where(active[..., None], torch.stack([cur[..., acol('type0')], cur[..., acol('type1')]], -1), types)
    boosts = torch.zeros(n, 12, 6, device=obs.device)
    for k, name in zip(BOOSTED, ('boost_atk', 'boost_def', 'boost_spa', 'boost_spd')):
        boosts[..., k] = torch.where(active, cur[..., acol(name)].float(), 0.0)
    known = lambda what: torch.where(mon[..., mcol(f'{what}_known')] > 0, mon[..., mcol(what)], 0)
    return {'level': level, 'stats': stats * stage(boosts), 'maxhp': maxhp, 'hpfrac': hpfrac, 'types': types,
            'status': mon[..., mcol('status')], 'ability': known('ability'), 'item': known('item'),
            'flash_fire': active & (cur[..., acol('flash_fire')] > 0), 'moves': mon[..., mcol('move0'):mcol('move0') + 4],
            'active': active}


def hit(tok: dict, T: dict, att: torch.Tensor, dfn: torch.Tensor, move: torch.Tensor, weather: torch.Tensor):
    """(least, most) fraction of the defender's max HP one use takes: tokens
    `att` and `dfn` [n, K], moves [n, K, 4]."""
    g = lambda key, idx: tok[key].gather(1, idx) if tok[key].dim() == 2 else \
        tok[key].gather(1, idx[..., None].expand(-1, -1, tok[key].shape[-1]))
    a = {k: g(k, att)[:, :, None] for k in ('level', 'hpfrac', 'status', 'ability', 'item', 'flash_fire')}
    d = {k: g(k, dfn)[:, :, None] for k in ('maxhp', 'ability')}
    a_stats, d_stats = g('stats', att)[:, :, None], g('stats', dfn)[:, :, None]
    a_types, d_types = g('types', att)[:, :, None], g('types', dfn)[:, :, None]
    t = T['mtype'][move]
    hp = move == ids.MOVE_HIDDENPOWER
    t = torch.where(hp, ids.TYPE_NORMAL, t)  # its type is the holder's secret; count it neutral, no STAB
    special = T['special'][t]
    physical = ~special
    atk = torch.where(special, a_stats[..., 3], a_stats[..., 1])
    dfs = torch.where(special, d_stats[..., 4], d_stats[..., 2])
    ab, it, st = a['ability'], a['item'], a['status']
    statused = torch.isin(st, T['statused'])
    boost = torch.ones_like(atk)
    boost = torch.where(physical & ((ab == ids.ABILITY_HUGEPOWER) | (ab == ids.ABILITY_PUREPOWER)), boost * 2, boost)
    boost = torch.where(physical & (ab == ids.ABILITY_HUSTLE), boost * 1.5, boost)
    boost = torch.where(physical & (ab == ids.ABILITY_GUTS) & statused, boost * 1.5, boost)
    boost = torch.where(physical & (it == ids.ITEM_CHOICEBAND), boost * 1.5, boost)
    boost = torch.where((d['ability'] == ids.ABILITY_THICKFAT) & ((t == ids.TYPE_FIRE) | (t == ids.TYPE_ICE)),
                        boost * 0.5, boost)
    dfs = torch.where(T['selfdestruct'][move], dfs / 2, dfs)
    power = T['power'][move]
    frac = a['hpfrac']
    flail = T['flail_power'][torch.bucketize(torch.floor(48 * frac), T['flail_at'], right=True)]
    # The vocabulary's moves whose dex power is not their power (Gen 3 Random Battle has no Frustration,
    # Eruption, Magnitude, Low Kick, Present, Psywave or Super Fang).
    for m, value in ((ids.MOVE_RETURN, 102.0), (ids.MOVE_HIDDENPOWER, 70.0)):
        power = torch.where(move == m, value, power)
    power = torch.where((move == ids.MOVE_FLAIL) | (move == ids.MOVE_REVERSAL), flail, power)
    power = torch.where((move == ids.MOVE_FACADE) & statused, 140.0, power)

    def formula(p):
        base = torch.floor(torch.floor(torch.floor(2 * a['level'] / 5 + 2) * p * atk * boost / dfs.clamp(min=1)) / 50)
        # Gen 3's order: burn, weather, Flash Fire, +2; then the roll, STAB and type, below.
        base = torch.where(physical & (st == ids.COND_BRN) & (ab != ids.ABILITY_GUTS), torch.floor(base / 2), base)
        wx = torch.ones_like(base)
        wx = torch.where((weather == ids.WEATHER_RAIN) & (t == ids.TYPE_WATER), 1.5, wx)
        wx = torch.where((weather == ids.WEATHER_RAIN) & (t == ids.TYPE_FIRE), 0.5, wx)
        wx = torch.where((weather == ids.WEATHER_SUN) & (t == ids.TYPE_FIRE), 1.5, wx)
        wx = torch.where((weather == ids.WEATHER_SUN) & (t == ids.TYPE_WATER), 0.5, wx)
        base = torch.floor(base * wx)
        base = torch.where(a['flash_fire'] & (t == ids.TYPE_FIRE), torch.floor(base * 1.5), base)
        return base + 2

    e1 = T['chart'][t, d_types[..., 0]]
    e2 = torch.where(d_types[..., 1] != d_types[..., 0], T['chart'][t, d_types[..., 1]], 0.0)
    immune = (e1 < -1) | (e2 < -1)
    da = d['ability']
    immune = immune | ((da == ids.ABILITY_LEVITATE) & (t == ids.TYPE_GROUND)) \
        | ((da == ids.ABILITY_FLASHFIRE) & (t == ids.TYPE_FIRE)) | ((da == ids.ABILITY_WATERABSORB) & (t == ids.TYPE_WATER)) \
        | ((da == ids.ABILITY_VOLTABSORB) & (t == ids.TYPE_ELECTRIC))
    eff = torch.where(hp, 1.0, torch.exp2(e1 + e2))
    immune = immune | ((da == ids.ABILITY_WONDERGUARD) & (eff <= 1) & ~hp)
    stab = torch.where(~hp & ((t == a_types[..., 0]) | (t == a_types[..., 1])), 1.5, 1.0)
    base = formula(power)
    hi = torch.floor(torch.floor(base * stab) * eff)
    lo = torch.floor(torch.floor(torch.floor(base * 85 / 100) * stab) * eff)
    least, most = T['least'][move], T['most'][move]
    lo, hi = lo * torch.where(least > 0, least, 1.0), hi * torch.where(most > 0, most, 1.0)
    fixed = T['fixed'][move]
    lvl = a['level']
    lo = torch.where(fixed == 255, lvl, torch.where(fixed > 0, fixed.float(), lo))  # Seismic Toss, Night Shade, ...
    hi = torch.where(fixed == 255, lvl, torch.where(fixed > 0, fixed.float(), hi))
    maxhp = d['maxhp'].clamp(min=1)
    lo, hi = lo / maxhp, hi / maxhp
    lo, hi = torch.where(T['ohko'][move], 1.0, lo), torch.where(T['ohko'][move], 1.0, hi)
    dead = (move == 0) | (T['cat'][move] == ids.CATEGORY_STATUS) | immune
    return torch.where(dead, 0.0, lo).clamp(0, 3), torch.where(dead, 0.0, hi).clamp(0, 3)


def features(obs: torch.Tensor, T: dict | None = None) -> dict:
    """{'own_moves': [n, 4, 2], 'foe_moves': [n, 4, 2], 'party': [n, 6, 2]}: (least,
    most) fractions for the actives' moves; each own Pokemon's most dealt to the
    foe active and most taken from the foe active's revealed moves. `T` is
    `tables`' dict; a compiled caller passes it (see `Features`)."""
    T = tables(str(obs.device)) if T is None else T
    tok = tokens(obs, T)
    n = obs.shape[0]
    weather = obs[:, L.FIELD_BASE + L.FIELD.index('weather')].long()[:, None, None]
    own = tok['active'][:, :6].float().argmax(1, keepdim=True)  # [n, 1]
    foe = 6 + tok['active'][:, 6:].float().argmax(1, keepdim=True)
    party = torch.arange(6, device=obs.device)[None].expand(n, -1)
    # One pass over 14 attacker/defender pairs: the actives each way, each own Pokemon into the foe active and back.
    att = torch.cat([own, foe, party, foe.expand(-1, 6)], 1)
    dfn = torch.cat([foe, own, foe.expand(-1, 6), party], 1)
    lo, hi = hit(tok, T, att, dfn, tok['moves'].gather(1, att[..., None].expand(-1, -1, 4)), weather)
    return {'own_moves': torch.stack([lo[:, 0], hi[:, 0]], -1),
            'foe_moves': torch.stack([lo[:, 1], hi[:, 1]], -1),
            'party': torch.stack([hi[:, 2:8].amax(-1), hi[:, 8:].amax(-1)], -1)}


class Features(nn.Module):
    """`features` with the tables as buffers: they move with the network and
    stay out of its checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.keys = tuple(tables('cpu'))
        for k, v in tables('cpu').items():
            self.register_buffer(k, v.clone(), persistent=False)

    def table_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.keys}

    def forward(self, obs: torch.Tensor) -> dict:
        return features(obs, self.table_dict())
