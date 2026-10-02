"""Sampling from the set table: which set a Pokemon has, given what it
revealed; which species an unseen one is; an HP behind a displayed percent.
`determinize.py` puts these together into a world."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import mon
from advsim.engine import obs as obs_
from advsim.engine import rng as rng_
from advsim.engine._generated import ids
from advsim.engine._generated.state import State

# Set table columns: build/build_pool.py's pool row.


@wp.struct
class SetDist:
    """artifacts/setdist.npz: distinct sets grouped by species, with counts."""
    sets: wp.array2d(dtype=wp.int16)
    weight: wp.array(dtype=wp.int32)
    start: wp.array(dtype=wp.int32)
    count: wp.array(dtype=wp.int32)
    gender: wp.array2d(dtype=wp.int32)
    species_weight: wp.array(dtype=wp.int32)


@wp.func
def transformed(s: State, b: int, side: int, slot: int) -> bool:
    return slot == int(s.active[b, side]) and (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) != 0


@wp.func
def own_move(s: State, b: int, side: int, slot: int, k: int) -> int:
    if transformed(s, b, side, slot):
        return int(s.xf_moves[b, side, k])
    return int(s.moves[b, side, slot, k])


@wp.func
def own_pp(s: State, b: int, side: int, slot: int, k: int) -> int:
    if transformed(s, b, side, slot):
        return int(s.xf_pp[b, side, k])
    return int(s.pp[b, side, slot, k])


@wp.func
def own_ability(s: State, b: int, side: int, slot: int) -> int:
    if slot == int(s.active[b, side]) and int(s.base_ability[b, side]) > 0:
        return int(s.base_ability[b, side]) - 1
    if transformed(s, b, side, slot):
        return int(s.xf_ability[b, side])
    return int(s.ability[b, side, slot])


@wp.func
def set_fits(s: State, sd: SetDist, b: int, side: int, slot: int, row: int, level: int) -> bool:
    """What the Pokemon revealed is in this set: at level 2 its moves and
    ability, at 1 its moves, at 0 anything goes."""
    if level == 0:
        return True
    revealed = int(s.revealed[b, side, slot])
    if level == 2 and (revealed & mon.REVEAL_ABILITY) != 0 and \
            int(sd.sets[row, ids.POOL_ABILITY]) != own_ability(s, b, side, slot):
        return False
    for k in range(4):
        if (revealed & (1 << (1 + k))) != 0:
            m = own_move(s, b, side, slot, k)
            found = bool(False)
            for i in range(4):
                if int(sd.sets[row, ids.POOL_MOVE1 + i]) == m:
                    found = True
            if not found:
                return False
    return True


@wp.func
def pick_set(s: State, sd: SetDist, b: int, side: int, slot: int, sp: int, constrained: bool,
             raw: wp.uint32) -> int:
    """A set of species `sp`, weighted by its count among the ones that fit.
    If none does (a set the table never saw), the ability is let go first,
    then the moves."""
    lo = int(sd.start[sp])
    hi = lo + int(sd.count[sp])
    level = int(3)
    if not constrained:
        level = 1
    total = int(0)
    for _ in range(3):
        if total == 0 and level > 0:
            level = level - 1
            for r in range(lo, hi):
                if set_fits(s, sd, b, side, slot, r, level):
                    total += int(sd.weight[r])
    pick = rng_.random_n(raw, total)
    for r in range(lo, hi):
        if set_fits(s, sd, b, side, slot, r, level):
            pick -= int(sd.weight[r])
            if pick < 0:
                return r
    return hi - 1


@wp.func
def hp_like(hp: int, maxhp: int, new_max: int, raw: wp.uint32) -> int:
    """An HP out of `new_max` that shows the same percent as hp / maxhp,
    uniformly; the nearest one when none does."""
    pct = obs_.hp_percent(hp, maxhp)
    if pct == 0 or pct == 100:
        return wp.where(pct == 0, 0, new_max)
    n = int(0)
    for h in range(1, new_max):
        if obs_.hp_percent(h, new_max) == pct:
            n += 1
    if n == 0:
        return wp.clamp(pct * new_max / 100, 1, new_max - 1)
    k = rng_.random_n(raw, n)
    for h in range(1, new_max):
        if obs_.hp_percent(h, new_max) == pct:
            if k == 0:
                return h
            k -= 1
    return new_max - 1


@wp.func
def pick_species(s: State, sd: SetDist, w: int, side: int, j: int, raw: wp.uint32) -> int:
    """A species for an unseen Pokemon, weighted by the pool, not already one
    of slots 0..j-1 of this side in world w (Species Clause)."""
    total = int(0)
    for sp in range(1, ids.N_SPECIES):
        taken = bool(False)
        for i in range(j):
            if obs_.shown_species(s, w, side, i) == sp:
                taken = True
        if not taken:
            total += int(sd.species_weight[sp])
    pick = rng_.random_n(raw, total)
    for sp in range(1, ids.N_SPECIES):
        taken = bool(False)
        for i in range(j):
            if obs_.shown_species(s, w, side, i) == sp:
                taken = True
        if not taken:
            pick -= int(sd.species_weight[sp])
            if pick < 0:
                return sp
    return ids.N_SPECIES - 1


@wp.func
def pick_gender(sd: SetDist, sp: int, raw: wp.uint32) -> int:
    total = int(0)
    for g in range(4):
        total += int(sd.gender[sp, g])
    pick = rng_.random_n(raw, total)
    for g in range(4):
        pick -= int(sd.gender[sp, g])
        if pick < 0:
            return g
    return 0
