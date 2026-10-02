"""Determinization: one full battle consistent with what `side` can see.

Everything `layout.py` marks hidden from `side` is overwritten with a sample,
and nothing the sample reads is hidden, so two true states that differ only in
hidden fields give identical worlds under one key (the leak test):

- The foe's party goes in a canonical order, since the real one is never
  shown: seen Pokemon by species id, then the unseen ones.
- An unseen foe Pokemon is a fresh species (Species Clause) and set from the
  set table. A seen one keeps what it revealed and takes the rest of a set that
  contains it: moves in id order, stats, max HP, Hidden Power type, the ability
  and item if unrevealed, and an exact HP that shows the same percent.
- Hidden durations on both sides (a foe-caused sleep, confusion, Encore, a
  partial trap) are drawn again over their whole range.

Every draw is a pure function of (key, world slot, what it is for), so no
counter is threaded through. Positive evidence only: a candidate set must
contain what was revealed, and nothing is ruled out by what failed to appear.
"""
import warp as wp

from advsim.engine import mon
from advsim.engine import obs as obs_
from advsim.engine import order as order_
from advsim.engine import rng as rng_
from advsim.engine._generated import ids
from advsim.engine._generated.state import State, copy_battle, copy_mon
from advsim.engine.setdist import (SetDist, hp_like, own_ability, own_move, own_pp, pick_gender, pick_set,
                                   pick_species, transformed)
from advsim.engine.dex import Dex

# What a draw is for, the low bits of its counter.
FOR_SPECIES = wp.constant(0)
FOR_SET = wp.constant(1)
FOR_GENDER = wp.constant(2)
FOR_HP = wp.constant(3)
FOR_DURATION = wp.constant(4)
MOVES_REVEALED = wp.constant(0x1E)


@wp.func
def draw(key: wp.uint32, w: int, what: int, j: int) -> wp.uint32:
    return rng_.rng_u32(rng_.mix32(key ^ wp.uint32(w) * rng_.KNUTH), wp.uint32(j * 8 + what))


@wp.func
def canonical_slot(s: State, b: int, side: int, slot: int) -> int:
    """Where a foe Pokemon goes: seen ones by species id, then unseen ones in
    their current order, which is harmless since they are all drawn afresh."""
    seen = (int(s.revealed[b, side, slot]) & mon.REVEAL_SEEN) != 0
    sp = obs_.shown_species(s, b, side, slot)
    rank = int(0)
    n_seen = int(0)
    for i in range(6):
        other = (int(s.revealed[b, side, i]) & mon.REVEAL_SEEN) != 0
        if other:
            n_seen += 1
        if seen and other and obs_.shown_species(s, b, side, i) < sp:
            rank += 1
        if not seen and not other and i < slot:
            rank += 1
    if seen:
        return rank
    return n_seen + rank


@wp.func
def remap_mask(s: State, b: int, side: int, bits: int) -> int:
    out = int(0)
    for i in range(6):
        if (bits & (1 << i)) != 0:
            out = out | (1 << canonical_slot(s, b, side, i))
    return out


@wp.func
def write_unseen(s: State, sd: SetDist, w: int, side: int, j: int, key: wp.uint32):
    sp = pick_species(s, sd, w, side, j, draw(key, w, FOR_SPECIES, j))
    r = pick_set(s, sd, w, side, j, sp, False, draw(key, w, FOR_SET, j))
    s.species[w, side, j] = wp.uint16(sp)
    s.level[w, side, j] = wp.uint8(int(sd.sets[r, ids.POOL_LEVEL]))
    s.gender[w, side, j] = wp.uint8(pick_gender(sd, sp, draw(key, w, FOR_GENDER, j)))
    s.ability[w, side, j] = wp.uint8(int(sd.sets[r, ids.POOL_ABILITY]))
    s.item[w, side, j] = wp.uint8(int(sd.sets[r, ids.POOL_ITEM]))
    s.hp_type[w, side, j] = wp.uint8(int(sd.sets[r, ids.POOL_HP_TYPE]))
    s.maxhp[w, side, j] = wp.uint16(int(sd.sets[r, ids.POOL_MAXHP]))
    s.hp[w, side, j] = wp.uint16(int(sd.sets[r, ids.POOL_MAXHP]))
    for i in range(5):
        s.stats[w, side, j, i] = wp.uint16(int(sd.sets[r, ids.POOL_ATK + i]))
    s.cached_spe[w, side, j] = wp.uint16(int(sd.sets[r, ids.POOL_ATK + 4]))
    for k in range(4):
        s.moves[w, side, j, k] = wp.uint8(int(sd.sets[r, ids.POOL_MOVE1 + k]))
        s.pp[w, side, j, k] = wp.uint8(int(sd.sets[r, ids.POOL_PP1 + k]))
        s.max_pp[w, side, j, k] = wp.uint8(int(sd.sets[r, ids.POOL_PP1 + k]))
    s.status[w, side, j] = wp.uint8(0)
    s.status_ctr[w, side, j] = wp.uint8(0)
    s.sleep_skipped[w, side, j] = wp.uint8(0)
    s.slept_by_foe[w, side, j] = wp.uint8(0)
    s.revealed[w, side, j] = wp.uint16(0)


@wp.func
def write_seen(s: State, sd: SetDist, b: int, w: int, side: int, slot: int, j: int, key: wp.uint32):
    """Pokemon `slot` of battle b, now at j in world w: the rest of a set that
    fits what it revealed. A transformed one keeps the copy in its live fields
    and takes the set where its own is kept."""
    revealed = int(s.revealed[b, side, slot])
    sp = obs_.shown_species(s, b, side, slot)
    r = pick_set(s, sd, b, side, slot, sp, True, draw(key, w, FOR_SET, j))
    xf = transformed(s, b, side, slot)
    new_max = int(sd.sets[r, ids.POOL_MAXHP])
    s.maxhp[w, side, j] = wp.uint16(new_max)
    s.hp[w, side, j] = wp.uint16(hp_like(int(s.hp[b, side, slot]), int(s.maxhp[b, side, slot]), new_max,
                                         draw(key, w, FOR_HP, j)))
    if (revealed & mon.REVEAL_ITEM) == 0:
        s.item[w, side, j] = wp.uint8(int(sd.sets[r, ids.POOL_ITEM]))
    bits = revealed & ~MOVES_REVEALED
    for k in range(4):
        m = int(sd.sets[r, ids.POOL_MOVE1 + k])
        pp = int(sd.sets[r, ids.POOL_PP1 + k])
        for i in range(4):
            if (revealed & (1 << (1 + i))) != 0 and own_move(s, b, side, slot, i) == m:
                pp = own_pp(s, b, side, slot, i)
                bits = bits | (1 << (1 + k))
        if xf:
            s.xf_moves[w, side, k] = wp.uint8(m)
            s.xf_pp[w, side, k] = wp.uint8(pp)
            s.xf_max_pp[w, side, k] = wp.uint8(int(sd.sets[r, ids.POOL_PP1 + k]))
        else:
            s.moves[w, side, j, k] = wp.uint8(m)
            s.pp[w, side, j, k] = wp.uint8(pp)
            s.max_pp[w, side, j, k] = wp.uint8(int(sd.sets[r, ids.POOL_PP1 + k]))
    s.revealed[w, side, j] = wp.uint16(bits)
    ability = int(sd.sets[r, ids.POOL_ABILITY])
    if (revealed & mon.REVEAL_ABILITY) != 0:
        ability = own_ability(s, b, side, slot)
    if xf:
        s.xf_ability[w, side] = wp.uint8(ability)
        s.xf_hp_type[w, side] = wp.uint8(int(sd.sets[r, ids.POOL_HP_TYPE]))
        for i in range(5):
            s.xf_stats[w, side, i] = wp.uint16(int(sd.sets[r, ids.POOL_ATK + i]))
    else:
        if int(s.base_ability[b, side]) == 0 or slot != int(s.active[b, side]):
            s.ability[w, side, j] = wp.uint8(ability)
        s.hp_type[w, side, j] = wp.uint8(int(sd.sets[r, ids.POOL_HP_TYPE]))
        for i in range(5):
            s.stats[w, side, j, i] = wp.uint16(int(sd.sets[r, ids.POOL_ATK + i]))
    s.cached_spe[w, side, j] = wp.uint16(int(sd.sets[r, ids.POOL_ATK + 4]))


@wp.func
def redraw(raw: wp.uint32, value: int, most: int) -> int:
    """A running duration drawn again over 1..most, or 0 if none is running."""
    if value == 0:
        return 0
    return 1 + rng_.random_n(raw, most)


@wp.func
def determinize(s: State, dex: Dex, sd: SetDist, b: int, w: int, side: int, key: wp.uint32):
    """World slot w: battle b as `side` sees it, the hidden rest drawn with `key`."""
    foe = 1 - side
    copy_battle(s, b, w)
    for i in range(6):
        copy_mon(s, b, foe, i, w, canonical_slot(s, b, foe, i))
    active = int(s.active[b, foe])
    s.active[w, foe] = wp.uint8(canonical_slot(s, b, foe, active))
    s.alive_mask[w, foe] = wp.uint8(remap_mask(s, b, foe, int(s.alive_mask[b, foe])))
    s.knocked_mask[w, foe] = wp.uint8(remap_mask(s, b, foe, int(s.knocked_mask[b, foe])))
    s.truant_mask[w, foe] = wp.uint8(remap_mask(s, b, foe, int(s.truant_mask[b, foe])))
    source = int(s.trap_source[b, side])
    if source > 0:
        s.trap_source[w, side] = wp.uint8(canonical_slot(s, b, foe, source - 1) + 1)

    for i in range(6):
        j = canonical_slot(s, b, foe, i)
        if (int(s.revealed[b, foe, i]) & mon.REVEAL_SEEN) != 0:
            write_seen(s, sd, b, w, foe, i, j, key)
    for j in range(6):
        if (int(s.revealed[w, foe, j]) & mon.REVEAL_SEEN) == 0:
            write_unseen(s, sd, w, foe, j, key)
            s.alive_mask[w, foe] = wp.uint8(int(s.alive_mask[w, foe]) | (1 << j))
            s.knocked_mask[w, foe] = wp.uint8(int(s.knocked_mask[w, foe]) & ~(1 << j))
            s.truant_mask[w, foe] = wp.uint8(int(s.truant_mask[w, foe]) & ~(1 << j))

    # The active is first in Showdown's party array; the rest in canonical order.
    fa = int(s.active[w, foe])
    for j in range(6):
        s.party_pos[w, foe, j] = wp.uint8(wp.where(j == fa, 0, wp.where(j < fa, j + 1, j)))

    # What the foe active's hidden set decides: a Substitute's size, the damage
    # it has taken this turn, a Choice lock.
    if int(s.sub_hp[w, foe]) > 0:
        s.sub_hp[w, foe] = wp.uint16(wp.max(int(s.maxhp[w, foe, fa]) / 4, 1))
    s.dmg_taken[w, foe] = wp.uint16(0)
    if (int(s.revealed[w, foe, fa]) & mon.REVEAL_ITEM) == 0:
        lock = int(0)
        if int(s.item[w, foe, fa]) == ids.ITEM_CHOICEBAND:
            lock = int(s.last_move[w, foe])
        s.choice_move[w, foe] = wp.uint8(lock)
    s.cached_spe[w, foe, fa] = wp.uint16(order_.speed_of(s, dex, w, foe))

    for p in range(2):
        d = 16 + p * 16
        s.confusion_turns[w, p] = wp.uint8(redraw(draw(key, w, FOR_DURATION, d), int(s.confusion_turns[w, p]), 5))
        s.encore_turns[w, p] = wp.uint8(redraw(draw(key, w, FOR_DURATION, d + 1), int(s.encore_turns[w, p]), 6))
        s.trap_turns[w, p] = wp.uint8(redraw(draw(key, w, FOR_DURATION, d + 2), int(s.trap_turns[w, p]), 6))
        for j in range(6):
            if int(s.status[w, p, j]) == ids.COND_SLP and int(s.slept_by_foe[w, p, j]) != 0:
                s.status_ctr[w, p, j] = wp.uint8(redraw(draw(key, w, FOR_DURATION, d + 3 + j),
                                                        int(s.status_ctr[w, p, j]), 5))

    s.rng_key[w] = rng_.mix32(key ^ rng_.mix32(wp.uint32(w) ^ rng_.GOLDEN))
    s.rng_ctr[w] = wp.uint32(0)
    s.rng_mode[w] = wp.uint8(0)
