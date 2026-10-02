"""The residual phase's handlers, one per effect, and the tests for whether a
Pokemon carries each. `residual.py` collects, sorts and runs them."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import moves as mv
from advsim.engine import rng as rng_
from advsim.engine import mon
from advsim.engine.effects import ability_fx
from advsim.engine.effects import item_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import status as status_
from advsim.engine import slots as slots_

RESIDUAL_ITEMS = wp.constant(1)    # order 10 sub 4
RESIDUAL_HERB = wp.constant(2)     # order 29


@wp.func
def leech_seed(s: State, dex: Dex, b: int, side: int):
    """Residual order 10 sub 5: an eighth of the seeded Pokemon, to the seeder."""
    if (int(s.vflags[b, side]) & mask_.VF_LEECHSEED) == 0:
        return
    slot = int(s.active[b, side])
    hp = int(s.hp[b, side, slot])
    if hp <= 0:
        return
    foe = 1 - side
    foe_slot = int(s.active[b, foe])
    if int(s.hp[b, foe, foe_slot]) <= 0:
        return
    loss = wp.min(wp.max(int(s.maxhp[b, side, slot]) / 8, 1), hp)
    s.hp[b, side, slot] = wp.uint16(hp - loss)
    if ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_LIQUIDOOZE):
        # What it drains, it loses: Liquid Ooze answers the seeder, not the seed.
        s.hp[b, foe, foe_slot] = wp.uint16(wp.max(int(s.hp[b, foe, foe_slot]) - loss, 0))
        mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
    else:
        foe_max = int(s.maxhp[b, foe, foe_slot])
        s.hp[b, foe, foe_slot] = wp.uint16(wp.min(int(s.hp[b, foe, foe_slot]) + loss, foe_max))


@wp.func
def residual_damage(s: State, b: int, side: int):
    """Burn, poison and toxic, at residual order 10 sub 6."""
    slot = int(s.active[b, side])
    hp = int(s.hp[b, side, slot])
    if hp <= 0:
        return
    status = int(s.status[b, side, slot])
    maxhp = int(s.maxhp[b, side, slot])
    loss = 0
    if status == ids.COND_BRN or status == ids.COND_PSN:
        loss = wp.max(maxhp / 8, 1)
    elif status == ids.COND_TOX:
        # The stage steps up first and stops at fifteen, and the sixteenth is
        # floored once and then multiplied, not the other way round.
        stage = wp.min(int(s.status_ctr[b, side, slot]) + 1, 15)
        s.status_ctr[b, side, slot] = wp.uint8(stage)
        loss = wp.max(maxhp / 16, 1) * stage
    if loss != 0:
        s.hp[b, side, slot] = wp.uint16(wp.max(hp - loss, 0))


@wp.func
def wish_residual(s: State, b: int, side: int):
    """Residual order 7 sub 3. The counter runs down first; at zero the slot's
    current occupant heals half of its own max HP, floored."""
    turns = int(s.wish_turns[b, side])
    if turns == 0:
        return
    s.wish_turns[b, side] = wp.uint8(turns - 1)
    if turns - 1 > 0:
        return
    slot = int(s.active[b, side])
    hp = int(s.hp[b, side, slot])
    maxhp = int(s.maxhp[b, side, slot])
    if hp > 0 and hp < maxhp:
        s.hp[b, side, slot] = wp.uint16(wp.min(hp + wp.max(maxhp / 2, 1), maxhp))


@wp.func
def encore_residual(s: State, b: int, side: int):
    """Residual order 10 sub 14: the counter runs down, and a lock whose move
    has run out of PP ends early."""
    turns = int(s.encore_turns[b, side])
    if turns == 0:
        return
    s.encore_turns[b, side] = wp.uint8(turns - 1)
    if turns - 1 <= 0:
        s.encore_turns[b, side] = wp.uint8(0)
        s.encore_move[b, side] = wp.uint8(0)
        return
    slot = slots_.encore_slot(s, b, side)
    if slot < 0 or int(s.pp[b, side, int(s.active[b, side]), slot]) <= 0:
        s.encore_turns[b, side] = wp.uint8(0)
        s.encore_move[b, side] = wp.uint8(0)


@wp.func
def yawn_residual(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int):
    """Residual order 10 sub 19: the counter runs down, and at zero the sleep it
    promised is tried like any other, Sleep Clause and all."""
    turns = int(s.yawn_turns[b, side])
    if turns == 0:
        return
    s.yawn_turns[b, side] = wp.uint8(turns - 1)
    if turns - 1 > 0:
        return
    slot = int(s.active[b, side])
    if int(s.hp[b, side, slot]) > 0 and status_.status_lands_on(s, dex, b, side, slot, ids.COND_SLP, True):
        status_.inflict_status(s, dex, log, b, side, slot, ids.COND_SLP, True)


@wp.func
def perish_residual(s: State, b: int, side: int):
    """Residual order 12: the count runs down and zero is fatal."""
    count = int(s.perish_count[b, side])
    if count == 0:
        return
    s.perish_count[b, side] = wp.uint8(count - 1)
    if count - 1 <= 0:
        s.hp[b, side, int(s.active[b, side])] = wp.uint16(0)


@wp.func
def truant_residual(s: State, dex: Dex, b: int, side: int):
    """Residual order 27: the loafing turn and the working turn trade places.

    The flag it flips belongs to the Pokemon and outlives the volatile, so a
    Pokemon that traced Truant picks up wherever it left off the last time it
    held it, rather than starting over.
    """
    slot = int(s.active[b, side])
    if not ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_TRUANT):
        return
    if ((int(s.truant_mask[b, side]) >> slot) & 1) != 0:
        s.truant_mask[b, side] = wp.uint8(int(s.truant_mask[b, side]) & ~(1 << slot))
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_TRUANT)
    else:
        s.truant_mask[b, side] = wp.uint8(int(s.truant_mask[b, side]) | (1 << slot))
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | mask_.VF_TRUANT)


@wp.func
def partial_trap(s: State, b: int, side: int):
    """Residual order 10 sub 9: a sixteenth a turn while it lasts.

    It also ends the moment the Pokemon holding it is no longer the one across
    the field: a different party slot, one already down, or one that only came
    in this turn and so has no active turns to its name.
    """
    turns = int(s.trap_turns[b, side])
    if turns == 0:
        return
    foe = 1 - side
    foe_slot = int(s.active[b, foe])
    if int(s.trap_source[b, side]) != foe_slot + 1 or int(s.hp[b, foe, foe_slot]) <= 0 or \
            (int(s.turn_flags[b, foe]) & mask_.TURN_FLAG_SWITCHED_IN) != 0:
        s.trap_turns[b, side] = wp.uint8(0)
        s.trap_source[b, side] = wp.uint8(0)
        return
    s.trap_turns[b, side] = wp.uint8(turns - 1)
    if turns - 1 <= 0:
        s.trap_source[b, side] = wp.uint8(0)
        return
    slot = int(s.active[b, side])
    hp = int(s.hp[b, side, slot])
    if hp > 0:
        s.hp[b, side, slot] = wp.uint16(wp.max(hp - wp.max(int(s.maxhp[b, side, slot]) / 16, 1), 0))


@wp.func
def has_status_residual(s: State, b: int, side: int) -> bool:
    """A handler at residual order 10 sub 6."""
    slot = int(s.active[b, side])
    if int(s.hp[b, side, slot]) <= 0:
        return False
    status = int(s.status[b, side, slot])
    return status == ids.COND_BRN or status == ids.COND_PSN or status == ids.COND_TOX


@wp.func
def has_item_residual(s: State, dex: Dex, b: int, side: int, order: int) -> bool:
    """Whether this side's item has a residual handler at the given order.

    Showdown counts a handler because the callback exists, not because it will
    do anything, so a full-HP Leftovers still takes part in the sort. Lum Berry
    has no residual handler at all, only an Update one, which is why the cure
    items cannot be treated alike here.
    """
    slot = int(s.active[b, side])
    if int(s.hp[b, side, slot]) <= 0:
        return False
    item = int(s.item[b, side, slot])
    if item == 0:
        return False
    family = dex.item_family[item]
    if order == RESIDUAL_ITEMS:
        return family == ids.ITEMFAM_RESIDUAL_HEAL or family == ids.ITEMFAM_PINCH_BERRY
    return family == ids.ITEMFAM_CURE_ITEM and dex.item_p0[item] == item_fx.CURE_BOOSTS


@wp.func
def has_ability_residual(s: State, dex: Dex, b: int, side: int) -> bool:
    """Speed Boost and Shed Skin hold a handler at residual order 10 sub 3."""
    slot = int(s.active[b, side])
    if int(s.hp[b, side, slot]) <= 0:
        return False
    ability = int(s.ability[b, side, slot])
    return ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_RESIDUAL_SELF


@wp.func
def ability_residual(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int):
    """Speed Boost raises Speed once its holder has been out a turn; Shed Skin
    rolls a third every turn it is statused, whether or not it can cure."""
    slot = int(s.active[b, side])
    ability = int(s.ability[b, side, slot])
    if int(s.hp[b, side, slot]) <= 0 or ability == 0:
        return
    if dex.ability_family[ability] != ids.ABILITYFAM_RESIDUAL_SELF:
        return
    if dex.ability_p0[ability] == 0:
        if (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_SWITCHED_IN) == 0:
            boosts = int(s.boosts[b, side])
            s.boosts[b, side] = wp.uint32(mv.set_boost(boosts, 4, mv.get_boost(boosts, 4) + 1))
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)  # `-boost ... [from] ability: Speed Boost`
    elif int(s.status[b, side, slot]) != 0:
        if rng_.random_chance(rng_.draw(s, log, b), 33, 100):
            mon.cure_status(s, b, side, slot)
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
