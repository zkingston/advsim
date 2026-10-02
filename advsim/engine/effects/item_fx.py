"""Item families.

Thirteen items appear in this format and they fall into six shapes. The stat
and crit items only matter inside the damage calculation; the rest act in the
residual phase or at an Update event, which is where Showdown lets a Lum Berry
cure the status that just landed.
"""
import warp as wp

from advsim.engine import damage as dmg
from advsim.engine import moves as mv
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import mon

CURE_STATUS = wp.constant(-5)  # SENTINELS in build/tables.py
CURE_BOOSTS = wp.constant(-6)


@wp.func
def species_matches(species_num: int, low: int, high: int) -> bool:
    """A dex-number range, or 0 for any species."""
    if low == 0:
        return True
    return species_num >= low and species_num <= high


@wp.func
def stat_modifier(dex: Dex, item: int, species_num: int, stat: int, move_type: int) -> int:
    """The 4096-scaled modifier this item contributes to one stat."""
    if item == 0:
        return dmg.NEUTRAL
    family = dex.item_family[item]
    mask = dex.item_p0[item]
    if ((mask >> stat) & 1) == 0:
        return dmg.NEUTRAL
    if family == ids.ITEMFAM_STAT_ITEM:
        if species_matches(species_num, dex.item_p2[item], dex.item_p3[item]):
            return dex.item_p1[item]
    elif family == ids.ITEMFAM_TYPE_ITEM:
        if move_type == dex.item_p2[item]:
            return dex.item_p1[item]
    return dmg.NEUTRAL


@wp.func
def crit_stages(dex: Dex, item: int, species_num: int) -> int:
    """Stick raises Farfetch'd's crit ratio; nothing else here touches it."""
    if item != 0 and dex.item_family[item] == ids.ITEMFAM_CRIT_ITEM:
        if species_matches(species_num, dex.item_p1[item], dex.item_p2[item]):
            return dex.item_p0[item]
    return 0


@wp.func
def residual_heal(s: State, dex: Dex, b: int, side: int, slot: int) -> bool:
    """Leftovers, at residual order 10 sub 4."""
    item = int(s.item[b, side, slot])
    if item == 0 or dex.item_family[item] != ids.ITEMFAM_RESIDUAL_HEAL:
        return False
    maxhp = int(s.maxhp[b, side, slot])
    hp = int(s.hp[b, side, slot])
    if hp <= 0 or hp >= maxhp:
        return False
    s.hp[b, side, slot] = wp.uint16(wp.min(hp + wp.max(maxhp / dex.item_p0[item], 1), maxhp))
    mon.reveal(s, b, side, slot, mon.REVEAL_ITEM)  # `-heal ... [from] item: Leftovers`
    return True


@wp.func
def pinch_berry(s: State, dex: Dex, b: int, side: int, slot: int) -> bool:
    """Salac, Liechi and Petaya: eaten at a quarter HP for one stage of a stat."""
    item = int(s.item[b, side, slot])
    if item == 0 or dex.item_family[item] != ids.ITEMFAM_PINCH_BERRY:
        return False
    hp = int(s.hp[b, side, slot])
    if hp <= 0 or hp > int(s.maxhp[b, side, slot]) / dex.item_p1[item]:
        return False
    mask = dex.item_p0[item]
    boosts = int(s.boosts[b, side])
    for stat in range(5):
        if ((mask >> stat) & 1) != 0:
            boosts = mv.set_boost(boosts, stat, mv.get_boost(boosts, stat) + 1)
    s.boosts[b, side] = wp.uint32(boosts)
    mon.reveal(s, b, side, slot, mon.REVEAL_ITEM)  # `-enditem`
    s.item[b, side, slot] = wp.uint8(0)
    return True


@wp.func
def white_herb(s: State, dex: Dex, b: int, side: int, slot: int) -> bool:
    """Clears the holder's negative boosts, then is used up. `useItem` refuses a
    Pokemon at 0 HP, so one knocked out after its drop keeps the herb."""
    item = int(s.item[b, side, slot])
    if item == 0 or dex.item_family[item] != ids.ITEMFAM_CURE_ITEM or dex.item_p0[item] != CURE_BOOSTS:
        return False
    if int(s.hp[b, side, slot]) <= 0:
        return False
    boosts = int(s.boosts[b, side])
    cleared = boosts
    for stat in range(7):
        if mv.get_boost(boosts, stat) < 0:
            cleared = mv.set_boost(cleared, stat, 0)
    if cleared == boosts:
        return False
    s.boosts[b, side] = wp.uint32(cleared)
    mon.reveal(s, b, side, slot, mon.REVEAL_ITEM)  # `-enditem`
    s.item[b, side, slot] = wp.uint8(0)
    return True


@wp.func
def lum_berry(s: State, dex: Dex, b: int, side: int, slot: int, confused: bool) -> bool:
    """Cures the status or the confusion that just landed, and takes the other
    with it: one berry answers both.

    A gen 3 berry is eaten at AfterSetStatus, the moment the status lands, and
    the Update event is only the catch-all behind it. The difference shows when
    the Pokemon goes down in between: the berry it ate is gone even though the
    status never outlived it.
    """
    item = int(s.item[b, side, slot])
    if item == 0 or dex.item_family[item] != ids.ITEMFAM_CURE_ITEM or dex.item_p0[item] != CURE_STATUS:
        return False
    if int(s.hp[b, side, slot]) <= 0:
        return False
    if int(s.status[b, side, slot]) == 0 and not confused:
        return False
    mon.cure_status(s, b, side, slot)
    mon.reveal(s, b, side, slot, mon.REVEAL_ITEM)  # `-enditem`
    s.item[b, side, slot] = wp.uint8(0)
    return True
