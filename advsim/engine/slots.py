"""Slots: what switching and fainting clear from the active slot, what Baton
Pass keeps, and which move slot an action or an Encore names."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine._generated import ids
from advsim.engine._generated.state import State


@wp.func
def move_slot_of(s: State, b: int, side: int, slot: int, move: int) -> int:
    """Which of the four slots holds this move, or -1 when none does."""
    found = int(-1)
    if move != 0:
        for i in range(4):
            if int(s.moves[b, side, slot, i]) == move:
                found = i
    return found


@wp.func
def encore_slot(s: State, b: int, side: int) -> int:
    return move_slot_of(s, b, side, int(s.active[b, side]), int(s.encore_move[b, side]))


@wp.func
def action_slot(s: State, b: int, side: int, action: int) -> int:
    """Which move slot an action really uses: an Encore can redirect it."""
    if int(s.encore_turns[b, side]) > 0:
        locked = encore_slot(s, b, side)
        if locked >= 0:
            return locked
    return action


@wp.func
def action_move(s: State, b: int, side: int, action: int) -> int:
    """The move an action resolves to, before any of it is spent. Showdown
    settles this in runMove ahead of the BeforeMove event, which is why the
    sleep check already knows which move it is looking at."""
    if action == mask_.ACTION_FORCED:
        # Nothing charged means nothing was selectable: that is Struggle.
        move = int(s.twoturn_move[b, side])
        if move == 0:
            move = ids.MOVE_STRUGGLE
        return move
    return int(s.moves[b, side, int(s.active[b, side]), action_slot(s, b, side, action)])


@wp.func
def revert_transform(s: State, b: int, side: int, slot: int):
    """Put back what Transform copied over. Showdown mutates the Pokemon itself
    and keeps the originals aside, so the engine does the same: everything reads
    the live fields and never has to ask whether a copy is in the way."""
    if (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) == 0:
        return
    s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_TRANSFORMED)
    s.species[b, side, slot] = s.xf_species[b, side]
    s.ability[b, side, slot] = s.xf_ability[b, side]
    s.hp_type[b, side, slot] = s.xf_hp_type[b, side]
    for i in range(5):
        s.stats[b, side, slot, i] = s.xf_stats[b, side, i]
    for i in range(4):
        s.moves[b, side, slot, i] = s.xf_moves[b, side, i]
        s.pp[b, side, slot, i] = s.xf_pp[b, side, i]
        s.max_pp[b, side, slot, i] = s.xf_max_pp[b, side, i]
    # The overlay is zero unless the transformed bit is set: the canonical hash
    # says so, and a copy left behind would be compared against Showdown, which
    # keeps nothing once the Pokemon has reverted.
    s.xf_species[b, side] = wp.uint16(0)
    s.xf_ability[b, side] = wp.uint8(0)
    s.xf_hp_type[b, side] = wp.uint8(0)
    for i in range(5):
        s.xf_stats[b, side, i] = wp.uint16(0)
    for i in range(4):
        s.xf_moves[b, side, i] = wp.uint8(0)
        s.xf_pp[b, side, i] = wp.uint8(0)
        s.xf_max_pp[b, side, i] = wp.uint8(0)


@wp.func
def has_bench(s: State, b: int, side: int) -> bool:
    """Whether anyone but the active is still standing."""
    return (int(s.alive_mask[b, side]) & ~(1 << int(s.active[b, side]))) != 0


@wp.func
def keep_slot(s: State, b: int, side: int):
    """Baton Pass: the boosts and every volatile Showdown does not mark noCopy
    cross over, and the rest go. `last_move` goes with them, since the copy
    starts by clearing the slot and never puts it back."""
    flags = int(s.vflags[b, side])
    s.vflags[b, side] = wp.uint32(flags & ~(mask_.VF_ATTRACT | mask_.VF_FLASHFIRE | mask_.VF_DESTINYBOND))
    s.encore_turns[b, side] = wp.uint8(0)
    s.encore_move[b, side] = wp.uint8(0)
    s.yawn_turns[b, side] = wp.uint8(0)
    s.choice_move[b, side] = wp.uint8(0)
    s.dmg_taken[b, side] = wp.uint16(0)
    s.dmg_cat[b, side] = wp.uint8(0)
    s.last_move[b, side] = wp.uint8(0)


@wp.func
def clear_slot(s: State, b: int, side: int):
    """Wipe the active-slot block. Everything in it belongs to the Pokemon that
    was standing there, so it goes when that Pokemon leaves or faints.

    Mean Look and Spider Web link the two: clearVolatile takes the trap off the
    Pokemon on the other side, so leaving lets its captive go.
    """
    s.vflags[b, 1 - side] = wp.uint32(int(s.vflags[b, 1 - side]) & ~mask_.VF_TRAPPED)
    s.boosts[b, side] = wp.uint32(0)
    s.vflags[b, side] = wp.uint32(0)
    s.sub_hp[b, side] = wp.uint16(0)
    s.confusion_turns[b, side] = wp.uint8(0)
    s.encore_turns[b, side] = wp.uint8(0)
    s.encore_move[b, side] = wp.uint8(0)
    s.choice_move[b, side] = wp.uint8(0)
    s.trap_turns[b, side] = wp.uint8(0)
    s.trap_source[b, side] = wp.uint8(0)
    s.perish_count[b, side] = wp.uint8(0)
    s.yawn_turns[b, side] = wp.uint8(0)
    s.stall_ctr[b, side] = wp.uint8(0)
    s.twoturn_move[b, side] = wp.uint8(0)
    s.last_move[b, side] = wp.uint8(0)
    s.dmg_taken[b, side] = wp.uint16(0)
    s.dmg_cat[b, side] = wp.uint8(0)
    s.turn_flags[b, side] = wp.uint16(0)
