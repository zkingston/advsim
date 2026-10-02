"""Legal actions, as Showdown's server would accept them, and the guard that
turns an illegal one into the lowest legal one."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine.effects import ability_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import switch as switch_


@wp.func
def partial_trap_holds(s: State, b: int, side: int) -> bool:
    """Wrap's onTrapPokemon asks whether its source is still active, so the
    volatile stops holding the moment its owner leaves or goes down, without
    waiting for the residual phase to take it away."""
    if int(s.trap_turns[b, side]) == 0:
        return False
    foe = 1 - side
    return int(s.trap_source[b, side]) == int(s.active[b, foe]) + 1


@wp.func
def trapped_by_ability(s: State, dex: Dex, b: int, side: int) -> bool:
    """Whether an ability is holding this side's active in.

    Shadow Tag holds anything, Arena Trap anything on the ground, and Magnet
    Pull anything made of Steel. Magnet Pull is an Any handler, so it is asked
    about its own holder as well, but `isAdjacent` answers no for a Pokemon
    and itself, so a Steel type with Magnet Pull is free to leave.
    """
    slot = int(s.active[b, side])
    foe = 1 - side
    steel = int(s.types[b, side, 0]) == ids.TYPE_STEEL or int(s.types[b, side, 1]) == ids.TYPE_STEEL
    theirs = int(s.ability[b, foe, int(s.active[b, foe])])
    if int(s.hp[b, foe, int(s.active[b, foe])]) <= 0:
        return False
    return ability_fx.traps_foe(dex, theirs, switch_.grounded(s, dex, b, side, slot), steel)


@wp.func
def legal_actions(s: State, dex: Dex, b: int, side: int) -> int:
    """Legal actions for one side, as a bitmask over the 12 action codes.

    Lives here rather than in mask.py because trapping reads the field, and
    mask.py may not import this module: turn.py already imports it.
    """
    request = int(s.request[b, side])
    if request == mask_.REQUEST_NONE or request == mask_.REQUEST_WAIT or int(s.result[b]) != 0:
        return 1 << mask_.ACTION_PASS

    active = int(s.active[b, side])
    switches = mask_.switch_mask(int(s.alive_mask[b, side]), active)
    if request == mask_.REQUEST_SWITCH:
        # A replacement is forced: no move, and no trap can stop it.
        if switches == 0:
            return 1 << mask_.ACTION_PASS
        return switches

    vflags = int(s.vflags[b, side])
    forced = mask_.forced_mask(vflags)
    if forced != 0:
        return forced

    moves = 0
    for slot in range(4):
        if mask_.move_is_legal(int(s.pp[b, side, active, slot]), int(s.moves[b, side, active, slot]),
                               int(s.encore_move[b, side]), int(s.choice_move[b, side])):
            moves |= 1 << (mask_.MOVE_BASE + slot)
    if moves == 0:
        moves = 1 << mask_.ACTION_FORCED  # Struggle

    held = 0
    if partial_trap_holds(s, b, side):
        held = 1
    if not mask_.can_switch_out(vflags, held, trapped_by_ability(s, dex, b, side)):
        switches = 0
    return moves | switches


@wp.func
def legalize(s: State, dex: Dex, b: int, side: int, action: int) -> int:
    """An illegal action is defined, not undefined: take the lowest legal one
    and set a bit in `err`, so a bad policy shows up in a counter instead of
    running a move slot that holds no move."""
    legal = legal_actions(s, dex, b, side)
    if action >= 0 and action < mask_.N_ACTIONS and (legal & (1 << action)) != 0:
        return action
    s.err[b] = wp.uint8(int(s.err[b]) | mask_.ERR_ILLEGAL)
    for code in range(mask_.N_ACTIONS):
        if (legal & (1 << code)) != 0:
            return code
    return mask_.ACTION_PASS
