"""The engine's codes: actions, requests, results, and the bits of `vflags`,
`turn_flags` and `err`. Legal actions are a u16 bitmask per side.

Action codes follow the spec: 0-3 move slots, 4-9 switch to a party index,
10 pass, 11 the forced action the engine resolves from state (Struggle, a
recharge turn, or the second turn of Solar Beam).

An illegal action is defined rather than undefined: the step substitutes the
first legal action and sets a bit in `err`, so a bad policy is visible in a
counter instead of corrupting a battle.
"""
import warp as wp

MOVE_BASE = wp.constant(0)

SWITCH_BASE = wp.constant(4)

ACTION_PASS = wp.constant(10)
ACTION_FORCED = wp.constant(11)

N_ACTIONS = wp.constant(12)

REQUEST_NONE = wp.constant(0)
REQUEST_MOVE = wp.constant(1)
REQUEST_SWITCH = wp.constant(2)
REQUEST_WAIT = wp.constant(3)

# Every vflags bit; export_state.js keeps the same numbering.
VF_MUSTRECHARGE = wp.constant(1 << 0)
VF_TWOTURN = wp.constant(1 << 1)
VF_TRAPPED = wp.constant(1 << 2)
VF_FLASHFIRE = wp.constant(1 << 3)
VF_ATTRACT = wp.constant(1 << 4)
VF_PROTECT = wp.constant(1 << 5)
VF_ENDURE = wp.constant(1 << 6)
VF_LEECHSEED = wp.constant(1 << 7)
# bit 8 is free: whether a side has acted is turn scratch, not a volatile, so
# it lives in turn_flags where it survives a mid-turn switch-in.
VF_FLINCH = wp.constant(1 << 9)
VF_DESTINYBOND = wp.constant(1 << 10)
VF_TRANSFORMED = wp.constant(1 << 11)
VF_TRUANT = wp.constant(1 << 12)

# `result`
RESULT_ONGOING = wp.constant(0)
RESULT_P1 = wp.constant(1)
RESULT_P2 = wp.constant(2)
RESULT_TIE = wp.constant(3)

# `err` bits
ERR_ILLEGAL = wp.constant(1 << 0)
ERR_LOG_EXHAUSTED = wp.constant(1 << 3)

# `turn_flags` bits: what a side has already done this turn
TURN_FLAG_SWITCHED_IN = wp.constant(1 << 0)
TURN_FLAG_REFLECT = wp.constant(1 << 2)  # Counter or Mirror Coat was chosen this turn
TURN_FLAG_CHARGE_SET = wp.constant(1 << 3)  # a charge move started charging this turn
TURN_FLAG_FOCUS = wp.constant(1 << 4)  # Focus Punch was chosen this turn
TURN_FLAG_PASSING = wp.constant(1 << 5)  # a Baton Pass is waiting for its recipient
TURN_FLAG_DRAGGED = wp.constant(1 << 6)  # Roar or Whirlwind is about to pull this side out
TURN_FLAG_PURSUIT = wp.constant(1 << 7)  # Pursuit was chosen this turn
TURN_FLAG_CHASING = wp.constant(1 << 8)  # and is landing on a Pokemon on its way out
TURN_FLAG_RECHARGE_SET = wp.constant(1 << 9)  # a recharge was left behind this turn
TURN_FLAG_MOVED = wp.constant(1 << 10)  # this side has already acted this turn
TURN_FLAG_FAINT_FIRST = wp.constant(1 << 11)  # this side's faint went into the queue first
TURN_FLAG_STALLED = wp.constant(1 << 12)  # Protect or Endure landed, so the stall volatile starts over


@wp.func
def is_switch(action: int) -> bool:
    """Codes 4 to 9 are switches; 10 is a pass and 11 the forced action."""
    return action >= SWITCH_BASE and action < SWITCH_BASE + 6


@wp.func
def move_is_legal(pp: int, move_id: int, encore_move: int, choice_move: int) -> bool:
    """PP, and the two locks that can pin a Pokemon to one move."""
    if move_id == 0 or pp <= 0:
        return False
    if encore_move != 0 and move_id != encore_move:
        return False
    if choice_move != 0 and move_id != choice_move:
        return False
    return True


@wp.func
def can_switch_out(vflags: int, trap_turns: int, foe_traps: bool) -> bool:
    """Mean Look and Spider Web set the volatile; Wrap sets a counter; abilities trap from the other side."""
    if (vflags & VF_TRAPPED) != 0 or trap_turns > 0 or foe_traps:
        return False
    return True


@wp.func
def switch_mask(alive_mask: int, active: int) -> int:
    """One bit per living party member that is not the active one."""
    mask = 0
    for slot in range(6):
        if (alive_mask & (1 << slot)) != 0 and slot != active:
            mask |= 1 << (SWITCH_BASE + slot)
    return mask


@wp.func
def forced_mask(vflags: int) -> int:
    """Recharge and charge turns leave exactly one legal action."""
    if (vflags & VF_MUSTRECHARGE) != 0 or (vflags & VF_TWOTURN) != 0:
        return 1 << ACTION_FORCED
    return 0
