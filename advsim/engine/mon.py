"""Writes to one Pokemon that the step and the effect modules share. A leaf,
so `effects/` can import it without importing the step."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine._generated.state import State

# `revealed` bits: what the foe has seen, set where Showdown's public log says it.
REVEAL_SEEN = wp.constant(1)
REVEAL_ABILITY = wp.constant(1 << 5)
REVEAL_ITEM = wp.constant(1 << 6)


@wp.func
def cure_status(s: State, b: int, side: int, slot: int):
    """The status and everything that counts it: the sleep and toxic counter,
    and the sleep bookkeeping, which a clause that still saw would misread."""
    s.status[b, side, slot] = wp.uint8(0)
    s.status_ctr[b, side, slot] = wp.uint8(0)
    s.slept_by_foe[b, side, slot] = wp.uint8(0)
    s.sleep_skipped[b, side, slot] = wp.uint8(0)


@wp.func
def reveal(s: State, b: int, side: int, slot: int, bit: int):
    # An ability shown while transformed is the copy's, and says nothing
    # about the Pokemon's own.
    if bit == REVEAL_ABILITY and slot == int(s.active[b, side]) and \
            (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) != 0:
        return
    s.revealed[b, side, slot] = wp.uint16(int(s.revealed[b, side, slot]) | bit)


@wp.func
def reveal_move(s: State, b: int, side: int, slot: int, move: int):
    """A `move` line shows the move, filed under the Pokemon's own set. A
    transformed Pokemon's moves are the copy's, and a foe cannot tell one it
    shares with its own set apart, so nothing it uses while a copy counts."""
    if (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) != 0 and slot == int(s.active[b, side]):
        return
    for i in range(4):
        m = int(s.moves[b, side, slot, i])
        if m != 0 and m == move:
            reveal(s, b, side, slot, 1 << (1 + i))
