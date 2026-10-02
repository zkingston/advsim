"""The L3 baseline policies (`showdown/lib/play.js`) in the engine, for the
Elo bracket: random, emerald, maxdamage, status, switchaverse.

Each is a move chooser, whether it switches of its own accord (one decision in
three), and a replacement chooser. Random choices use the bracket's own
stream, never a battle's. maxdamage is deterministic and must pick what
play.js picks (tests/test_policy_heuristic.py); its estimate reads the dex the
way play.js does, so Return and Hidden Power, whose gen 3 `basePower` is 0,
estimate at 0.
"""
import warp as wp

from advsim.engine import legal as legal_
from advsim.engine import mask as mask_
from advsim.engine import moves as mv
from advsim.engine import policy_emerald
from advsim.engine import rng as rng_
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex

RANDOM = wp.constant(0)
EMERALD = wp.constant(1)
MAXDAMAGE = wp.constant(2)
STATUS = wp.constant(3)
SWITCHAVERSE = wp.constant(4)


@wp.func
def pick_bit(bits: int, raw: wp.uint32) -> int:
    """A uniformly random set bit, as its index."""
    n = int(0)
    for a in range(12):
        n += (bits >> a) & 1
    k = rng_.random_n(raw, n)
    for a in range(12):
        if (bits >> a) & 1 == 1:
            if k == 0:
                return a
            k -= 1
    return mask_.ACTION_PASS


@wp.func
def estimate(s: State, dex: Dex, b: int, side: int, move: int) -> wp.vec2l:
    """play.js's estimate as an exact fraction (numerator, denominator):
    power x STAB x 2^effectiveness x attack / defence, scaled by 8 so STAB's
    1.5 and a quarter effectiveness stay whole. 0 for status moves and
    immunities."""
    if move == 0 or dex.move_category[move] == ids.CATEGORY_STATUS:
        return wp.vec2l(wp.int64(0), wp.int64(1))
    att = int(s.active[b, side])
    foe = 1 - side
    dfn = int(s.active[b, foe])
    move_type = dex.move_type[move]
    if dex.move_type_from_mon[move] != 0:
        move_type = int(s.hp_type[b, side, att])
    t1 = int(s.types[b, foe, 0])
    t2 = int(s.types[b, foe, 1])
    eff = dex.type_chart[move_type, t1]
    if t2 != t1:
        eff += dex.type_chart[move_type, t2]
    if dex.type_chart[move_type, t1] < -1 or (t2 != t1 and dex.type_chart[move_type, t2] < -1):
        return wp.vec2l(wp.int64(0), wp.int64(1))
    power = dex.move_power[move]
    if dex.move_type_from_mon[move] != 0 or move == ids.MOVE_RETURN:
        power = 0
    fixed = dex.move_fixed_damage[move]
    if power == 0 and fixed == mv.FIXED_LEVEL:
        power = int(s.level[b, side, att])
    elif power == 0 and fixed > 0:
        power = fixed
    stat = int(0)  # stats are atk, def, spa, spd, spe; the type decides the split in gen 3
    if dex.type_is_special[move_type] != 0:
        stat = 2
    stab = int(2)
    if int(s.types[b, side, 0]) == move_type or int(s.types[b, side, 1]) == move_type:
        stab = 3
    num = wp.int64(power * stab) * wp.int64(int(s.stats[b, side, att, stat]))
    den = wp.int64(int(s.stats[b, foe, dfn, stat + 1]))
    if eff >= 0:
        num = num * wp.int64(1 << (eff + 2))
    else:
        num = num * wp.int64(4 >> (-eff))
    return wp.vec2l(num, den)


@wp.func
def maxdamage_move(s: State, dex: Dex, b: int, side: int, moves: int) -> int:
    """The legal move that hits hardest, the first on a tie; moves 0-3 in slot
    order, then the forced action, which estimates at 0."""
    att = int(s.active[b, side])
    best = int(-1)
    best_num = wp.int64(-1)
    best_den = wp.int64(1)
    for a in range(12):
        if (moves >> a) & 1 == 1:
            e = wp.vec2l(wp.int64(0), wp.int64(1))
            if a < 4:
                e = estimate(s, dex, b, side, int(s.moves[b, side, att, a]))
            if best < 0 or e[0] * best_den > best_num * e[1]:
                best = a
                best_num = e[0]
                best_den = e[1]
    return best


@wp.func
def status_move(s: State, dex: Dex, b: int, side: int, moves: int, raw: wp.uint32) -> int:
    """A status move when one is legal, otherwise any move."""
    att = int(s.active[b, side])
    status = int(0)
    for a in range(4):
        if (moves >> a) & 1 == 1 and dex.move_category[int(s.moves[b, side, att, a])] == ids.CATEGORY_STATUS:
            status = status | (1 << a)
    if status != 0:
        return pick_bit(status, raw)
    return pick_bit(moves, raw)


@wp.func
def act(s: State, dex: Dex, b: int, side: int, policy: int, key: wp.uint32, tick: int) -> int:
    """play.js's `pick` for one side of one battle."""
    legal = legal_.legal_actions(s, dex, b, side)
    moves = legal & (0xF | (1 << mask_.ACTION_FORCED))
    switches = legal & (0x3F << mask_.SWITCH_BASE)
    base = rng_.mix32(key ^ wp.uint32(tick) * rng_.KNUTH) ^ rng_.mix32(wp.uint32(b * 2 + side))
    if moves == 0 and switches == 0:
        return mask_.ACTION_PASS
    if moves == 0:
        if policy == EMERALD or policy == MAXDAMAGE:
            return mask_.SWITCH_BASE + policy_emerald.choose_replacement(s, dex, b, side)
        return pick_bit(switches, rng_.rng_u32(base, wp.uint32(0)))
    wanders = policy == RANDOM or policy == EMERALD or policy == STATUS
    if wanders and switches != 0 and rng_.random_n(rng_.rng_u32(base, wp.uint32(1)), 3) == 0:
        return pick_bit(switches, rng_.rng_u32(base, wp.uint32(2)))
    if policy == MAXDAMAGE:
        return maxdamage_move(s, dex, b, side, moves)
    if policy == STATUS:
        return status_move(s, dex, b, side, moves, rng_.rng_u32(base, wp.uint32(3)))
    return pick_bit(moves, rng_.rng_u32(base, wp.uint32(3)))
