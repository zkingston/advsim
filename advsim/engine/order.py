"""Turn order: effective speed, priority comparison, and Showdown's tie shuffles.

Gen 3's `getActionSpeed` reads the boosted, modified Speed. Paralysis quarters
it through a chained modifier, so it rounds like every other modifier rather
than by plain division. Ties are not broken deterministically: `speedSort`
shuffles them, and those draws are part of the RNG stream that replay parity
depends on, so they are taken here at the same point Showdown takes them.

Quick Claw does not exist in this vocabulary, but Gen 3 still rolls for it once
per `endTurn`; `turn.py` consumes that draw.
"""
import warp as wp

from advsim.engine import damage as dmg
from advsim.engine import moves as mv
from advsim.engine import rng as rng_
from advsim.engine.effects import ability_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import weather as weather_

SPEED_CAP = wp.constant(10000)


@wp.func
def effective_speed(spe: int, boost: int, paralyzed: bool, speed_mod: int) -> int:
    """Boosted Speed with the modifiers applied, capped as Showdown caps it.

    `speed_mod` carries ability and item multipliers already chained at 4096
    scale, so callers add Swift Swim or Chlorophyll without touching this.
    """
    stat = dmg.boosted_stat(spe, boost)
    mod = speed_mod
    if paralyzed:
        mod = dmg.chain(mod, 1, 4)
    stat = dmg.apply_mod(stat, mod)
    if stat > SPEED_CAP:
        return SPEED_CAP
    return stat


@wp.func
def moves_first(priority_a: int, speed_a: int, priority_b: int, speed_b: int) -> int:
    """-1 if a goes first, 1 if b does, 0 on an exact tie that must be shuffled."""
    if priority_a != priority_b:
        if priority_a > priority_b:
            return -1
        return 1
    if speed_a != speed_b:
        if speed_a > speed_b:
            return -1
        return 1
    return 0


@wp.func
def tie_goes_to_second(raw: wp.uint32) -> bool:
    """`speedSort` on two tied actions is one `shuffle`, which is one `random(0, 2)`."""
    return (int((wp.uint64(raw) * wp.uint64(2)) >> wp.uint64(32))) == 1


@wp.func
def speed_of(s: State, dex: Dex, b: int, side: int) -> int:
    """`getActionSpeed`: the Speed an action is queued with, worked out fresh."""
    slot = int(s.active[b, side])
    paralyzed = int(s.status[b, side, slot]) == ids.COND_PAR
    speed_mod = ability_fx.speed_modifier(dex, int(s.ability[b, side, slot]), weather_.effective_weather(s, dex, b))
    return effective_speed(int(s.stats[b, side, slot, 4]),
                                mv.get_boost(int(s.boosts[b, side]), 4), paralyzed, speed_mod)


@wp.func
def sort_speed(s: State, b: int, side: int) -> int:
    """`pokemon.speed`: what every speedSort reads. It is a cached value, and
    only updateSpeed refreshes it, so a Pokemon that has just come in sorts at
    whatever it had the last time it was on the field."""
    return int(s.cached_spe[b, side, int(s.active[b, side])])


@wp.func
def update_speed(s: State, dex: Dex, b: int):
    """updateSpeed: the actives, and only the actives, get a fresh Speed."""
    for side in range(2):
        s.cached_spe[b, side, int(s.active[b, side])] = wp.uint16(speed_of(s, dex, b, side))


@wp.func
def tie_draw(s: State, log: wp.array2d(dtype=wp.uint32), b: int, both_alive: bool):
    """`speedSort` over the two actives draws only when their speeds tie exactly.

    Showdown sorts them far more often than a turn's logic suggests: once per
    eachEvent, and gen 3 runs one at the end of every action. Those draws are
    invisible in the battle log and are the main way a replay goes out of step.
    """
    if both_alive and sort_speed(s, b, 0) == sort_speed(s, b, 1):
        rng_.draw(s, log, b)


@wp.func
def both_actives_alive(s: State, b: int) -> bool:
    return int(s.hp[b, 0, int(s.active[b, 0])]) > 0 and int(s.hp[b, 1, int(s.active[b, 1])]) > 0


@wp.func
def queue_speed(s: State, dex: Dex, b: int, side: int) -> int:
    """The Speed an action is queued with. A Pokemon that has fainted is off
    the field by then, so nothing modifies its stat any more."""
    slot = int(s.active[b, side])
    if int(s.hp[b, side, slot]) <= 0:
        return int(s.stats[b, side, slot, 4])
    return speed_of(s, dex, b, side)
