"""The generic move path: one code path every move walks, with neutral values.

Nothing here branches on a move's identity. A status move has power 0, a move
with no secondary has chance 0, and so on, so the same instructions run in every
lane and divergence stays low. Families and one-offs hook in around it, in
`execute.py`, `move_effects.py` and `effects/`.

Draw order follows Showdown's `tryMoveHit`: the accuracy roll happens even when
the target is immune, and only the damage draws are skipped.
"""
import warp as wp

from advsim.engine import damage as dmg
from advsim.engine._generated import ids
from advsim.engine.dex import Dex

NEVER_MISS = wp.constant(255)  # move_accuracy
FIXED_LEVEL = wp.constant(255)  # move_fixed_damage: the user's level
TARGET_SELF = wp.constant(ids.TARGET_SELF)  # the generated table owns the indices
TARGET_ALL = wp.constant(ids.TARGET_ALL)
TARGET_ALLY_TEAM = wp.constant(ids.TARGET_ALLYTEAM)
TARGET_FOE_SIDE = wp.constant(ids.TARGET_FOESIDE)
TOX_MAPS_TO_PSN_FROM = wp.constant(ids.COND_TOX)
TOX_MAPS_TO_PSN_TO = wp.constant(ids.COND_PSN)


@wp.func
def get_boost(packed: int, index: int) -> int:
    """Seven signed nibbles in one word; the nibble is two's complement."""
    nib = (packed >> (4 * index)) & 0xF
    if nib >= 8:
        return nib - 16
    return nib


@wp.func
def set_boost(packed: int, index: int, value: int) -> int:
    v = wp.clamp(value, -6, 6)
    return (packed & ~(0xF << (4 * index))) | ((v & 0xF) << (4 * index))


@wp.func
def apply_boost_word(packed: int, word: int) -> int:
    """Add a move's packed boost word to a slot's boosts, clamping each stat."""
    out = packed
    for i in range(7):
        delta = get_boost(word, i)
        if delta != 0:
            out = set_boost(out, i, get_boost(out, i) + delta)
    return out


@wp.func
def targets_the_field(target: int) -> bool:
    """A move aimed at the field or a side takes an earlier branch in tryMoveHit:
    no accuracy roll, no immunity check, and no chance for the foe's ability to
    absorb it. Rain Dance is a Water move, so this matters."""
    return target == TARGET_ALL or target == TARGET_ALLY_TEAM or target == TARGET_FOE_SIDE


@wp.func
def status_lands(dex: Dex, status: int, t1: int, t2: int, current: int) -> bool:
    """A type can be immune to a status: Fire to burn, Poison and Steel to poison.

    Toxic checks the poison immunity, as Showdown maps tox to psn before looking.
    """
    if current != 0 or status == 0:
        return False
    probe = status
    if status == TOX_MAPS_TO_PSN_FROM:
        probe = TOX_MAPS_TO_PSN_TO
    blocked = dex.type_status_immune[t1] | dex.type_status_immune[t2]
    return (blocked & (1 << probe)) == 0


@wp.func
def accuracy_check(raw: wp.uint32, accuracy: int) -> bool:
    """`randomChance(accuracy, 100)`. No move in this format alters accuracy or
    evasion, so the boost table never applies and accuracy stays integral."""
    return int((wp.uint64(raw) * wp.uint64(100)) >> wp.uint64(32)) < accuracy


@wp.func
def type_multiplier_of(dex: Dex, move_type: int, def_t1: int, def_t2: int) -> int:
    """Type effectiveness for a move against a defender; -99 means immune."""
    return dmg.type_multiplier(dex.type_chart, move_type, def_t1, def_t2)


@wp.func
def type_of(dex: Dex, move: int, hp_type: int) -> int:
    """Hidden Power is the one move whose type comes from the Pokemon, not the move."""
    if dex.move_type_from_mon[move] != 0:
        return hp_type
    return dex.move_type[move]


@wp.func
def hit_damage(dex: Dex, move: int, move_type: int, category: int, level: int, power: int, atk: int, df: int,
               atk_boost: int, def_boost: int, atk_imm: int, atk_mod: int, def_imm: int, def_mod: int,
               crit_stage: int, crit_blocked: bool, guts: bool,
               att_t1: int, att_t2: int, def_t1: int, def_t2: int, burned: bool, weather: int,
               phase1: int, crit_raw: wp.uint32, roll_raw: wp.uint32, fire: int, water: int) -> int:
    """One hit: the crit draw, then the damage pipeline, then the roll draw.

    Stats arrive raw, because whether a boost counts depends on the crit that is
    drawn here: a crit ignores the attacker's negative and the defender's
    positive boosts. Item modifiers apply after the boost, as Showdown's
    ModifyAtk event does.
    """
    tm = type_multiplier_of(dex, move_type, def_t1, def_t2)
    is_crit = dmg.random_crit(crit_raw, dmg.crit_denominator(crit_stage))
    if crit_blocked:
        is_crit = False  # Battle Armor and Shell Armor refuse the crit after the roll
    # An immediate modifier (Hustle) is applied as its handler runs; the chained
    # ones (abilities then items) combine and apply once at the end of the event.
    a = dmg.apply_mod(dmg.apply_mod(dmg.boosted_stat(atk, dmg.offensive_boost(atk_boost, is_crit)), atk_imm), atk_mod)
    d = dmg.apply_mod(dmg.apply_mod(dmg.boosted_stat(df, dmg.defensive_boost(def_boost, is_crit)), def_imm), def_mod)
    d = dmg.defense_for_hit(d, category, dex.move_selfdestruct[move])
    bd = dmg.base_damage(level, power, a, d)
    stab = move_type == att_t1 or move_type == att_t2
    wmod = dmg.weather_modifier(weather, move_type, fire, water)
    roll = int((wp.uint64(roll_raw) * wp.uint64(16)) >> wp.uint64(32))
    return dmg.modify_damage(bd, category, burned, guts, phase1, wmod, is_crit, stab, tm, roll)
