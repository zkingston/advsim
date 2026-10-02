"""Gen 3 damage, in Showdown's arithmetic and Showdown's order.

Two rounding conventions matter and are easy to get wrong. `modify` truncates
`(value * modifier + 2047) / 4096` with the modifier itself truncated from a
fraction, and `chainModify` combines modifiers at 4096 scale with a different
rounding, `(prev * next + 2048) >> 12`, before a single `modify` applies the
result. Handlers inside one event chain; separate steps of `modifyDamage` do
not. Integer fractions reproduce Showdown's float modifiers exactly: 1.5 is
6144, 0.5 is 2048, 1.1 is 4505.
"""
import warp as wp

from advsim.engine._generated import ids

NEUTRAL = wp.constant(4096)
PHYSICAL = wp.constant(0)


@wp.func
def chain(mod: int, num: int, den: int) -> int:
    """`chainModify`: fold one handler's multiplier into the running modifier."""
    nxt = (num * 4096) / den
    return (mod * nxt + 2048) / 4096


@wp.func
def apply_mod(value: int, mod: int) -> int:
    """`modify` with an already-scaled modifier; int64 because value * mod overflows u32."""
    return int((wp.int64(value) * wp.int64(mod) + wp.int64(2047)) / wp.int64(4096))


@wp.func
def modify(value: int, num: int, den: int) -> int:
    return apply_mod(value, (num * 4096) / den)


@wp.func
def boosted_stat(stat: int, boost: int) -> int:
    """Showdown's boost table is exactly (2 + b) / 2, so no table is needed."""
    b = wp.clamp(boost, -6, 6)
    if b >= 0:
        return stat * (2 + b) / 2
    return stat * 2 / (2 - b)


@wp.func
def offensive_boost(boost: int, crit: bool) -> int:
    """A critical hit ignores the attacker's negative offensive boosts."""
    if crit and boost < 0:
        return 0
    return boost


@wp.func
def defensive_boost(boost: int, crit: bool) -> int:
    """A critical hit ignores the defender's positive defensive boosts."""
    if crit and boost > 0:
        return 0
    return boost


@wp.func
def crit_denominator(crit_stage: int) -> int:
    """critMult = [0, 16, 8, 4, 3, 2] indexed by the move's crit ratio."""
    ratio = wp.clamp(crit_stage + 1, 0, 5)
    if ratio <= 1:
        return 16
    if ratio == 2:
        return 8
    if ratio == 3:
        return 4
    if ratio == 4:
        return 3
    return 2


@wp.func
def random_crit(raw: wp.uint32, denominator: int) -> bool:
    """`randomChance(1, critMult[ratio])`, drawn inside getDamage before the roll."""
    return int((wp.uint64(raw) * wp.uint64(denominator)) >> wp.uint64(32)) < 1


@wp.func
def type_multiplier(dex_chart: wp.array2d(dtype=wp.int32), move_type: int, t1: int, t2: int) -> int:
    """Sum of the per-type codes; anything below -6 is an immunity, not a resistance."""
    m = dex_chart[move_type, t1]
    if t2 != t1:
        m += dex_chart[move_type, t2]
    if m < -6:
        return -99
    return wp.clamp(m, -6, 6)


@wp.func
def defense_for_hit(defense: int, category: int, selfdestruct: int) -> int:
    """Gen 3 halves the defender's Def against Explosion and Self-Destruct."""
    if selfdestruct != 0 and category == PHYSICAL:
        return wp.max(defense / 2, 1)
    return defense


@wp.func
def base_damage(level: int, power: int, attack: int, defense: int) -> int:
    """trunc(trunc(trunc(trunc(2L/5 + 2) * power * atk) / def) / 50)."""
    return ((2 * level / 5 + 2) * power * attack / defense) / 50


@wp.func
def modify_damage(bd: int, category: int, burned: bool, guts: bool, phase1_mod: int,
                  weather_mod: int, crit: bool, stab: bool, typemod: int, roll: int) -> int:
    """The gen3 `modifyDamage` chain, from the burn halving to the damage roll.

    `roll` is Showdown's `random(16)`; the caller draws it so that no draw
    happens anywhere but at Showdown's own call sites.
    """
    physical = category == PHYSICAL
    if burned and bd > 0 and physical and not guts:
        bd = modify(bd, 1, 2)
    bd = apply_mod(bd, phase1_mod)   # ModifyDamagePhase1: Flash Fire
    bd = apply_mod(bd, weather_mod)  # sun and rain on Fire and Water
    if physical and bd == 0:
        bd = 1
    bd += 2
    if crit:
        bd = modify(bd, 2, 1)
    if stab:
        bd = modify(bd, 3, 2)
    for _ in range(wp.max(typemod, 0)):
        bd *= 2
    for _ in range(wp.max(-typemod, 0)):
        bd = bd / 2
    bd = (bd * (100 - roll)) / 100
    if bd == 0:
        return 1
    return bd


@wp.func
def weather_modifier(weather: int, move_type: int, fire: int, water: int) -> int:
    """Sun boosts Fire and halves Water; rain does the reverse."""
    mod = NEUTRAL
    if weather == ids.WEATHER_SUN:
        if move_type == fire:
            mod = chain(mod, 3, 2)
        elif move_type == water:
            mod = chain(mod, 1, 2)
    elif weather == ids.WEATHER_RAIN:
        if move_type == water:
            mod = chain(mod, 3, 2)
        elif move_type == fire:
            mod = chain(mod, 1, 2)
    return mod
