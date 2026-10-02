"""Ability families.

Most abilities that carry callbacks fall into a few parametric shapes; the rest
are one-offs. Several of these draw: a contact punisher rolls
once per contact hit, Effect Spore rolls again to pick which status, and Shed
Skin rolls every residual phase while its holder is statused.
"""
import warp as wp

from advsim.engine import damage as dmg
from advsim.engine._generated import ids
from advsim.engine.dex import Dex

SPORE = wp.constant(-1)   # SENTINELS in build/tables.py
MODE_CHAIN = wp.constant(0)
MODE_IMMEDIATE = wp.constant(1)
ABSORB_HEAL = wp.constant(1)
ABSORB_FLASH_FIRE = wp.constant(2)


@wp.func
def stat_modifier(dex: Dex, ability: int, stat: int, statused: bool, mode: int,
                  partner: int) -> int:
    """The 4096-scaled modifier this ability gives one stat, for one apply mode.
    `partner` is the other active's ability, which is what Plus and Minus read."""
    if ability == 0 or dex.ability_family[ability] != ids.ABILITYFAM_STAT_MULT:
        return dmg.NEUTRAL
    if ((dex.ability_p0[ability] >> stat) & 1) == 0:
        return dmg.NEUTRAL
    if dex.ability_p2[ability] == 1 and not statused:
        return dmg.NEUTRAL
    if dex.ability_p2[ability] == 2:
        # Plus wants Minus and Minus wants Plus: the other half is the other
        # ability in this family that asks the same question.
        if partner == 0 or partner == ability or \
                dex.ability_family[partner] != ids.ABILITYFAM_STAT_MULT or \
                dex.ability_p2[partner] != 2:
            return dmg.NEUTRAL
    if dex.ability_p3[ability] != mode:
        return dmg.NEUTRAL
    return dex.ability_p1[ability]


@wp.func
def speed_modifier(dex: Dex, ability: int, weather: int) -> int:
    """Swift Swim and Chlorophyll double Speed in their weather."""
    if ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_WEATHER_SPEED:
        if dex.ability_p0[ability] == weather:
            return dmg.chain(dmg.NEUTRAL, 2, 1)
    return dmg.NEUTRAL


@wp.func
def base_power_modifier(dex: Dex, ability: int, move_type: int, hp: int, maxhp: int) -> int:
    """Overgrow and friends: half again on their type at a third HP or less."""
    if ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_PINCH_TYPE_BOOST:
        if dex.ability_p0[ability] == move_type and hp * 3 <= maxhp:
            return dmg.chain(dmg.NEUTRAL, 3, 2)
    return dmg.NEUTRAL


@wp.func
def one_accuracy_modifier(dex: Dex, ability: int, whose: int, move_type: int,
                          weather: int, sand: int) -> int:
    """This ability's share of the accuracy chain, or neutral when it has none."""
    if ability == 0 or dex.ability_acc_whose[ability] != whose or dex.ability_acc_mod[ability] == 0:
        return dmg.NEUTRAL
    when = dex.ability_acc_when[ability]
    if when == 1:
        # The nine types gen 3 counts as physical, which is every type but the
        # eight special ones. Typeless is on neither list.
        if move_type == 0 or dex.type_is_special[move_type] != 0:
            return dmg.NEUTRAL
    elif when == 2:
        if weather != sand:
            return dmg.NEUTRAL
    return dex.ability_acc_mod[ability]


@wp.func
def accuracy_modifier(dex: Dex, attacker: int, defender: int, move_type: int,
                      weather: int, sand: int) -> int:
    """Both sides' accuracy handlers, chained in Showdown's priority order.

    Each step truncates, so the order is not free: Compound Eyes runs at 9,
    Sand Veil at 8 and Hustle at 7, and only one of each side can apply.
    """
    a = one_accuracy_modifier(dex, attacker, 0, move_type, weather, sand)
    d = one_accuracy_modifier(dex, defender, 1, move_type, weather, sand)
    if a == dmg.NEUTRAL:
        return d
    if d == dmg.NEUTRAL:
        return a
    if dex.ability_acc_prio[attacker] >= dex.ability_acc_prio[defender]:
        return dmg.chain(dmg.chain(dmg.NEUTRAL, a, 4096), d, 4096)
    return dmg.chain(dmg.chain(dmg.NEUTRAL, d, 4096), a, 4096)


@wp.func
def has(dex: Dex, ability: int, family: int) -> bool:
    """Whether this ability is the one-off that family stands for."""
    return ability != 0 and dex.ability_family[ability] == family


@wp.func
def weakens_type(dex: Dex, ability: int, move_type: int, fire: int, ice: int) -> int:
    """Thick Fat halves the base power of Fire and Ice aimed at its holder, in
    the same event the attacker's pinch boost uses but at a lower priority."""
    if has(dex, ability, ids.ABILITYFAM_THICKFAT) and (move_type == fire or move_type == ice):
        return dmg.chain(dmg.NEUTRAL, 1, 2)
    return dmg.NEUTRAL


@wp.func
def blocks_sand(dex: Dex, ability: int) -> bool:
    """Sand Veil sits out the sandstorm it hides in."""
    return ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_SANDVEIL


@wp.func
def blocks_crit(dex: Dex, ability: int) -> bool:
    return ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_CRIT_IMMUNE


@wp.func
def ignores_burn_drop(ability: int) -> bool:
    """Guts is checked by name in modifyDamage, so a burn does not halve its Attack."""
    return ability == ids.ABILITY_GUTS


@wp.func
def blocks_status(dex: Dex, ability: int, status: int) -> bool:
    if ability == 0 or dex.ability_family[ability] != ids.ABILITYFAM_STATUS_IMMUNE:
        return False
    return dex.ability_p0[ability] == status


@wp.func
def blocks_drop(dex: Dex, ability: int, stat: int) -> bool:
    """Clear Body and the narrower ones refuse the foe's stat drops."""
    if ability == 0 or dex.ability_family[ability] != ids.ABILITYFAM_BLOCK_DROPS:
        return False
    return ((dex.ability_p0[ability] >> stat) & 1) != 0


@wp.func
def absorbs_type(dex: Dex, ability: int, move_type: int, move: int) -> int:
    """0 if the move lands, otherwise the absorb effect. Volt Absorb lets Thunder
    Wave through, which is the one move-specific exception in this family."""
    if ability == 0 or dex.ability_family[ability] != ids.ABILITYFAM_TYPE_IMMUNE:
        return 0
    if dex.ability_p0[ability] != move_type:
        return 0
    if move == ids.MOVE_THUNDERWAVE and dex.ability_p1[ability] == ABSORB_HEAL:
        return 0
    return dex.ability_p1[ability] + 1  # +1 so "no effect but still immune" is not 0


@wp.func
def weather_set_by(dex: Dex, ability: int) -> int:
    if ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_WEATHER_SETTER:
        return dex.ability_p0[ability]
    return 0


@wp.func
def suppresses_weather(dex: Dex, ability: int) -> bool:
    return ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_WEATHER_SUPPRESS


@wp.func
def traps_foe(dex: Dex, ability: int, foe_grounded: bool, foe_is_steel: bool) -> bool:
    if ability == 0 or dex.ability_family[ability] != ids.ABILITYFAM_TRAP:
        return False
    mode = dex.ability_p0[ability]
    if mode == 0:
        return True
    if mode == 1:
        return foe_grounded
    return foe_is_steel


@wp.func
def is_trapper(dex: Dex, ability: int) -> bool:
    """Holds one of the three trapping abilities, whether or not it would bite."""
    return ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_TRAP


@wp.func
def traps_from_anywhere(dex: Dex, ability: int) -> bool:
    """Magnet Pull alone answers as an Any handler in gen 3, so it is found for
    its own holder as well as for the Pokemon across the field."""
    return is_trapper(dex, ability) and dex.ability_p0[ability] == 2


@wp.func
def contact_recoil(dex: Dex, ability: int, attacker_maxhp: int) -> int:
    """Rough Skin: a fixed bite out of the attacker, with no roll."""
    if ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_CONTACT_RECOIL:
        return wp.max(attacker_maxhp / dex.ability_p0[ability], 1)
    return 0
