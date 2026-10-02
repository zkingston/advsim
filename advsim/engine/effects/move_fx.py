"""Move families that need code rather than table values.

These are the ones whose numbers depend on the battle: how hurt the user is,
what status it carries, what the weather is doing.
"""
import warp as wp

from advsim.engine import damage as dmg
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import mon
from advsim.engine import moves as mv



@wp.func
def hp_scaled_power(hp: int, maxhp: int) -> int:
    """Flail and Reversal, off the Gen 3 ladder of HP ratios in forty-eighths."""
    ratio = wp.max(hp * 48 / maxhp, 1)
    if ratio < 2:
        return 200
    if ratio < 5:
        return 150
    if ratio < 10:
        return 100
    if ratio < 17:
        return 80
    if ratio < 33:
        return 40
    return 20


@wp.func
def status_boosted_power(status: int) -> int:
    """Facade doubles for any status but sleep."""
    if status != 0 and status != ids.COND_SLP:
        return dmg.chain(dmg.NEUTRAL, 2, 1)
    return dmg.NEUTRAL


@wp.func
def heal_weather_numerator(weather: int) -> int:
    """Synthesis and friends: two thirds in sun, a quarter in other weather,
    half in clear skies."""
    if weather == ids.WEATHER_SUN:
        return 2
    if weather == ids.WEATHER_RAIN or weather == ids.WEATHER_SAND:
        return 1
    return 1


@wp.func
def heal_weather_denominator(weather: int) -> int:
    if weather == ids.WEATHER_SUN:
        return 3
    if weather == ids.WEATHER_RAIN or weather == ids.WEATHER_SAND:
        return 4
    return 2


@wp.func
def weather_accuracy(weather: int, accuracy: int) -> int:
    """Thunder never misses in rain and halves its accuracy in sun."""
    if weather == ids.WEATHER_RAIN:
        return mv.NEVER_MISS
    if weather == ids.WEATHER_SUN:
        return 50
    return accuracy


@wp.func
def cure_team(s: State, dex: Dex, b: int, side: int, sound: int):
    """Heal Bell and Aromatherapy clear the whole party. Only Heal Bell is a
    sound move, and only it passes over a Soundproof holder."""
    for slot in range(6):
        ab = int(s.ability[b, side, slot])
        deaf = sound != 0 and ab != 0 and dex.ability_family[ab] == ids.ABILITYFAM_SOUNDPROOF
        if int(s.hp[b, side, slot]) > 0 and not deaf:
            mon.cure_status(s, b, side, slot)


@wp.func
def refresh_cures(status: int) -> bool:
    """Refresh works on burn, paralysis and poison, and fails on the rest."""
    return status == ids.COND_BRN or status == ids.COND_PAR or \
        status == ids.COND_PSN or status == ids.COND_TOX


@wp.func
def stall_denominator(stall_ctr: int) -> int:
    """Protect and Endure halve their odds each consecutive turn: the first use
    is free, then one in two, one in four, and so on."""
    if stall_ctr <= 0:
        return 0  # no roll at all
    return stall_ctr


@wp.func
def solar_beam_weather_power(weather: int, power: int) -> int:
    """Solar Beam is halved in anything but sun and clear skies."""
    if weather == ids.WEATHER_RAIN or weather == ids.WEATHER_SAND:
        return dmg.apply_mod(power, dmg.chain(dmg.NEUTRAL, 1, 2))
    return power
