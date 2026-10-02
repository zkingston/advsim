"""The Emerald cartridge AI's choice of replacement, as a scripted baseline.

The same chooser as `showdown/lib/policy.js`, which the sweep plays with, and
the same tables, built from `showdown/lib/emerald.json`. It is a policy, not
battle semantics: it reads the state, writes nothing and draws nothing.
"""
import warp as wp

from advsim.engine import moves as mv
from advsim.engine import weather as weather_
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex


@wp.func
def apply_type(dex: Dex, value: int, atk: int, d1: int, d2: int) -> int:
    """One attacking type against a defender, row by row, flooring at each step."""
    v = int(value)
    for i in range(dex.type_ai_order.shape[0]):
        d = dex.type_ai_order[i, 1]
        if dex.type_ai_order[i, 0] == atk and (d == d1 or d == d2):
            v = v * dex.type_ai_order[i, 2] / 10
    return v


@wp.func
def eligible(s: State, b: int, side: int, slot: int) -> bool:
    return int(s.species[b, side, slot]) != 0 and int(s.hp[b, side, slot]) > 0 and slot != int(s.active[b, side])


@wp.func
def section1(s: State, dex: Dex, b: int, side: int, f1: int, f2: int, levitate: bool) -> int:
    """The best-scoring bench Pokemon with a super-effective damaging move, or -1.
    A single-typed foe is stored with its type twice, as on the cartridge, so
    it hits twice."""
    left = int(0x3F)
    for _ in range(6):
        best = int(-1)
        best_score = int(0)
        for slot in range(6):
            if ((left >> slot) & 1) != 0 and eligible(s, b, side, slot):
                sp = int(s.species[b, side, slot])
                t1 = dex.species_type1[sp]
                t2 = dex.species_type2[sp]
                score = apply_type(dex, apply_type(dex, 10, f1, t1, t2), f2, t1, t2)
                if score > best_score:
                    best = slot
                    best_score = score
        if best < 0:
            return -1
        for i in range(4):
            m = int(s.moves[b, side, best, i])
            if m != 0 and dex.move_category[m] != ids.CATEGORY_STATUS:
                t = dex.move_type[m]
                if not (t == ids.TYPE_GROUND and levitate) and apply_type(dex, 10, t, f1, f2) > 10:
                    return best
        left = left & ~(1 << best)
    return -1


@wp.func
def base_damage(s: State, dex: Dex, b: int, side: int, foe: int) -> int:
    """CalculateBaseDamage for the last move used, as the Pokemon going out would
    deal it: max roll, no crit, no STAB or effectiveness. Mirrors `baseDamage`
    in policy.js line for line; this vocabulary has no screens."""
    move = int(s.last_used[b])
    if move == 0 or dex.move_category[move] == ids.CATEGORY_STATUS:
        return 3
    t = dex.move_type[move]
    physical = dex.type_is_special[t] == 0
    att = int(s.active[b, side])
    dfn = int(s.active[b, foe])
    ability = int(s.ability[b, side, att])
    item = int(s.item[b, side, att])
    species = int(s.species[b, side, att])
    status = int(s.status[b, side, att])
    power = dex.move_power[move]
    if dex.move_ai_skip[move] != 0:
        power = 1
    a_idx = 2
    d_idx = 3
    if physical:
        a_idx = 0
        d_idx = 1
    atk = int(s.stats[b, side, att, a_idx])
    df = int(s.stats[b, foe, dfn, d_idx])
    if ability == ids.ABILITY_HUGEPOWER or ability == ids.ABILITY_PUREPOWER:
        atk = atk * 2
    if item == ids.ITEM_CHOICEBAND and physical:
        atk = atk * 15 / 10
    if (item == ids.ITEM_SILKSCARF and t == ids.TYPE_NORMAL) or (item == ids.ITEM_TWISTEDSPOON and t == ids.TYPE_PSYCHIC):
        atk = atk * 110 / 100
    latis = species == ids.SPECIES_LATIOS or species == ids.SPECIES_LATIAS
    if item == ids.ITEM_SOULDEW and latis and not physical:
        atk = atk * 15 / 10
    if item == ids.ITEM_LIGHTBALL and species == ids.SPECIES_PIKACHU and not physical:
        atk = atk * 2
    if item == ids.ITEM_THICKCLUB and species == ids.SPECIES_MAROWAK and physical:
        atk = atk * 2
    foe_species = int(s.species[b, foe, dfn])
    if int(s.item[b, foe, dfn]) == ids.ITEM_SOULDEW and not physical and \
            (foe_species == ids.SPECIES_LATIOS or foe_species == ids.SPECIES_LATIAS):
        df = df * 15 / 10
    if ability == ids.ABILITY_HUSTLE and physical:
        atk = atk * 15 / 10
    if ability == ids.ABILITY_GUTS and status != 0 and physical:
        atk = atk * 15 / 10
    if int(s.ability[b, foe, dfn]) == ids.ABILITY_MARVELSCALE and int(s.status[b, foe, dfn]) != 0 and physical:
        df = df * 15 / 10
    pinch = int(-1)
    if ability == ids.ABILITY_OVERGROW:
        pinch = ids.TYPE_GRASS
    elif ability == ids.ABILITY_BLAZE:
        pinch = ids.TYPE_FIRE
    elif ability == ids.ABILITY_TORRENT:
        pinch = ids.TYPE_WATER
    elif ability == ids.ABILITY_SWARM:
        pinch = ids.TYPE_BUG
    if pinch == t and int(s.hp[b, side, att]) <= int(s.maxhp[b, side, att]) / 3:
        power = power * 15 / 10
    ab = mv.get_boost(int(s.boosts[b, side]), a_idx)
    db = mv.get_boost(int(s.boosts[b, foe]), d_idx)
    if ab >= 0:
        atk = atk * (2 + ab) / 2
    else:
        atk = atk * 2 / (2 - ab)
    if db >= 0:
        df = df * (2 + db) / 2
    else:
        df = df * 2 / (2 - db)
    damage = atk * power * (2 * int(s.level[b, side, att]) / 5 + 2) / df / 50
    if physical and status == ids.COND_BRN and ability != ids.ABILITY_GUTS:
        damage = damage / 2
    weather = weather_.effective_weather(s, dex, b)
    if not physical and weather == ids.WEATHER_RAIN:
        if t == ids.TYPE_WATER:
            damage = damage * 15 / 10
        if t == ids.TYPE_FIRE:
            damage = damage / 2
    if not physical and weather == ids.WEATHER_SUN:
        if t == ids.TYPE_FIRE:
            damage = damage * 15 / 10
        if t == ids.TYPE_WATER:
            damage = damage / 2
    return wp.max(damage, 1) + 2


@wp.func
def section2(s: State, dex: Dex, b: int, side: int, foe: int, f1: int, f2: int, levitate: bool) -> int:
    """The bench Pokemon with the move the AI thinks hits hardest, or -1. STAB
    is the outgoing Pokemon's; the best damage is kept in a byte."""
    base = base_damage(s, dex, b, side, foe)
    s1 = int(s.types[b, side, 0])
    s2 = int(s.types[b, side, 1])
    best = int(-1)
    best_dmg = int(0)
    for slot in range(6):
        if eligible(s, b, side, slot):
            for i in range(4):
                m = int(s.moves[b, side, slot, i])
                if m != 0 and dex.move_ai_skip[m] == 0:
                    t = dex.move_type[m]
                    dmg = int(base)
                    if t == s1 or t == s2:
                        dmg = dmg * 15 / 10
                    if not (t == ids.TYPE_GROUND and levitate):
                        dmg = apply_type(dex, dmg, t, f1, f2)
                    if dmg > best_dmg:
                        best_dmg = dmg % 256
                        best = slot
    return best


@wp.func
def choose_replacement(s: State, dex: Dex, b: int, side: int) -> int:
    """The party slot to send in; immune to everything means the first that can."""
    foe = 1 - side
    dfn = int(s.active[b, foe])
    f1 = int(s.types[b, foe, 0])
    f2 = int(s.types[b, foe, 1])
    levitate = int(s.ability[b, foe, dfn]) == ids.ABILITY_LEVITATE
    pick = int(section1(s, dex, b, side, f1, f2, levitate))
    if pick < 0:
        pick = section2(s, dex, b, side, foe, f1, f2, levitate)
    if pick < 0:
        for slot in range(6):
            if pick < 0 and eligible(s, b, side, slot):
                pick = slot
    return pick
