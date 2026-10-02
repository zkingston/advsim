"""Status conditions: whether one lands, what it stops, and the Update event that
lets a berry or an ability cure it. Also the end of an action, which is a sort
of the actives and then that Update."""
import warp as wp

from advsim.engine import damage as dmg
from advsim.engine import mask as mask_
from advsim.engine import moves as mv
from advsim.engine import order as ord_
from advsim.engine import rng as rng_
from advsim.engine import mon
from advsim.engine.effects import ability_fx
from advsim.engine.effects import item_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import weather as weather_


@wp.func
def confusion_blocks(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int) -> bool:
    """BeforeMove priority 3. The counter runs down first and reaching zero
    costs nothing; otherwise a coin decides, and losing it is a 40-power
    typeless physical hit on yourself that never crits."""
    turns = int(s.confusion_turns[b, side])
    if turns == 0:
        return False
    s.confusion_turns[b, side] = wp.uint8(turns - 1)
    if turns - 1 <= 0:
        return False
    if rng_.random_chance(rng_.draw(s, log, b), 1, 2):
        return False
    slot = int(s.active[b, side])
    ability = int(s.ability[b, side, slot])
    item = int(s.item[b, side, slot])
    num = dex.species_num[int(s.species[b, side, slot])]
    statused = int(s.status[b, side, slot]) != 0
    atk_mod = dmg.chain(ability_fx.stat_modifier(dex, ability, 0, statused, ability_fx.MODE_CHAIN, 0),
                        item_fx.stat_modifier(dex, item, num, 0, 0), 4096)
    def_mod = dmg.chain(ability_fx.stat_modifier(dex, ability, 1, statused, ability_fx.MODE_CHAIN, 0),
                        item_fx.stat_modifier(dex, item, num, 1, 0), 4096)
    boosts = int(s.boosts[b, side])
    t1 = int(s.types[b, side, 0])
    t2 = int(s.types[b, side, 1])
    hurt = mv.hit_damage(
        dex, 0, 0, 0, int(s.level[b, side, slot]), 40,
        int(s.stats[b, side, slot, 0]), int(s.stats[b, side, slot, 1]),
        mv.get_boost(boosts, 0), mv.get_boost(boosts, 1),
        ability_fx.stat_modifier(dex, ability, 0, statused, ability_fx.MODE_IMMEDIATE, 0), atk_mod,
        ability_fx.stat_modifier(dex, ability, 1, statused, ability_fx.MODE_IMMEDIATE, 0), def_mod,
        0, True, ability_fx.ignores_burn_drop(ability), t1, t2, t1, t2,
        int(s.status[b, side, slot]) == ids.COND_BRN, weather_.effective_weather(s, dex, b), dmg.NEUTRAL,
        wp.uint32(0), rng_.draw(s, log, b), ids.TYPE_FIRE, ids.TYPE_WATER)
    hp = int(s.hp[b, side, slot])
    s.hp[b, side, slot] = wp.uint16(wp.max(hp - wp.min(hurt, hp), 0))
    return True


@wp.func
def can_move(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int, move: int) -> bool:
    """Before-move status checks, in Showdown's order and with its draws. The
    move matters because Sleep Talk is spent from inside sleep rather than
    stopped by it, and the timer runs down all the same."""
    slot = int(s.active[b, side])
    if (int(s.vflags[b, side]) & mask_.VF_MUSTRECHARGE) != 0:
        # Priority 11, above every other BeforeMove handler: the turn is spent
        # recharging and the volatile goes with it, and so does Truant's, which
        # is what makes the recharge count as the loafing turn. `truantTurn`
        # stays, so the residual puts the volatile back in step.
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~(mask_.VF_MUSTRECHARGE | mask_.VF_TRUANT))
        return False
    status = int(s.status[b, side, slot])
    if status == ids.COND_SLP:
        # The timer is spent before it is read, so a Pokemon with one turn left
        # wakes and moves in the same turn. Early Bird spends two.
        spend = 1
        if ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_EARLYBIRD):
            spend = 2
        left = int(s.status_ctr[b, side, slot]) - spend
        s.status_ctr[b, side, slot] = wp.uint8(wp.max(left, 0))
        if left <= 0:
            s.status[b, side, slot] = wp.uint8(0)
            s.slept_by_foe[b, side, slot] = wp.uint8(0)
            s.sleep_skipped[b, side, slot] = wp.uint8(0)
        elif dex.move_family[move] == ids.MOVEFAM_SLEEPTALK:
            # Gen 3 keeps count of the turns Sleep Talk acted through and hands
            # them back the next time the Pokemon comes in.
            s.sleep_skipped[b, side, slot] = wp.uint8(int(s.sleep_skipped[b, side, slot]) + 1)
        else:
            s.sleep_skipped[b, side, slot] = wp.uint8(0)
            return False
    if status == ids.COND_FRZ:
        if rng_.random_chance(rng_.draw(s, log, b), 1, 5):
            mon.cure_status(s, b, side, slot)  # thawed, and the move still runs
        elif (dex.move_flags[move] & ids.FLAG_DEFROST) == 0:
            return False
    if (int(s.vflags[b, side]) & mask_.VF_TRUANT) != 0:
        mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)  # `cant|...|ability: Truant`
        return False  # priority 9, under sleep and over the flinch
    # BeforeMove handlers run by priority: sleep and freeze at 10, Truant at 9,
    # flinch at 8, confusion at 3, attraction at 2, paralysis at 1. The flinch is not cleared here; its
    # duration runs out in the residual phase.
    if (int(s.vflags[b, side]) & mask_.VF_FLINCH) != 0:
        return False
    if confusion_blocks(s, dex, log, b, side):
        return False
    if (int(s.vflags[b, side]) & mask_.VF_ATTRACT) != 0:
        # Attraction rolls every turn it is in place, win or lose, and at
        # priority 2 it asks before paralysis does. A Pokemon that is both
        # rolls twice, and only the first of them can stop the move.
        if rng_.random_chance(rng_.draw(s, log, b), 1, 2):
            return False
    if status == ids.COND_PAR:
        return not rng_.random_chance(rng_.draw(s, log, b), 1, 4)
    return True


@wp.func
def status_lands_on(s: State, dex: Dex, b: int, side: int, slot: int, status: int, from_foe: bool) -> bool:
    """Type immunity, then the abilities that refuse a particular status."""
    if not mv.status_lands(dex, status, int(s.types[b, side, 0]), int(s.types[b, side, 1]),
                           int(s.status[b, side, slot])):
        return False
    if status == ids.COND_FRZ and weather_.effective_weather(s, dex, b) == ids.WEATHER_SUN:
        return False  # nothing freezes in the sun, and Air Lock lifts the ban
    probe = status
    if status == ids.COND_TOX:
        probe = ids.COND_PSN  # Immunity covers both, as the type chart does
    if ability_fx.blocks_status(dex, int(s.ability[b, side, slot]), probe):
        return False
    if status == ids.COND_SLP and from_foe and sleep_clause_blocks(s, b, side):
        return False
    return True


@wp.func
def ability_refuses(s: State, dex: Dex, b: int, side: int, slot: int, status: int) -> bool:
    """Whether a status move failed on the target's ability rather than on an
    existing status or its type: only then does Showdown name the ability."""
    if int(s.status[b, side, slot]) != 0:
        return False
    if not mv.status_lands(dex, status, int(s.types[b, side, 0]), int(s.types[b, side, 1]), 0):
        return False
    probe = status
    if status == ids.COND_TOX:
        probe = ids.COND_PSN
    return ability_fx.blocks_status(dex, int(s.ability[b, side, slot]), probe)


@wp.func
def sleep_clause_blocks(s: State, b: int, side: int) -> bool:
    """Sleep Clause Mod: one foe asleep at a time, counting only sleep the other
    side caused. A Pokemon that put itself to sleep does not hold the slot."""
    for slot in range(6):
        if int(s.hp[b, side, slot]) > 0 and int(s.status[b, side, slot]) == ids.COND_SLP:
            if int(s.slept_by_foe[b, side, slot]) != 0:
                return True
    return False


@wp.func
def inflict_status(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int,
                   side: int, slot: int, status: int, from_foe: bool):
    """Set a status that has already passed its immunity checks. Sleep draws for
    how long it lasts, which is the one status that costs a draw to apply."""
    s.status[b, side, slot] = wp.uint8(status)
    if status == ids.COND_SLP:
        s.status_ctr[b, side, slot] = wp.uint8(rng_.random_range(rng_.draw(s, log, b), 2, 6))
        s.sleep_skipped[b, side, slot] = wp.uint8(0)
        if from_foe:
            s.slept_by_foe[b, side, slot] = wp.uint8(1)
    elif status == ids.COND_TOX:
        s.status_ctr[b, side, slot] = wp.uint8(0)  # the stage steps up before it bites
    else:
        s.status_ctr[b, side, slot] = wp.uint8(1)
    if from_foe and status != ids.COND_SLP and status != ids.COND_FRZ and \
            ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_SYNCHRONIZE):
        # AfterSetStatus hands it straight back, toxic arriving as plain poison.
        # Sleep and freeze are never passed, which is why this needs no roll.
        back = status
        if back == ids.COND_TOX:
            back = ids.COND_PSN
        foe = 1 - side
        foe_slot = int(s.active[b, foe])
        # The handback shows as `-status ... [from] ability: Synchronize|[of]`,
        # so only one that lands says anything; setStatus refuses a source
        # that has already fainted.
        if int(s.hp[b, foe, foe_slot]) > 0 and status_lands_on(s, dex, b, foe, foe_slot, back, True):
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
            s.status[b, foe, foe_slot] = wp.uint8(back)
            s.status_ctr[b, foe, foe_slot] = wp.uint8(1)
            item_fx.lum_berry(s, dex, b, foe, foe_slot, False)
    # AfterSetStatus, where a berry is eaten straight away rather than waiting
    # for the Update event. Its priority puts it behind Synchronize.
    item_fx.lum_berry(s, dex, b, side, slot, False)


@wp.func
def apply_drops(s: State, dex: Dex, b: int, side: int, word: int, announce: bool):
    """Boosts aimed at the foe, minus whatever its ability refuses to lose. The
    refusal is announced, naming the ability, unless the drop is a move's
    secondary effect."""
    slot = int(s.active[b, side])
    ability = int(s.ability[b, side, slot])
    boosts = int(s.boosts[b, side])
    refused = bool(False)
    for stat in range(7):
        delta = mv.get_boost(word, stat)
        if delta == 0:
            continue
        if delta < 0 and ability_fx.blocks_drop(dex, ability, stat):
            refused = True
            continue
        boosts = mv.set_boost(boosts, stat, mv.get_boost(boosts, stat) + delta)
    s.boosts[b, side] = wp.uint32(boosts)
    if refused and announce:
        mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)


@wp.func
def contact_response(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int,
                     side: int, att: int, foe: int, dfn: int):
    """What the defender's ability does to whoever just touched it."""
    ability = int(s.ability[b, foe, dfn])
    if ability == 0:
        return
    family = dex.ability_family[ability]
    if family == ids.ABILITYFAM_CONTACT_RECOIL:
        loss = ability_fx.contact_recoil(dex, ability, int(s.maxhp[b, side, att]))
        s.hp[b, side, att] = wp.uint16(wp.max(int(s.hp[b, side, att]) - loss, 0))
        mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)  # `-damage ... [from] ability: Rough Skin|[of]`
    elif family == ids.ABILITYFAM_CONTACT_PUNISH:
        if rng_.random_chance(rng_.draw(s, log, b), dex.ability_p0[ability], dex.ability_p1[ability]):
            effect = dex.ability_p2[ability]
            if effect == ability_fx.SPORE:
                # Effect Spore picks among sleep, paralysis and poison with a
                # second draw, whether or not the status can land.
                pick = rng_.random_n(rng_.draw(s, log, b), 3)
                effect = ids.COND_SLP
                if pick == 1:
                    effect = ids.COND_PAR
                elif pick == 2:
                    effect = ids.COND_PSN
            if effect == ids.COND_ATTRACT:
                effect = 0
                # Attract needs one male and one female; anything else fails.
                att_gender = int(s.gender[b, side, att])
                holder_gender = int(s.gender[b, foe, dfn])
                opposite = (att_gender == ids.GENDER_M and holder_gender == ids.GENDER_F) or \
                           (att_gender == ids.GENDER_F and holder_gender == ids.GENDER_M)
                if opposite and int(s.hp[b, side, att]) > 0 and \
                        not ability_fx.blocks_status(dex, int(s.ability[b, side, att]), ids.COND_ATTRACT):
                    # Oblivious refuses it like any other condition an ability
                    # will not take.
                    s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | mask_.VF_ATTRACT)
                    mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)
            if effect > 0 and status_lands_on(s, dex, b, side, att, effect, True):
                # `-status ... [from] ability: Static|[of]` names the holder,
                # except for sleep: gen 3's sleep announces itself bare.
                if effect != ids.COND_SLP:
                    mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)
                inflict_status(s, dex, log, b, side, att, effect, True)


@wp.func
def run_update(s: State, dex: Dex, b: int):
    """The Update event: where a Lum Berry cures the status that just landed.

    Every ability that refuses a status also carries an `onUpdate` that throws
    one off, which matters when the ability arrives after the status does: a
    Ditto paralysed while transformed is cured by its own Limber the moment it
    changes back.
    """
    for side in range(2):
        slot = int(s.active[b, side])
        ability = int(s.ability[b, side, slot])
        status = int(s.status[b, side, slot])
        refused = status
        if status == ids.COND_TOX:
            refused = ids.COND_PSN  # Immunity names the plain one and cures both
        if status != 0 and int(s.hp[b, side, slot]) > 0 and \
                ability_fx.blocks_status(dex, ability, refused):
            mon.cure_status(s, b, side, slot)
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)  # `-activate|...|ability: Limber`
        if int(s.confusion_turns[b, side]) > 0 and \
                ability_fx.blocks_status(dex, ability, ids.COND_CONFUSION):
            s.confusion_turns[b, side] = wp.uint8(0)
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
        if (int(s.vflags[b, side]) & mask_.VF_ATTRACT) != 0 and \
                ability_fx.blocks_status(dex, ability, ids.COND_ATTRACT):
            s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_ATTRACT)
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
        if item_fx.lum_berry(s, dex, b, side, int(s.active[b, side]),
                             int(s.confusion_turns[b, side]) > 0):
            s.confusion_turns[b, side] = wp.uint8(0)


@wp.func
def close_action(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int):
    """How gen 3 ends an action: a sort of the actives, then Update."""
    ord_.tie_draw(s, log, b, ord_.both_actives_alive(s, b))
    run_update(s, dex, b)
