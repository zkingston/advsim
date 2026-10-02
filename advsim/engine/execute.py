"""One move, from the moment it is used: Showdown's tryMoveHit and moveHit for a
single target, with every family's effect at its place in them."""
import warp as wp

from advsim.engine import damage as dmg
from advsim.engine import mask as mask_
from advsim.engine import moves as mv
from advsim.engine import order as ord_
from advsim.engine import rng as rng_
from advsim.engine import mon
from advsim.engine.effects import ability_fx
from advsim.engine.effects import item_fx
from advsim.engine.effects import move_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import move_effects
from advsim.engine import weather as weather_
from advsim.engine import status as status_
from advsim.engine import slots as slots_

WEATHER_TURNS = wp.constant(5)


@wp.func
def color_change(s: State, dex: Dex, b: int, side: int, slot: int, move_type: int):
    """AfterMoveSecondary: a Color Change holder still standing takes the type
    it was just hit by, both slots; a typeless hit leaves it alone."""
    if move_type == 0 or int(s.hp[b, side, slot]) <= 0:
        return
    if not ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_COLORCHANGE):
        return
    if int(s.types[b, side, 0]) == move_type or int(s.types[b, side, 1]) == move_type:
        return
    s.types[b, side, 0] = wp.uint8(move_type)
    s.types[b, side, 1] = wp.uint8(move_type)
    mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)  # `-start ... typechange`


@wp.func
def one_hit(s: State, dex: Dex, b: int, side: int, move: int, move_type: int, category: int,
            chasing: bool, crit_raw: wp.uint32, roll_raw: wp.uint32) -> int:
    """One hit of a damaging move, from the state as it stands: the stat and
    power modifiers, then the crit and roll the two draws decide. Showdown
    calls getDamage afresh for every hit, so a multi-hit move comes back here
    each time. L1 calls it too, which is how the harness reaches the whole
    modifier chain rather than a copy of it."""
    foe = 1 - side
    att = int(s.active[b, side])
    dfn = int(s.active[b, foe])
    atk_index = 0
    def_index = 1
    if category != 0:
        atk_index = 2
        def_index = 3
    atk_species = int(s.species[b, side, att])
    def_species = int(s.species[b, foe, dfn])
    att_ability = int(s.ability[b, side, att])
    def_ability = int(s.ability[b, foe, dfn])
    att_statused = int(s.status[b, side, att]) != 0
    def_statused = int(s.status[b, foe, dfn]) != 0
    # Handlers chain in Showdown's order: conditions, then ability, then item.
    atk_mod = dmg.chain(ability_fx.stat_modifier(dex, att_ability, atk_index, att_statused, ability_fx.MODE_CHAIN, def_ability),
                        item_fx.stat_modifier(dex, int(s.item[b, side, att]), dex.species_num[atk_species],
                                              atk_index, move_type), 4096)
    def_mod = dmg.chain(ability_fx.stat_modifier(dex, def_ability, def_index, def_statused, ability_fx.MODE_CHAIN, att_ability),
                        item_fx.stat_modifier(dex, int(s.item[b, foe, dfn]), dex.species_num[def_species],
                                              def_index, move_type), 4096)
    atk_imm = ability_fx.stat_modifier(dex, att_ability, atk_index, att_statused, ability_fx.MODE_IMMEDIATE, def_ability)
    def_imm = ability_fx.stat_modifier(dex, def_ability, def_index, def_statused, ability_fx.MODE_IMMEDIATE, att_ability)
    base = dex.move_power[move]
    if chasing:
        base = base * 2
    if dex.move_family[move] == ids.MOVEFAM_HP_SCALED_POWER:
        base = move_fx.hp_scaled_power(int(s.hp[b, side, att]), int(s.maxhp[b, side, att]))
    elif dex.move_family[move] == ids.MOVEFAM_STATUS_BOOSTED_POWER:
        base = dmg.apply_mod(base, move_fx.status_boosted_power(int(s.status[b, side, att])))
    elif dex.move_family[move] == ids.MOVEFAM_CHARGE_TURN:
        base = move_fx.solar_beam_weather_power(weather_.effective_weather(s, dex, b), base)
    power = dmg.apply_mod(base, dmg.chain(
        ability_fx.base_power_modifier(
            dex, att_ability, move_type, int(s.hp[b, side, att]), int(s.maxhp[b, side, att])),
        ability_fx.weakens_type(dex, def_ability, move_type, ids.TYPE_FIRE, ids.TYPE_ICE), 4096))
    phase1 = dmg.NEUTRAL
    if (int(s.vflags[b, side]) & mask_.VF_FLASHFIRE) != 0 and move_type == ids.TYPE_FIRE:
        phase1 = dmg.chain(dmg.NEUTRAL, 3, 2)  # Flash Fire, once it has caught
    crit_stage = dex.move_crit_stage[move] + item_fx.crit_stages(
        dex, int(s.item[b, side, att]), dex.species_num[atk_species])
    crit_blocked = ability_fx.blocks_crit(dex, def_ability)
    return mv.hit_damage(
        dex, move, move_type, category, int(s.level[b, side, att]), power,
        int(s.stats[b, side, att, atk_index]), int(s.stats[b, foe, dfn, def_index]),
        mv.get_boost(int(s.boosts[b, side]), atk_index), mv.get_boost(int(s.boosts[b, foe]), def_index),
        atk_imm, atk_mod, def_imm, def_mod, crit_stage, crit_blocked,
        ability_fx.ignores_burn_drop(att_ability),
        int(s.types[b, side, 0]), int(s.types[b, side, 1]),
        int(s.types[b, foe, 0]), int(s.types[b, foe, 1]),
        int(s.status[b, side, att]) == ids.COND_BRN, weather_.effective_weather(s, dex, b), phase1,
        crit_raw, roll_raw, ids.TYPE_FIRE, ids.TYPE_WATER)


@wp.func
def execute_move(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int, move: int) -> int:
    """One known move, through the generic path. Returns 1 when Showdown runs an
    Update event inside the hit: it dealt damage, hit a Substitute, or set weather.

    Nothing here reads the action that chose the move: Sleep Talk calls this
    with a move the user never selected, and it must behave the same way.
    """
    foe = 1 - side
    att = int(s.active[b, side])
    dfn = int(s.active[b, foe])
    move_type = mv.type_of(dex, move, int(s.hp_type[b, side, att]))
    family = dex.move_family[move]
    # Pursuit landing on a Pokemon that is leaving: doubled, and unmissable.
    chasing = family == ids.MOVEFAM_PURSUIT and (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_CHASING) != 0
    if (dex.move_flags[move] & ids.FLAG_DEFROST) != 0 and int(s.status[b, side, att]) == ids.COND_FRZ:
        # onModifyMove thaws the user before the move goes out.
        mon.cure_status(s, b, side, att)

    if family == ids.MOVEFAM_PROTECT:
        # It fails outright if nothing else is left to act, and otherwise rolls
        # against a counter that doubles for each consecutive use.
        if (int(s.turn_flags[b, 1 - side]) & mask_.TURN_FLAG_MOVED) != 0:
            return 0
        denominator = move_fx.stall_denominator(int(s.stall_ctr[b, side]))
        if denominator > 0:
            if not rng_.random_chance(rng_.draw(s, log, b), 1, denominator):
                # A failed roll fails the move but leaves the stall counter
                # standing, so it is still a residual handler this turn. The
                # end of the turn is what clears it, since Protect did not land.
                return 0
        flag = mask_.VF_PROTECT
        if dex.move_p0[move] == 1:
            flag = mask_.VF_ENDURE
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | flag)
        s.stall_ctr[b, side] = wp.uint8(wp.min(wp.max(int(s.stall_ctr[b, side]) * 2, 2), 8))
        s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) | mask_.TURN_FLAG_STALLED)
        return 0

    if family == ids.MOVEFAM_REFLECT_DAMAGE:
        taken = int(s.dmg_taken[b, side])
        marker = int(s.dmg_cat[b, side])
        answers = marker == 2
        if dex.move_p0[move] == 0:  # Counter answers physical hits
            answers = marker == 1 or marker == 3
        if taken == 0 or not answers:
            return 0

    if family == ids.MOVEFAM_FOCUSPUNCH:
        # onTry, ahead of the accuracy roll: anything that hit for damage this
        # turn breaks the focus, and a Substitute taking the hit does not.
        if int(s.dmg_taken[b, side]) != 0:
            return 0

    if family == ids.MOVEFAM_CHARGE_TURN:
        # Solar Beam charges unless the sun is out; the charge is spent here.
        if (int(s.vflags[b, side]) & mask_.VF_TWOTURN) == 0:
            if weather_.effective_weather(s, dex, b) != ids.WEATHER_SUN:
                s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | mask_.VF_TWOTURN)
                s.twoturn_move[b, side] = wp.uint8(move)
                s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) | mask_.TURN_FLAG_CHARGE_SET)
                return 0
        # The twoturnmove volatile lasts two residual phases, so it is still
        # here on the turn the beam fires and the end of the turn is what takes
        # it away. Only the move-specific volatile goes now, and the engine has
        # no separate bit for that.

    # Explosion and Self-Destruct faint the user as the move starts, before the
    # accuracy roll, so the user is gone even when the move misses.
    if dex.move_selfdestruct[move] != 0:
        s.hp[b, side, att] = wp.uint16(0)
        # Gen 3 queues the user's faint before the move hits, so it is the one
        # faintMessages takes first.
        s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) | mask_.TURN_FLAG_FAINT_FIRST)

    # Immunity is checked before the accuracy roll but does not skip it: Showdown
    # records the immunity, rolls anyway, and only then reports it.
    dealt = 0
    immune = False
    field_move = mv.targets_the_field(dex.move_target[move])
    if dex.move_ignore_immunity[move] == 0 and not field_move:
        immune = mv.type_multiplier_of(dex, move_type, int(s.types[b, foe, 0]), int(s.types[b, foe, 1])) == -99
        if ability_fx.absorbs_type(dex, int(s.ability[b, foe, dfn]), move_type, move) == 1:
            immune = True  # Levitate blocks the move outright, like a type immunity
    accuracy = dex.move_accuracy[move]
    if dex.move_family[move] == ids.MOVEFAM_WEATHER_ACCURACY:
        accuracy = move_fx.weather_accuracy(weather_.effective_weather(s, dex, b), accuracy)
    if chasing:
        accuracy = mv.NEVER_MISS  # onModifyMove: a Pokemon on its way out cannot be missed
    if accuracy != mv.NEVER_MISS and not field_move:
        accuracy = dmg.apply_mod(accuracy, ability_fx.accuracy_modifier(
            dex, int(s.ability[b, side, att]), int(s.ability[b, foe, dfn]), move_type,
            weather_.effective_weather(s, dex, b), ids.WEATHER_SAND))
        if not mv.accuracy_check(rng_.draw(s, log, b), accuracy):
            return 0
    protected = (int(s.vflags[b, foe]) & mask_.VF_PROTECT) != 0 and (dex.move_flags[move] & ids.FLAG_PROTECT) != 0
    if immune:
        # Wonder Guard's TryHit comes before the immunity is reported, so a
        # move its holder's type ignores still names the ability.
        if not protected and dex.move_category[move] != 2 and move_type != 0 and \
                ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_WONDERGUARD):
            mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)
        return 0
    if protected:
        return 0
    def_ability_now = int(s.ability[b, foe, dfn])
    # Soundproof is an onTryHit on whoever the move is aimed at, so it only
    # answers for a move aimed at its holder: Roar here, never Heal Bell over
    # the other side's team. Perish Song reaches everyone and is refused one
    # Pokemon at a time, in its own branch.
    if ability_fx.has(dex, def_ability_now, ids.ABILITYFAM_SOUNDPROOF) and \
            (dex.move_flags[move] & ids.FLAG_SOUND) != 0 and \
            dex.move_target[move] != mv.TARGET_ALL and \
            dex.move_target[move] != mv.TARGET_ALLY_TEAM and \
            dex.move_target[move] != mv.TARGET_SELF:
        mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)  # `-immune ... [from] ability: Soundproof`
        return 0
    if ability_fx.has(dex, def_ability_now, ids.ABILITYFAM_WONDERGUARD) and \
            dex.move_category[move] != 2 and move_type != 0:
        # Only a super-effective hit gets through, and a typeless one is waved
        # past rather than judged.
        if mv.type_multiplier_of(dex, move_type, int(s.types[b, foe, 0]), int(s.types[b, foe, 1])) <= 0:
            mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)
            return 0
    absorb = 0
    if not field_move:
        absorb = ability_fx.absorbs_type(dex, int(s.ability[b, foe, dfn]), move_type, move)
    if absorb == ability_fx.ABSORB_FLASH_FIRE + 1 and int(s.status[b, foe, dfn]) == ids.COND_FRZ:
        absorb = 0  # a frozen holder takes the hit, which is what thaws it
    if absorb == ability_fx.ABSORB_FLASH_FIRE + 1 and move == ids.MOVE_WILLOWISP and \
            (int(s.types[b, foe, 0]) == ids.TYPE_FIRE or int(s.types[b, foe, 1]) == ids.TYPE_FIRE or
             int(s.status[b, foe, dfn]) != 0 or int(s.sub_hp[b, foe]) > 0):
        # Gen 3 lets Will-O-Wisp past Flash Fire when the burn could not have
        # landed anyway, rather than feeding it. It then fails on its own terms.
        absorb = 0
    if absorb != 0:
        # A heal, a Flash Fire starting, or an `-immune` for either.
        mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)
    if absorb == ability_fx.ABSORB_HEAL + 1:
        maxhp = int(s.maxhp[b, foe, dfn])
        s.hp[b, foe, dfn] = wp.uint16(wp.min(int(s.hp[b, foe, dfn]) + wp.max(maxhp / 4, 1), maxhp))
        return 0
    if absorb == ability_fx.ABSORB_FLASH_FIRE + 1:
        s.vflags[b, foe] = wp.uint32(int(s.vflags[b, foe]) | mask_.VF_FLASHFIRE)
        return 0

    category = dex.move_category[move]
    if dex.move_type_from_mon[move] != 0:
        # Gen 3 reads the category off the type, so Hidden Power's category
        # follows whatever type the holder gives it.
        category = dex.type_is_special[move_type]

    # A Substitute stands in front of its holder. Showdown asks the damage
    # formula for a number first: a status move has no power, so it returns
    # nothing and the move fails outright, while a damaging move is rolled
    # exactly as it would be and spends what it dealt on the Substitute.
    # A non-Ghost Curse rewrites itself into a self-boost, target included, and
    # the build proves no Curse learner in this vocabulary is a Ghost.
    aims_at_self = dex.move_target[move] == mv.TARGET_SELF or \
        dex.move_family[move] == ids.MOVEFAM_CURSE

    sub_blocks = False
    if int(s.sub_hp[b, foe]) > 0 and not field_move and \
            not aims_at_self and (dex.move_flags[move] & ids.FLAG_BYPASSSUB) == 0:
        sub_blocks = True
        if category == 2:
            return 0
    sub_ate = int(0)  # 1 while the Substitute has swallowed every hit
    if sub_blocks:
        sub_ate = 1
    # Recoil is worked out with the damage and taken at the end of the hit:
    # Showdown applies it after the secondaries, the contact punishers and the
    # faint the hit caused, so a Pokemon it kills is still standing for those.
    recoil_loss = int(0)

    if category != 2:  # a damaging move
        # Showdown checks immunity, then fixed damage, and only then draws for a
        # crit. Both early exits skip the crit and roll draws, so the draws have
        # to happen here rather than in the arguments of the damage call.
        att_ability = int(s.ability[b, side, att])
        def_ability = int(s.ability[b, foe, dfn])
        # Rock Head refuses recoil, and gen 3 spells out Struggle as the one it
        # does not refuse: a Pokemon out of PP takes the quarter either way.
        rockhead = ability_fx.has(dex, att_ability, ids.ABILITYFAM_ROCKHEAD) and \
            move != ids.MOVE_STRUGGLE
        fixed = dex.move_fixed_damage[move]
        damage = 0
        if family == ids.MOVEFAM_REFLECT_DAMAGE:
            damage = 2 * int(s.dmg_taken[b, side])
        elif fixed != 0:
            damage = int(s.level[b, side, att])
            if fixed != mv.FIXED_LEVEL:
                damage = fixed
        else:
            # A multi-hit move rolls crit and damage per hit, and Showdown runs
            # an Update event between hits.
            hits = 1
            if dex.move_multihit_min[move] != 0:
                hits = dex.move_multihit_min[move]
            total = int(0)
            for hit in range(hits):
                # Showdown's multi-hit loop stops when either side drops. A
                # single-hit move never enters that loop, so the guard and the
                # Update that closes each pass apply only to a real multi-hit.
                if hit > 0 and (int(s.hp[b, foe, dfn]) <= 0 or int(s.hp[b, side, att]) <= 0):
                    continue
                crit_raw = rng_.draw(s, log, b)
                roll_raw = rng_.draw(s, log, b)
                one = one_hit(s, dex, b, side, move, move_type, category, chasing, crit_raw, roll_raw)
                left = int(s.sub_hp[b, foe])
                if sub_blocks and left > 0:
                    # The Substitute takes it. Recoil and drain still run, from
                    # inside the Substitute's own handler, where the drain
                    # rounds up rather than down and is not clamped to one.
                    one = wp.min(one, left)
                    s.sub_hp[b, foe] = wp.uint16(left - one)
                    if one > 0 and dex.move_recoil_num[move] != 0 and not rockhead:
                        loss = wp.max(one * dex.move_recoil_num[move] / dex.move_recoil_den[move], 1)
                        s.hp[b, side, att] = wp.uint16(wp.max(int(s.hp[b, side, att]) - loss, 0))
                    if one > 0 and dex.move_drain_num[move] != 0:
                        den = dex.move_drain_den[move]
                        gain = (one * dex.move_drain_num[move] + den - 1) / den
                        cap = int(s.maxhp[b, side, att])
                        s.hp[b, side, att] = wp.uint16(wp.min(int(s.hp[b, side, att]) + gain, cap))
                else:
                    sub_ate = 0
                    hp = int(s.hp[b, foe, dfn])
                    if (int(s.vflags[b, foe]) & mask_.VF_ENDURE) != 0 and one >= hp:
                        one = hp - 1
                    one = wp.min(one, hp)
                    s.hp[b, foe, dfn] = wp.uint16(hp - one)
                    total += one
                    if dex.move_multihit_min[move] != 0 and one > 0:
                        # Gen 3 runs a multi-hit move's AfterMoveSecondary per
                        # hit, so Color Change turns between hits and the next
                        # hit meets the new type.
                        color_change(s, dex, b, foe, dfn, move_type)
                if dex.move_multihit_min[move] != 0:
                    ord_.tie_draw(s, log, b, True)
            damage = total
        if damage > 0 and dex.move_multihit_min[move] == 0 and \
                (dex.move_fixed_damage[move] != 0 or family == ids.MOVEFAM_REFLECT_DAMAGE):
            left = int(s.sub_hp[b, foe])
            if sub_blocks and left > 0:
                damage = wp.min(damage, left)
                s.sub_hp[b, foe] = wp.uint16(left - damage)
            else:
                sub_ate = 0
                hp = int(s.hp[b, foe, dfn])
                if (int(s.vflags[b, foe]) & mask_.VF_ENDURE) != 0 and damage >= hp:
                    damage = hp - 1  # Endure answers any move damage, fixed included
                damage = wp.min(damage, hp)
                s.hp[b, foe, dfn] = wp.uint16(hp - damage)
        if damage > 0 and sub_ate == 0:
            # The Damage event never fires for a hit a Substitute took, so
            # Counter and Mirror Coat never see it.
            s.dmg_taken[b, foe] = wp.uint16(damage)
            # 1 physical, 2 special, 3 Hidden Power. Gen 3 lets Counter answer
            # Hidden Power whatever its category, and Mirror Coat never can.
            marker = category + 1
            if dex.move_type_from_mon[move] != 0:
                marker = 3
            s.dmg_cat[b, foe] = wp.uint8(marker)
        # A hit a Substitute took still reports a number, zero, so the Update
        # inside tryMoveHit runs exactly as it does for a hit that connected.
        dealt = 1
        if damage > 0 and sub_ate == 0 and dex.move_recoil_num[move] != 0 and not rockhead:
            # Recoil is taken from the whole total, floored, never below one.
            recoil_loss = wp.max(damage * dex.move_recoil_num[move] / dex.move_recoil_den[move], 1)
        if damage > 0 and sub_ate == 0 and dex.move_drain_num[move] != 0:
            gain = wp.max(damage * dex.move_drain_num[move] / dex.move_drain_den[move], 1)
            if ability_fx.has(dex, def_ability, ids.ABILITYFAM_LIQUIDOOZE):
                # The drink is poison: the drainer takes what it would have had.
                s.hp[b, side, att] = wp.uint16(wp.max(int(s.hp[b, side, att]) - gain, 0))
                mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)  # `-damage ... [from] ability: Liquid Ooze|[of]`
            else:
                maxhp = int(s.maxhp[b, side, att])
                s.hp[b, side, att] = wp.uint16(wp.min(int(s.hp[b, side, att]) + gain, maxhp))

    # Primary effects. A move's boosts land on whoever it targets: Calm Mind
    # targets self, Charm targets the foe, and the table says which.
    if category == 2:
        status = dex.move_status[move]
        if int(s.hp[b, foe, dfn]) > 0 and status_.status_lands_on(s, dex, b, foe, dfn, status, True):
            status_.inflict_status(s, dex, log, b, foe, dfn, status, True)
        elif int(s.hp[b, foe, dfn]) > 0 and status_.ability_refuses(s, dex, b, foe, dfn, status):
            mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)  # `-immune ... [from] ability: Insomnia`
    boosts = dex.move_boosts[move]
    if boosts != 0:
        if dex.move_target[move] == mv.TARGET_SELF:
            s.boosts[b, side] = wp.uint32(mv.apply_boost_word(int(s.boosts[b, side]), boosts))
        elif sub_ate == 0 and int(s.hp[b, foe, dfn]) > 0:
            status_.apply_drops(s, dex, b, foe, boosts, True)
    # Setting weather counts as an effect that reaches the end of the hit, so it
    # runs the Update event that a status or a boost does not.
    move_effects.primary_effect(s, dex, log, b, side, move)


    heal_num = dex.move_heal_num[move]
    if heal_num != 0:
        maxhp = int(s.maxhp[b, side, att])
        hp = int(s.hp[b, side, att])
        if hp > 0 and hp < maxhp:
            # Healing yourself does not run the Update event inside the hit, the
            # same way a boost or a status does not.
            s.hp[b, side, att] = wp.uint16(wp.min(hp + maxhp * heal_num / dex.move_heal_den[move], maxhp))

    if dex.move_volatile[move] == ids.COND_PARTIALLYTRAPPED and sub_ate == 0 and \
            int(s.hp[b, foe, dfn]) > 0 and int(s.trap_turns[b, foe]) == 0:
        # A Substitute keeps the target out of runMoveEffects entirely, so the
        # trap never reaches it. Already trapped means addVolatile returns
        # before the length is rolled, which is the other way to spend nothing.
        s.trap_turns[b, foe] = wp.uint8(rng_.random_range(rng_.draw(s, log, b), 3, 7))
        s.trap_source[b, foe] = wp.uint8(att + 1)

    if dex.move_volatile[move] == ids.COND_LEECHSEED:
        if int(s.types[b, foe, 0]) != ids.TYPE_GRASS and int(s.types[b, foe, 1]) != ids.TYPE_GRASS:
            s.vflags[b, foe] = wp.uint32(int(s.vflags[b, foe]) | mask_.VF_LEECHSEED)

    if dex.move_switch_mode[move] == 1 and int(s.hp[b, foe, dfn]) > 0 and \
            int(s.hp[b, side, att]) > 0 and slots_.has_bench(s, b, foe) and \
            ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_SUCTIONCUPS):
        mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)  # `-activate|...|ability: Suction Cups`
    if dex.move_switch_mode[move] == 1 and int(s.hp[b, foe, dfn]) > 0 and \
            int(s.hp[b, side, att]) > 0 and slots_.has_bench(s, b, foe) and \
            not ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_SUCTIONCUPS):
        # forceSwitch only raises a flag; runAction drags the Pokemon out once
        # the move is over, and the replacement's own runSwitch is queued after.
        s.turn_flags[b, foe] = wp.uint16(int(s.turn_flags[b, foe]) | mask_.TURN_FLAG_DRAGGED)

    weather = dex.move_weather[move]
    if weather != 0 and int(s.weather[b]) != weather:
        s.weather[b] = wp.uint8(weather)
        s.weather_turns[b] = wp.uint8(WEATHER_TURNS)
        dealt = 1
        for k in range(2):
            weather_.forecast(s, dex, b, k)

    self_boosts = dex.move_self_boosts[move]
    if self_boosts != 0:
        # `selfDrops` rolls for a chance the move does not carry, so Overheat and
        # Superpower each spend a draw here and then always take the drop. The
        # user's own ability never refuses it, which is why this is not
        # `apply_drops`. `boost` refuses a Pokemon at 0 HP, though, so one that a
        # Rough Skin just knocked out takes no drop and its White Herb stays.
        rng_.draw(s, log, b)
        if int(s.hp[b, side, att]) > 0:
            s.boosts[b, side] = wp.uint32(mv.apply_boost_word(int(s.boosts[b, side]), self_boosts))

    chance = dex.move_sec_chance[move]
    # Read here, not from the damage branch's copy: a status move never sets it.
    if chance != 0 and ability_fx.has(dex, int(s.ability[b, side, att]), ids.ABILITYFAM_SERENEGRACE):
        chance = chance * 2  # onModifyMove, before anything is rolled
    if chance != 0 and sub_ate == 0 and dex.move_sec_self_boosts[move] == 0 and \
            ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_SHIELDDUST):
        # ModifySecondaries empties the list, so the loop that would roll never
        # runs: Shield Dust costs the draw as well as the effect. It is read off
        # the target, though, and a Substitute leaves no target to read, so the
        # roll comes back the moment one is up. It also keeps any secondary that
        # boosts the user rather than touching the holder.
        chance = 0
    if chance != 0:
        # The roll happens even when a Substitute took the hit; only what it
        # would have done to the Pokemon behind is skipped.
        fired = rng_.random_n(rng_.draw(s, log, b), 100) < chance
        if fired and sub_ate == 0 and int(s.hp[b, foe, dfn]) > 0:
            sec_status = dex.move_sec_status[move]
            if status_.status_lands_on(s, dex, b, foe, dfn, sec_status, True):
                status_.inflict_status(s, dex, log, b, foe, dfn, sec_status, True)
            sec_boosts = dex.move_sec_boosts[move]
            if sec_boosts != 0:
                status_.apply_drops(s, dex, b, foe, sec_boosts, False)
            if dex.move_sec_volatile[move] == ids.COND_CONFUSION and \
                    int(s.confusion_turns[b, foe]) == 0 and \
                    not ability_fx.blocks_status(dex, int(s.ability[b, foe, dfn]), ids.COND_CONFUSION):
                # Own Tempo refuses it at TryAddVolatile, before the length is
                # rolled, so a refusal costs nothing.
                s.confusion_turns[b, foe] = wp.uint8(rng_.random_range(rng_.draw(s, log, b), 2, 6))
            if dex.move_sec_volatile[move] == ids.COND_FLINCH:
                # A flinch only stops a Pokemon that has not moved yet, but the
                # volatile is set either way and lives until the residual phase.
                if not ability_fx.blocks_status(dex, int(s.ability[b, foe, dfn]), ids.COND_FLINCH) and \
                        (int(s.turn_flags[b, foe]) & mask_.TURN_FLAG_FOCUS) == 0:
                    # A Pokemon waiting on Focus Punch refuses the volatile.
                    s.vflags[b, foe] = wp.uint32(int(s.vflags[b, foe]) | mask_.VF_FLINCH)
        if fired and dex.move_sec_self_boosts[move] != 0 and int(s.hp[b, side, att]) > 0:
            # A secondary's own boost reaches the user through `selfDrops`,
            # which asks nothing about the target: it lands even when a
            # Substitute took the hit, and even when the hit was fatal. Only
            # the user has to be standing.
            s.boosts[b, side] = wp.uint32(mv.apply_boost_word(int(s.boosts[b, side]),
                                                              dex.move_sec_self_boosts[move]))

    if dealt == 1 and (dex.move_flags[move] & ids.FLAG_RECHARGE) != 0:
        # `self.volatileStatus` on the move, applied through selfDrops, which
        # draws nothing when the self block carries no boosts.
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | mask_.VF_MUSTRECHARGE)
        s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) | mask_.TURN_FLAG_RECHARGE_SET)
    if dealt == 1 and sub_ate == 0 and family == ids.MOVEFAM_KNOCKOFF and int(s.hp[b, side, att]) > 0 and \
            int(s.item[b, foe, dfn]) != 0 and ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_STICKYHOLD):
        mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)  # takeItem: `-activate|...|ability: Sticky Hold`
    if dealt == 1 and sub_ate == 0 and family == ids.MOVEFAM_KNOCKOFF and int(s.hp[b, side, att]) > 0 and \
            not ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_STICKYHOLD):
        # AfterHit, which a Substitute keeps the target out of entirely. Only a
        # Pokemon that loses something is marked: takeItem hands back nothing
        # from an empty hand, and a later Trick reads the mark.
        if int(s.item[b, foe, dfn]) != 0:
            s.item[b, foe, dfn] = wp.uint8(0)
            mon.reveal(s, b, foe, dfn, mon.REVEAL_ITEM)  # `-enditem ... [from] move: Knock Off`
            s.knocked_mask[b, foe] = wp.uint8(int(s.knocked_mask[b, foe]) | (1 << dfn))
    if dealt == 1 and family == ids.MOVEFAM_RAPIDSPIN:
        # The spin is a self-hit, so it happens through a Substitute as well.
        # It sheds a Leech Seed, the Spikes on its own side, and a partial trap.
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_LEECHSEED)
        s.spikes[b, side] = wp.uint8(0)
        s.trap_turns[b, side] = wp.uint8(0)
        s.trap_source[b, side] = wp.uint8(0)

    if dealt == 1 and sub_ate == 0 and category != 2 and damage > 0 and dex.move_multihit_min[move] == 0:
        color_change(s, dex, b, foe, dfn, move_type)
    if dealt == 1 and sub_ate == 0 and category != 2 and dex.move_type[move] == ids.TYPE_FIRE and \
            int(s.status[b, foe, dfn]) == ids.COND_FRZ and int(s.hp[b, foe, dfn]) > 0:
        # DamagingHit, where the status handler runs ahead of the ability one.
        # It reads the move's type out of the dex rather than off the move being
        # used, so a Fire-type Hidden Power thaws nothing. cureStatus refuses a
        # Pokemon at 0 HP, so one this hit just dropped stays frozen.
        mon.cure_status(s, b, foe, dfn)

    # The contact punisher rolls after the move's own secondary, and it runs
    # before the faint is processed, so a defender that just dropped to 0 still
    # punishes whoever touched it.
    if dealt == 1 and sub_ate == 0 and category != 2 and (dex.move_flags[move] & ids.FLAG_CONTACT) != 0:
        status_.contact_response(s, dex, log, b, side, att, foe, dfn)

    if recoil_loss > 0:
        s.hp[b, side, att] = wp.uint16(wp.max(int(s.hp[b, side, att]) - recoil_loss, 0))

    # A target this move dropped went into the faint queue before anything the
    # move then did to its user: recoil, a contact punisher, Destiny Bond. An
    # Explosion's user is the exception, and has already said so.
    if int(s.hp[b, foe, dfn]) <= 0 and ((int(s.alive_mask[b, foe]) >> dfn) & 1) != 0 and \
            (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_FAINT_FIRST) == 0:
        s.turn_flags[b, foe] = wp.uint16(int(s.turn_flags[b, foe]) | mask_.TURN_FLAG_FAINT_FIRST)

    # A move that dealt damage or changed the field runs the Update event inside
    # tryMoveHit; one that merely set a status or a boost returns before it.
    return dealt
