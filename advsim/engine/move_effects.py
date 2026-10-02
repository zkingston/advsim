"""What a move does once it has hit, when that is more than damage, a status
or a boost: one branch per move family, from Haze to Transform. Runs inside
`execute_move`, after the move's boosts and before its heal and secondaries."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import moves as mv
from advsim.engine import rng as rng_
from advsim.engine import mon
from advsim.engine import slots as slots_
from advsim.engine import weather as weather_
from advsim.engine.effects import ability_fx
from advsim.engine.effects import move_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex


@wp.func
def primary_effect(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int, move: int):
    foe = 1 - side
    att = int(s.active[b, side])
    dfn = int(s.active[b, foe])
    family = dex.move_family[move]
    if family == ids.MOVEFAM_CURE_TEAM:
        move_fx.cure_team(s, dex, b, side, dex.move_p0[move])
    elif family == ids.MOVEFAM_SELF_CURE:
        counter = dex.move_p0[move]
        if counter == 0:
            if move_fx.refresh_cures(int(s.status[b, side, att])):
                mon.cure_status(s, b, side, att)
        else:
            # Rest. It refuses to run when its user is already asleep, at full
            # HP, or holds an ability that will not sleep, and it overwrites any
            # other status. The sleep still rolls its own length before Rest
            # writes a fixed count over it, so the draw happens either way.
            maxhp = int(s.maxhp[b, side, att])
            hp = int(s.hp[b, side, att])
            status = int(s.status[b, side, att])
            blocked = dex.type_status_immune[int(s.types[b, side, 0])] | \
                dex.type_status_immune[int(s.types[b, side, 1])]
            if status != ids.COND_SLP and hp < maxhp and (blocked & (1 << ids.COND_SLP)) == 0 and \
                    not ability_fx.blocks_status(dex, int(s.ability[b, side, att]), ids.COND_SLP):
                rng_.draw(s, log, b)
                s.status[b, side, att] = wp.uint8(ids.COND_SLP)
                s.status_ctr[b, side, att] = wp.uint8(counter)
                s.slept_by_foe[b, side, att] = wp.uint8(0)
                s.sleep_skipped[b, side, att] = wp.uint8(0)
                s.hp[b, side, att] = wp.uint16(maxhp)
    elif family == ids.MOVEFAM_HEAL_WEATHER:
        maxhp = int(s.maxhp[b, side, att])
        hp = int(s.hp[b, side, att])
        if hp > 0 and hp < maxhp:
            w = weather_.effective_weather(s, dex, b)
            gain = maxhp * move_fx.heal_weather_numerator(w) / move_fx.heal_weather_denominator(w)
            s.hp[b, side, att] = wp.uint16(wp.min(hp + gain, maxhp))
    elif family == ids.MOVEFAM_SUBSTITUTE:
        # It fails if one is already standing, or if a quarter of max HP would
        # not leave the user alive. That comparison is exact, not floored, and
        # the doll's HP and the damage are both the floor.
        maxhp = int(s.maxhp[b, side, att])
        hp = int(s.hp[b, side, att])
        if int(s.sub_hp[b, side]) == 0 and hp * 4 > maxhp and maxhp > 1:
            cost = wp.max(maxhp / 4, 1)
            s.sub_hp[b, side] = wp.uint16(cost)
            s.hp[b, side, att] = wp.uint16(wp.max(hp - cost, 0))
            # Putting one up shrugs off a partial trap, doll and all.
            s.trap_turns[b, side] = wp.uint8(0)
            s.trap_source[b, side] = wp.uint8(0)
    elif family == ids.MOVEFAM_SPIKES:
        layers = int(s.spikes[b, foe])
        if layers < 3:
            s.spikes[b, foe] = wp.uint8(layers + 1)
    elif family == ids.MOVEFAM_TRANSFORM:
        # Gen 3 lets a transformed Pokemon transform again, but never lets one
        # copy a Pokemon that is itself a copy. The first copy is the one whose
        # originals are kept.
        if int(s.hp[b, foe, dfn]) > 0 and (int(s.vflags[b, foe]) & mask_.VF_TRANSFORMED) == 0:
            if (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) == 0:
                s.xf_species[b, side] = s.species[b, side, att]
                s.xf_ability[b, side] = s.ability[b, side, att]
                s.xf_hp_type[b, side] = s.hp_type[b, side, att]
                for i in range(5):
                    s.xf_stats[b, side, i] = s.stats[b, side, att, i]
                for i in range(4):
                    s.xf_moves[b, side, i] = s.moves[b, side, att, i]
                    s.xf_pp[b, side, i] = s.pp[b, side, att, i]
                    s.xf_max_pp[b, side, i] = s.max_pp[b, side, att, i]
                s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | mask_.VF_TRANSFORMED)
            s.species[b, side, att] = s.species[b, foe, dfn]
            s.ability[b, side, att] = s.ability[b, foe, dfn]
            s.hp_type[b, side, att] = s.hp_type[b, foe, dfn]
            for i in range(5):
                s.stats[b, side, att, i] = s.stats[b, foe, dfn, i]
            for i in range(4):
                copied = int(s.moves[b, foe, dfn, i])
                s.moves[b, side, att, i] = wp.uint8(copied)
                s.pp[b, side, att, i] = wp.uint8(wp.min(5, dex.move_pp[copied]))
                # Showdown rebuilds the copied slot's maxpp from the copier's own
                # PP Ups, by slot index, so a Ditto's boosted first slot boosts
                # whatever it copies into that slot and the rest come out raw.
                own = int(s.xf_moves[b, side, i])
                ups = 0
                if own != 0 and dex.move_pp[own] > 0:
                    ups = int(s.xf_max_pp[b, side, i]) * 5 // dex.move_pp[own] - 5
                s.max_pp[b, side, att, i] = wp.uint8(dex.move_pp[copied] * (5 + ups) // 5)
            s.types[b, side, 0] = s.types[b, foe, 0]
            s.types[b, side, 1] = s.types[b, foe, 1]
            s.boosts[b, side] = s.boosts[b, foe]
            # setSpecies recomputes the stats from the copier's own spread over
            # the copied base stats and writes the Speed among them, and only
            # then does the copy overwrite the stats. So the sorting speed is
            # neither Pokemon's: it is the copied base at the copier's level.
            # Every set in this format runs 85 EVs and 31 IVs bar the drop a
            # Hidden Power asks for, which its own Speed gives away.
            level = int(s.level[b, side, att])
            own = dex.species_base_stats[int(s.xf_species[b, side]), 5]
            iv = 31
            if ((2 * own + 31 + 21) * level / 100) + 5 != int(s.xf_stats[b, side, 4]):
                iv = 30
            base = dex.species_base_stats[int(s.species[b, side, att]), 5]
            s.cached_spe[b, side, att] = wp.uint16(((2 * base + iv + 21) * level / 100) + 5)
    elif family == ids.MOVEFAM_TRICK:
        # A plain swap, an empty hand included: takeItem reports nothing rather
        # than a refusal for those, so it fails only when neither side is
        # holding anything. A hand Knock Off emptied is a refusal, though, and
        # in gen 3 that sticks to the Pokemon for the rest of the battle.
        mine = int(s.item[b, side, att])
        theirs = int(s.item[b, foe, dfn])
        knocked = ((int(s.knocked_mask[b, side]) >> att) & 1) != 0 or \
                  ((int(s.knocked_mask[b, foe]) >> dfn) & 1) != 0
        if ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_STICKYHOLD):
            knocked = True  # Sticky Hold refuses takeItem the same way, and in gen 3 silently
        if (mine != 0 or theirs != 0) and not knocked:
            s.item[b, side, att] = wp.uint8(theirs)
            s.item[b, foe, dfn] = wp.uint8(mine)
            # An `-item` or `-enditem` line for each side.
            mon.reveal(s, b, side, att, mon.REVEAL_ITEM)
            mon.reveal(s, b, foe, dfn, mon.REVEAL_ITEM)
    elif family == ids.MOVEFAM_BATONPASS:
        # It fails outright with nobody to pass to; otherwise the turn stops
        # here and the side is asked who comes in.
        if slots_.has_bench(s, b, side):
            s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) | mask_.TURN_FLAG_PASSING)
    elif family == ids.MOVEFAM_TRAP_FOE:
        # Mean Look and Spider Web are onHit, so a Substitute stops them and a
        # miss stops them: both are settled before this point.
        s.vflags[b, foe] = wp.uint32(int(s.vflags[b, foe]) | mask_.VF_TRAPPED)
    elif family == ids.MOVEFAM_DESTINYBOND:
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | mask_.VF_DESTINYBOND)
    elif family == ids.MOVEFAM_YAWN:
        # Landing it asks whether the target is already statused, whether its
        # type can sleep at all, and whether its ability refuses sleep, which
        # Insomnia and Vital Spirit do at TryAddVolatile. Sleep Clause is not
        # consulted until the residual phase two turns on, so a Yawn can land
        # and fail there instead.
        blocked = dex.type_status_immune[int(s.types[b, foe, 0])] | \
            dex.type_status_immune[int(s.types[b, foe, 1])]
        if int(s.yawn_turns[b, foe]) == 0 and int(s.status[b, foe, dfn]) == 0 and \
                (blocked & (1 << ids.COND_SLP)) == 0:
            if ability_fx.blocks_status(dex, int(s.ability[b, foe, dfn]), ids.COND_SLP):
                mon.reveal(s, b, foe, dfn, mon.REVEAL_ABILITY)  # `-immune ... [from] ability: Insomnia`
            else:
                s.yawn_turns[b, foe] = wp.uint8(2)
    elif family == ids.MOVEFAM_PERISHSONG:
        # It reaches both actives, the user included, and skips anyone already
        # counting down.
        for k in range(2):
            deaf = ability_fx.has(dex, int(s.ability[b, k, int(s.active[b, k])]),
                                  ids.ABILITYFAM_SOUNDPROOF)
            if deaf and k != side and int(s.hp[b, k, int(s.active[b, k])]) > 0:
                # Each active's TryHit runs in turn, so the holder says so
                # whether or not it already has a count.
                mon.reveal(s, b, k, int(s.active[b, k]), mon.REVEAL_ABILITY)
            if int(s.perish_count[b, k]) == 0 and int(s.hp[b, k, int(s.active[b, k])]) > 0 and \
                    not deaf:
                s.perish_count[b, k] = wp.uint8(4)
    elif family == ids.MOVEFAM_WISH:
        # A slot condition, not a volatile: it heals whoever stands here two
        # residual phases from now, and a second Wish cannot stack on it.
        if int(s.wish_turns[b, side]) == 0:
            s.wish_turns[b, side] = wp.uint8(2)
    elif family == ids.MOVEFAM_ENCORE:
        # A second Encore on an already-encored Pokemon returns before the
        # volatile is created, so it costs nothing. Otherwise the duration is
        # rolled as the volatile is made, which is before onStart decides
        # whether it may stay: the draw happens even when the lock then fails.
        if int(s.encore_turns[b, foe]) == 0:
            turns = rng_.random_range(rng_.draw(s, log, b), 3, 7)
            last = int(s.last_move[b, foe])
            slot = slots_.move_slot_of(s, b, foe, dfn, last)
            if last != 0 and slot >= 0 and (dex.move_flags[last] & ids.FLAG_FAILENCORE) == 0 and \
                    int(s.pp[b, foe, dfn, slot]) > 0:
                s.encore_turns[b, foe] = wp.uint8(turns)
                s.encore_move[b, foe] = wp.uint8(last)
    elif family == ids.MOVEFAM_HAZE:
        for k in range(2):
            s.boosts[b, k] = wp.uint32(0)
    elif family == ids.MOVEFAM_BELLYDRUM:
        # It fails unless more than half the user's max HP is left, and the
        # twelve stages it asks for clamp to six like any other boost.
        maxhp = int(s.maxhp[b, side, att])
        hp = int(s.hp[b, side, att])
        boosts = int(s.boosts[b, side])
        if hp * 2 > maxhp and maxhp > 1 and mv.get_boost(boosts, 0) < 6:
            s.hp[b, side, att] = wp.uint16(wp.max(hp - wp.max(maxhp / 2, 1), 0))
            s.boosts[b, side] = wp.uint32(mv.set_boost(boosts, 0, 6))
    elif family == ids.MOVEFAM_PAINSPLIT:
        # Both ends are set to the average, each capped by its own max HP.
        avg = wp.max((int(s.hp[b, side, att]) + int(s.hp[b, foe, dfn])) / 2, 1)
        s.hp[b, foe, dfn] = wp.uint16(wp.min(avg, int(s.maxhp[b, foe, dfn])))
        s.hp[b, side, att] = wp.uint16(wp.min(avg, int(s.maxhp[b, side, att])))
    elif family == ids.MOVEFAM_CURSE:
        # Only the non-Ghost half exists here: Curse becomes a self-boost that
        # arrives through `selfDrops`, which rolls before it applies.
        rng_.draw(s, log, b)
        boosts = int(s.boosts[b, side])
        boosts = mv.set_boost(boosts, 0, mv.get_boost(boosts, 0) + 1)
        boosts = mv.set_boost(boosts, 1, mv.get_boost(boosts, 1) + 1)
        boosts = mv.set_boost(boosts, 4, mv.get_boost(boosts, 4) - 1)
        s.boosts[b, side] = wp.uint32(boosts)
