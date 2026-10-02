"""Switching in: the slot changes hands, then runSwitch brings the hazards and
the switch-in abilities."""
import warp as wp

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
from advsim.engine import status as status_
from advsim.engine import slots as slots_


@wp.func
def grounded(s: State, dex: Dex, b: int, side: int, slot: int) -> bool:
    """Spikes only reach a Pokemon standing on the ground: not a Flying type,
    not a Levitate holder. Gen 3 has nothing else that lifts one."""
    if int(s.types[b, side, 0]) == ids.TYPE_FLYING or int(s.types[b, side, 1]) == ids.TYPE_FLYING:
        return False
    ability = int(s.ability[b, side, slot])
    if ability != 0 and dex.ability_family[ability] == ids.ABILITYFAM_TYPE_IMMUNE and \
            dex.ability_p0[ability] == ids.TYPE_GROUND:
        return False
    return True


@wp.func
def switch_in(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int,
              slot: int, pass_on: bool):
    """Bring in a party member: the slot block clears and types come from the
    species. A Baton Pass keeps what `keep_slot` says it keeps."""
    # Showdown swaps the arrival into slot zero of its own party array and the
    # Pokemon leaving into the arrival's old place. Nothing but a drag can see
    # that order, and a drag samples over it.
    leaving = int(s.active[b, side])
    held = int(s.ability[b, side, leaving])
    if ability_fx.suppresses_weather(dex, held) and int(s.hp[b, side, leaving]) > 0:
        # The End event on the way out: Air Lock and Cloud Nine announce a
        # weather change as they go, which sorts the actives. The ability is
        # already gone by the time anything answers, which is what lets a
        # Castform on the other side change forme.
        s.ability[b, side, leaving] = wp.uint8(0)
        ord_.tie_draw(s, log, b, ord_.both_actives_alive(s, b))
        for k in range(2):
            weather_.forecast(s, dex, b, k)
        s.ability[b, side, leaving] = wp.uint8(held)
    # SwitchOut runs before clearVolatile, so it reads whatever ability the
    # Pokemon is holding right now: a traced Natural Cure cures, and a Natural
    # Cure holder that traced something else does not.
    if ability_fx.has(dex, int(s.ability[b, side, leaving]), ids.ABILITYFAM_NATURALCURE) and \
            int(s.status[b, side, leaving]) != 0:
        # SwitchOut, which fires for a drag as well as a choice.
        mon.cure_status(s, b, side, leaving)
        mon.reveal(s, b, side, leaving, mon.REVEAL_ABILITY)  # `-curestatus ... [from] ability: Natural Cure`
    slots_.revert_transform(s, b, side, leaving)
    if int(s.base_ability[b, side]) != 0:
        # clearVolatile puts the base ability back, so a Trace holder traces
        # again every time it comes in.
        s.ability[b, side, leaving] = wp.uint8(int(s.base_ability[b, side]) - 1)
        s.base_ability[b, side] = wp.uint8(0)
    # clearVolatile ends with setSpecies, which writes the cached Speed back to
    # the Pokemon's own stat: what it was sorting at on the field does not
    # follow it to the bench.
    s.cached_spe[b, side, leaving] = s.stats[b, side, leaving, 4]
    here = int(s.party_pos[b, side, slot])
    s.party_pos[b, side, slot] = s.party_pos[b, side, leaving]
    s.party_pos[b, side, leaving] = wp.uint8(here)
    s.active[b, side] = wp.uint8(slot)
    mon.reveal(s, b, side, slot, mon.REVEAL_SEEN)  # the switch line
    if pass_on:
        slots_.keep_slot(s, b, side)
    else:
        slots_.clear_slot(s, b, side)
    species = int(s.species[b, side, slot])
    s.types[b, side, 0] = wp.uint8(dex.species_type1[species])
    s.types[b, side, 1] = wp.uint8(dex.species_type2[species])
    s.turn_flags[b, side] = wp.uint16(mask_.TURN_FLAG_SWITCHED_IN)
    # Queueing the runSwitch action calls updateSpeed on the Pokemon coming in,
    # so it sorts at its real Speed from here rather than the stat setSpecies
    # left behind: a Swift Swim holder arriving in rain is already doubled.
    s.cached_spe[b, side, slot] = wp.uint16(ord_.speed_of(s, dex, b, side))
    # Cute Charm is the only source of attraction here, so the foe's attraction
    # ends the moment its source leaves the field.
    s.vflags[b, 1 - side] = wp.uint32(int(s.vflags[b, 1 - side]) & ~mask_.VF_ATTRACT)


@wp.func
def run_switch(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int):
    """The runSwitch action, which is the second of the two a switch costs.

    Entry hazards go first, then the SwitchIn event, and only then does the
    ability start, so Spikes land before a weather setter announces. Keeping
    this out of the swap matters: the swap's own Update still sees a Pokemon
    that Spikes are about to drop.
    """
    slot = int(s.active[b, side])
    # The SwitchIn event hands a sleeper back every turn Sleep Talk acted
    # through, which is why a RestTalk Pokemon can stall a whole team.
    if int(s.status[b, side, slot]) == ids.COND_SLP:
        s.status_ctr[b, side, slot] = wp.uint8(
            int(s.status_ctr[b, side, slot]) + int(s.sleep_skipped[b, side, slot]))
        s.sleep_skipped[b, side, slot] = wp.uint8(0)
    elif int(s.status[b, side, slot]) == ids.COND_TOX:
        s.status_ctr[b, side, slot] = wp.uint8(0)  # the toxic stage starts over
    layers = int(s.spikes[b, side])
    if layers > 0 and grounded(s, dex, b, side, slot):
        maxhp = int(s.maxhp[b, side, slot])
        loss = maxhp * 3 / 24
        if layers == 2:
            loss = maxhp * 4 / 24
        elif layers >= 3:
            loss = maxhp * 6 / 24
        s.hp[b, side, slot] = wp.uint16(wp.max(int(s.hp[b, side, slot]) - wp.max(loss, 1), 0))
    # A White Herb on either side undoes its drops here. That sort is broken by
    # field position, so it never draws.
    for k in range(2):
        item_fx.white_herb(s, dex, b, k, int(s.active[b, k]))
    if ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_TRUANT):
        # Gen 3 Truant hooks SwitchIn rather than Start, and runSwitch fires
        # SwitchIn before it checks what the hazards left, so this runs even
        # for a Pokemon that Spikes just dropped. It arrives loafing unless the
        # battle has not started yet; a Pokemon that traced Truant never runs
        # this, so it keeps whatever it had.
        if int(s.turn[b]) != 0:
            s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) | mask_.VF_TRUANT)
            s.truant_mask[b, side] = wp.uint8(int(s.truant_mask[b, side]) | (1 << slot))
        else:
            s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_TRUANT)
            s.truant_mask[b, side] = wp.uint8(int(s.truant_mask[b, side]) & ~(1 << slot))
    if int(s.hp[b, side, slot]) <= 0:
        # runSwitch gives up here when the hazards were enough: a Pokemon that
        # Spikes drop on the way in never starts its ability at all.
        return
    # One ability, one Start event. Gen 3 does not start the ability Trace
    # copies, so a Pokemon that traces Drizzle sets no weather and one that
    # traces Intimidate frightens nobody: reading the ability once, before the
    # copy, is what keeps those apart.
    started = int(s.ability[b, side, slot])
    if ability_fx.has(dex, started, ids.ABILITYFAM_TRACE):
        # randomFoe samples even a list of one, so it draws whenever there is
        # anyone to copy at all.
        foe = 1 - side
        foe_slot = int(s.active[b, foe])
        if int(s.hp[b, foe, foe_slot]) > 0:
            rng_.random_n(rng_.draw(s, log, b), 1)
            # Gen 3 has no notrace flag, so Trace copies Trace, which changes
            # nothing and leaves nothing to revert.
            # `-ability|...|[from] ability: Trace|[of] foe` shows both.
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
            mon.reveal(s, b, foe, foe_slot, mon.REVEAL_ABILITY)
            if int(s.ability[b, foe, foe_slot]) != started:
                s.base_ability[b, side] = wp.uint8(started + 1)
                s.ability[b, side, slot] = s.ability[b, foe, foe_slot]


    if ability_fx.has(dex, started, ids.ABILITYFAM_INTIMIDATE):
        # Gen 3 skips it entirely when the only target is behind a Substitute.
        foe = 1 - side
        if int(s.sub_hp[b, foe]) == 0 and int(s.hp[b, foe, int(s.active[b, foe])]) > 0:
            mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
            status_.apply_drops(s, dex, b, foe, mv.set_boost(0, 0, -1), True)

    weather_.forecast(s, dex, b, side)
    weather = ability_fx.weather_set_by(dex, started)
    if weather != 0 and (int(s.weather[b]) != weather or int(s.weather_turns[b]) != 0):
        # Setting weather ends with a WeatherChange event, which sorts the
        # actives. An ability re-setting weather a move had put up still counts
        # as a change, because it makes it permanent.
        s.weather[b] = wp.uint8(weather)
        s.weather_turns[b] = wp.uint8(0)  # an ability sets it for good in gen 3
        mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)
        ord_.tie_draw(s, log, b, ord_.both_actives_alive(s, b))
        for k in range(2):
            weather_.forecast(s, dex, b, k)
