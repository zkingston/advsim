"""Faints: marking them, asking for replacements, bringing those in, and
deciding the battle when a side has nobody left."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import order as ord_
from advsim.engine import rng as rng_
from advsim.engine import mon
from advsim.engine.effects import ability_fx
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import weather as weather_
from advsim.engine import status as status_
from advsim.engine import slots as slots_
from advsim.engine import switch as switch_


@wp.func
def needs_replacement(s: State, b: int, side: int) -> bool:
    """A side whose active fainted must replace it, if it has anyone left."""
    if int(s.hp[b, side, int(s.active[b, side])]) > 0:
        return False
    alive = int(s.alive_mask[b, side])
    return alive != 0


@wp.func
def battle_over(s: State, b: int) -> bool:
    """A side whose active is down with nobody behind it has lost. Showdown
    calls faintMessages after every residual handler and returns the moment
    that happens, so the rest of the residual never runs."""
    for side in range(2):
        if int(s.hp[b, side, int(s.active[b, side])]) <= 0 and not slots_.has_bench(s, b, side):
            return True
    return False


@wp.func
def mark_faints(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int):
    """Clear the alive bit and the status of anything at 0 HP.

    Showdown drops a fainted Pokemon's status, and the canonical hash needs
    fields owned by an off flag to be zero, so this is where that holds.

    faintMessages ends the Pokemon's ability before it marks it fainted, so an
    Air Lock or Cloud Nine going down announces a weather change first, and the
    sort that goes with it still counts both actives.

    The status itself is not faintMessages' to clear: checkFainted turns it into
    'fnt' afterwards, and checkFainted never runs once a faint has ended the
    battle. So a Pokemon that goes down with its side's last HP keeps its status
    in the final state.
    """
    ending = False
    for side in range(2):
        left = int(s.alive_mask[b, side])
        for slot in range(6):
            if int(s.hp[b, side, slot]) <= 0:
                left = left & ~(1 << slot)
        if left == 0:
            ending = True
    # faintMessages takes the queue in order, and the order shows in the one
    # thing here that reads the other side: a weather suppressor's End sort.
    lead = int(0)
    if (int(s.turn_flags[b, 1]) & mask_.TURN_FLAG_FAINT_FIRST) != 0:
        lead = 1
    for k in range(2):
        side = lead ^ k
        s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) & ~mask_.TURN_FLAG_FAINT_FIRST)
        for slot in range(6):
            if int(s.hp[b, side, slot]) <= 0:
                if slot == int(s.active[b, side]) and \
                        ability_fx.suppresses_weather(dex, int(s.ability[b, side, slot])):
                    # The sort still counts this one, which is not marked
                    # fainted yet, and the other active unless it has already
                    # been through here: two faints at once leave the second
                    # one sorting alone.
                    foe = 1 - side
                    seen = (int(s.alive_mask[b, foe]) >> int(s.active[b, foe])) & 1
                    ord_.tie_draw(s, log, b, seen != 0)
                    for j in range(2):
                        weather_.forecast(s, dex, b, j)
                s.alive_mask[b, side] = wp.uint8(int(s.alive_mask[b, side]) & ~(1 << slot))
                if not ending:
                    mon.cure_status(s, b, side, slot)
                if slot == int(s.active[b, side]):
                    slots_.revert_transform(s, b, side, slot)
                    if int(s.base_ability[b, side]) != 0:
                        s.ability[b, side, slot] = wp.uint8(int(s.base_ability[b, side]) - 1)
                        s.base_ability[b, side] = wp.uint8(0)
                    slots_.clear_slot(s, b, side)
                    # clearVolatile ends with setSpecies, which puts the species'
                    # own types back: a copy or a Color Change does not outlive
                    # the faint, even though the Pokemon stays in the slot.
                    species = int(s.species[b, side, slot])
                    s.types[b, side, 0] = wp.uint8(dex.species_type1[species])
                    s.types[b, side, 1] = wp.uint8(dex.species_type2[species])
                # After the revert, so a Transform's copied Speed goes with it.
                s.cached_spe[b, side, slot] = s.stats[b, side, slot, 4]


@wp.func
def pause_for_replacement(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int) -> bool:
    """Gen 3 asks for a replacement as soon as a Pokemon faints, mid-turn, and
    the rest of the turn runs after it. Returns True if the turn has to stop."""
    mark_faints(s, dex, log, b)
    stop = False
    for side in range(2):
        if needs_replacement(s, b, side):
            s.request[b, side] = wp.uint8(mask_.REQUEST_SWITCH)
            stop = True
        else:
            s.request[b, side] = wp.uint8(mask_.REQUEST_WAIT)
    if stop:
        # A replacement ends the action queue for this turn: whatever had not
        # moved yet does not move, and only the residual phase is left.
        s.pending_act[b, 0] = wp.uint8(mask_.ACTION_PASS)
        s.pending_act[b, 1] = wp.uint8(mask_.ACTION_PASS)
    else:
        for side in range(2):
            s.request[b, side] = wp.uint8(mask_.REQUEST_MOVE)
    settle_result(s, b)
    return stop


@wp.func
def stop_here(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, resume: int) -> bool:
    """After an action: a faint pauses the turn, to pick up at `resume` once
    the replacement is in, and a side with nothing left ends the battle."""
    if pause_for_replacement(s, dex, log, b):
        s.phase[b] = wp.uint8(resume)
        return True
    return decided(s, b)


@wp.func
def close_requests(s: State, b: int):
    """A finished battle asks nothing of anyone. Showdown makes no request once
    checkWin has ended it, so a replacement the last faint opened is not one."""
    if int(s.result[b]) != mask_.RESULT_ONGOING:
        s.request[b, 0] = wp.uint8(mask_.REQUEST_NONE)
        s.request[b, 1] = wp.uint8(mask_.REQUEST_NONE)


@wp.func
def settle_result(s: State, b: int):
    """A side with nothing left has lost; both empty is a tie."""
    out0 = int(s.alive_mask[b, 0]) == 0
    out1 = int(s.alive_mask[b, 1]) == 0
    if out0 and out1:
        s.result[b] = wp.uint8(mask_.RESULT_TIE)
    elif out0:
        s.result[b] = wp.uint8(mask_.RESULT_P2)
    elif out1:
        s.result[b] = wp.uint8(mask_.RESULT_P1)


@wp.func
def pause_for_pass(s: State, b: int) -> bool:
    """A Baton Pass asks for its recipient in the middle of the turn. Nothing
    fainted, so nothing was cancelled: whatever had not moved yet still moves
    once the recipient is in."""
    stop = False
    for side in range(2):
        if (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_PASSING) != 0:
            s.request[b, side] = wp.uint8(mask_.REQUEST_SWITCH)
            s.request[b, 1 - side] = wp.uint8(mask_.REQUEST_WAIT)
            stop = True
    return stop


@wp.func
def decided(s: State, b: int) -> bool:
    """A side with nothing left ends the battle where it stands. Gen 3 clears
    the queue, so no later action, no residual and no endTurn roll follow."""
    settle_result(s, b)
    return int(s.result[b]) != mask_.RESULT_ONGOING


@wp.func
def run_replacements(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, act0: int, act1: int):
    """The replacement segment: instaswitch actions, then their switch-ins.

    Showdown skips an action's closing sort when the next queued action is
    another instaswitch, and inserting the second switch-in draws once because
    it ties with the first.
    """
    both = (int(s.request[b, 0]) == mask_.REQUEST_SWITCH) and (int(s.request[b, 1]) == mask_.REQUEST_SWITCH)
    # Committing the replacements sorts the queue, and the two instaswitch
    # actions carry the speeds of the Pokemon leaving, not the ones arriving.
    # faintMessages clears a fainted Pokemon's volatiles and takes it off the
    # field, so findEventHandlers passes it over: its action speed is the raw
    # stored stat, with no ability, item, boost or paralysis on top.
    # Which instaswitch the sort put first: the faster Pokemon leaving, and on a
    # tie the shuffle, where 0 keeps the order the choices came in, p1 first.
    first_out = int(0)
    if both:
        q0 = ord_.queue_speed(s, dex, b, 0)
        q1 = ord_.queue_speed(s, dex, b, 1)
        if q0 == q1:
            if rng_.random_range(rng_.draw(s, log, b), 0, 2) == 1:
                first_out = 1
        elif q1 > q0:
            first_out = 1
    # The instaswitches run in the order the sort left them, and that order
    # shows: a Cloud Nine arriving first is already there when the other
    # Pokemon's Speed is worked out on its way in.
    tie = int(0)
    for k in range(2):
        side = first_out ^ k
        if int(s.request[b, side]) == mask_.REQUEST_SWITCH:
            action = act0
            if side == 1:
                action = act1
            switch_.switch_in(s, dex, log, b, side, action - mask_.SWITCH_BASE,
                      (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_PASSING) != 0)
            if both and k == 0:
                continue  # the next action is the other instaswitch, so no sort here
            if both and ord_.sort_speed(s, b, 0) == ord_.sort_speed(s, b, 1):
                # Inserting the second switch-in ties with the first, and the
                # draw is where it goes: 0 puts it ahead.
                tie = rng_.random_range(rng_.draw(s, log, b), 0, 2)
            status_.close_action(s, dex, log, b)
    # The runSwitch actions go in the order insertChoice left them: the faster
    # Pokemon arriving first, whichever instaswitch put it there.
    lead = int(0)
    if both:
        second = 1 - first_out
        sp_first = ord_.sort_speed(s, b, 0)
        sp_second = ord_.sort_speed(s, b, 1)
        if first_out == 1:
            sp_first = ord_.sort_speed(s, b, 1)
            sp_second = ord_.sort_speed(s, b, 0)
        if sp_second > sp_first or (sp_second == sp_first and tie == 0):
            lead = second
        else:
            lead = first_out
    for k in range(2):
        side = lead ^ k
        if int(s.request[b, side]) == mask_.REQUEST_SWITCH:
            switch_.run_switch(s, dex, log, b, side)
            # The sort reads the cache as it stands now: a Forecast forme change
            # during the switch-in has just written the raw Speed back into it.
            # Showdown asks for the next replacement before this action's own
            # Update, which is what the liveness guard says here.
            status_.close_action(s, dex, log, b)
            # In gen 3 singles faintMessages cancels the queued action of every
            # active, so a Pokemon that Spikes kill on the way in takes the
            # other side's runSwitch with it: the second one never happens, and
            # the Pokemon that is already on the field never meets the hazards.
            if not ord_.both_actives_alive(s, b):
                break
    for side in range(2):
        s.request[b, side] = wp.uint8(mask_.REQUEST_MOVE)
