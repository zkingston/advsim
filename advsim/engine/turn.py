"""The step skeleton: one call advances a battle to the next decision point.

A turn is Showdown's queue: the order roll, switches, the before-turn moves,
the two actions with a faint check after each, the residual, and Gen 3's Quick
Claw roll in `endTurn`, drawn every turn though no Quick Claw exists here. A
faint pauses the turn for a replacement and `phase` says where to resume.

Every draw goes through `draw`, so replay mode reads Showdown's logged outputs
at exactly the call sites Showdown drew them.
"""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import order as ord_
from advsim.engine import rng as rng_
from advsim.engine.effects import ability_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import status as status_
from advsim.engine import slots as slots_
from advsim.engine import switch as switch_
from advsim.engine import action as action_
from advsim.engine import faint as faint_
from advsim.engine import residual as residual_
from advsim.engine import legal as legal_

SWITCH_PRIORITY = wp.constant(100)  # above every move priority in this format

PHASE_START = wp.constant(0)
PHASE_SECOND_ACTION = wp.constant(1)
PHASE_RESIDUAL = wp.constant(2)
PHASE_ENDTURN = wp.constant(3)


@wp.func
def before_turn_flag(dex: Dex, move: int) -> int:
    """The turn flag for a move that adds a volatile from an action of its own
    at the head of the turn, or 0. Counter and Mirror Coat, Focus Punch and
    Pursuit each cost a sort there and leave a one-turn handler behind whether
    or not the move ever runs."""
    if move == 0:
        return 0
    family = dex.move_family[move]
    if family == ids.MOVEFAM_REFLECT_DAMAGE:
        return mask_.TURN_FLAG_REFLECT
    if family == ids.MOVEFAM_FOCUSPUNCH:
        return mask_.TURN_FLAG_FOCUS
    if family == ids.MOVEFAM_PURSUIT:
        return mask_.TURN_FLAG_PURSUIT
    return 0


@wp.func
def refresh_choice_lock(s: State, dex: Dex, b: int, side: int):
    """choicelock removes itself the moment its holder is not holding a Choice
    item, which is how Knock Off and Trick let a locked Pokemon go. It happens
    inside the DisableMove event, so the sort has already counted it."""
    slot = int(s.active[b, side])
    item = int(s.item[b, side, slot])
    if item == 0 or dex.item_family[item] != ids.ITEMFAM_STAT_ITEM or dex.item_p3[item] != 0:
        s.choice_move[b, side] = wp.uint8(0)
        return
    # It lets go of a move the Pokemon no longer has, which is how a Transform
    # or a Struggle leaves nothing to lock to.
    locked = int(s.choice_move[b, side])
    if locked != 0:
        has = False
        for i in range(4):
            if int(s.moves[b, side, slot, i]) == locked:
                has = True
        if not has:
            s.choice_move[b, side] = wp.uint8(0)


@wp.func
def trap_checks(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int):
    """nextTurn asks, for each active, who is holding it in. Only two handlers
    can answer in singles: the Pokemon's own Magnet Pull, which is an Any
    handler and so finds itself, and whatever trapping ability is across the
    field. Two handlers at the same speed shuffle, and a shuffle costs a draw.
    TrapPokemon and MaybeTrapPokemon each collect the same pair, and no type in
    this chart is immune to trapping, so the second event always runs too."""
    foe = 1 - side
    mine = int(s.ability[b, side, int(s.active[b, side])])
    theirs = int(s.ability[b, foe, int(s.active[b, foe])])
    # Two volatiles can answer as well: Mean Look's trap and a partial trap sit
    # on the same Pokemon, so they tie on speed and on sub-order, and they are
    # sorted ahead of any ability because a condition's sub-order is lower.
    # Only TrapPokemon collects them; neither carries a MaybeTrap handler.
    if (int(s.vflags[b, side]) & mask_.VF_TRAPPED) != 0 and int(s.trap_turns[b, side]) > 0:
        rng_.draw(s, log, b)
    if ord_.sort_speed(s, b, 0) == ord_.sort_speed(s, b, 1) and ability_fx.traps_from_anywhere(dex, mine) and \
            ability_fx.is_trapper(dex, theirs):
        rng_.draw(s, log, b)
        rng_.draw(s, log, b)


@wp.func
def end_turn(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int):
    """Gen 3 rolls for Quick Claw every endTurn, even with none in the format.

    A faint during residuals pauses the turn before this roll, so it lands in
    the segment after the replacement.
    """
    # nextTurn rebuilds each side's disabled set before the Quick Claw roll.
    # Encore and the Choice lock are the only two handlers here, and they sit on
    # the same Pokemon at the same speed, so holding both costs a shuffle.
    s.turn[b] = wp.uint16(int(s.turn[b]) + 1)
    for side in range(2):
        locks = 0
        if int(s.encore_turns[b, side]) > 0:
            locks += 1
        if int(s.choice_move[b, side]) != 0:
            locks += 1
        if locks > 1:
            rng_.draw(s, log, b)
        refresh_choice_lock(s, dex, b, side)
        trap_checks(s, dex, log, b, side)
    if int(s.turn[b]) > 1000:
        # The turn limit, checked before the Quick Claw roll and so costing no
        # draw. Nothing else in Endless Battle Clause can fire here: the
        # no-progress rule is gen 1 only, and staleness needs a Leppa Berry.
        s.result[b] = wp.uint8(mask_.RESULT_TIE)
        return
    rng_.random_chance(rng_.draw(s, log, b), 1, 5)
    for side in range(2):
        s.turn_flags[b, side] = wp.uint16(0)


@wp.func
def advance(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, act0: int, act1: int):
    """Advance one decision point: a whole turn, or a replacement and the rest of one."""
    if int(s.result[b]) != mask_.RESULT_ONGOING:
        return
    act0 = legal_.legalize(s, dex, b, 0, act0)
    act1 = legal_.legalize(s, dex, b, 1, act1)
    # commitChoices calls updateSpeed, so every decision point starts with a
    # fresh Speed on the actives. Anyone on the bench keeps what it had.
    ord_.update_speed(s, dex, b)

    phase = int(s.phase[b])
    if int(s.request[b, 0]) == mask_.REQUEST_SWITCH or int(s.request[b, 1]) == mask_.REQUEST_SWITCH:
        faint_.run_replacements(s, dex, log, b, act0, act1)
        act0 = int(s.pending_act[b, 0])
        act1 = int(s.pending_act[b, 1])
        # Spikes can drop the Pokemon that just came in, which asks for another
        # replacement before the rest of the turn runs.
        if faint_.pause_for_replacement(s, dex, log, b):
            return
        if faint_.decided(s, b):
            return
    else:
        phase = PHASE_START
        s.pending_act[b, 0] = wp.uint8(act0)
        s.pending_act[b, 1] = wp.uint8(act1)

    first = int(s.turn_first[b])

    if phase == PHASE_START:
        # A forced action carries the priority of the move it resolves to, not a
        # switch's; only codes 4 to 9 are switches.
        move0 = 0
        move1 = 0
        # resolvePriority settles OverrideAction before it reads the priority,
        # so an Encore redirects the sort and the beforeTurn queueing too.
        if act0 == mask_.ACTION_FORCED or (not mask_.is_switch(act0) and act0 != mask_.ACTION_PASS):
            move0 = slots_.action_move(s, b, 0, act0)
        if act1 == mask_.ACTION_FORCED or (not mask_.is_switch(act1) and act1 != mask_.ACTION_PASS):
            move1 = slots_.action_move(s, b, 1, act1)
        prio0 = dex.move_priority[move0]
        prio1 = dex.move_priority[move1]
        if mask_.is_switch(act0):
            prio0 = SWITCH_PRIORITY
        if mask_.is_switch(act1):
            prio1 = SWITCH_PRIORITY
        # One sort seats the whole queue, and the beforeTurnMove actions sit
        # ahead of the moves in it, so a tie between those two resolves first.
        if before_turn_flag(dex, move0) != 0 and before_turn_flag(dex, move1) != 0 and \
                ord_.sort_speed(s, b, 0) == ord_.sort_speed(s, b, 1):
            rng_.draw(s, log, b)
        cmp = ord_.moves_first(prio0, ord_.sort_speed(s, b, 0), prio1, ord_.sort_speed(s, b, 1))
        first = 0
        if cmp > 0:
            first = 1
        elif cmp == 0 and ord_.tie_goes_to_second(rng_.draw(s, log, b)):
            first = 1
        s.turn_first[b] = wp.uint8(first)
        for side in range(2):
            s.dmg_taken[b, side] = wp.uint16(0)
            s.dmg_cat[b, side] = wp.uint8(0)
            s.moved_at[b, side] = wp.uint8(0)

        # The beforeTurn action: its own event, then the Update every gen 3
        # action ends with. A beforeTurnMove action of its own follows for each
        # of Counter, Mirror Coat and Focus Punch, and ends with an Update too.
        ord_.tie_draw(s, log, b, ord_.both_actives_alive(s, b))
        status_.close_action(s, dex, log, b)
        for k in range(2):
            early = move0
            if k == 1:
                early = move1
            flag = before_turn_flag(dex, early)
            if flag != 0:
                s.turn_flags[b, k] = wp.uint16(int(s.turn_flags[b, k]) | flag)
                status_.close_action(s, dex, log, b)

        # Switches run before any move, faster first, and each is two actions.
        for k in range(2):
            side = first ^ k
            action = act0
            if side == 1:
                action = act1
            if mask_.is_switch(action):
                # Gen 3 leaves the faint to the end of the action, so a Pursuit
                # that knocks the Pokemon out does not stop the switch it was
                # already making: the chosen replacement still comes in.
                action_.pursuit_intercept(s, dex, log, b, side)
                if faint_.battle_over(s, b):
                    # The chase ended the battle, which a Destiny Bond can do
                    # from the Pokemon it just knocked out: faintMessages ends
                    # it and nothing queued behind runs, this switch included.
                    faint_.pause_for_replacement(s, dex, log, b)
                    return
                switch_.switch_in(s, dex, log, b, side, action - mask_.SWITCH_BASE, False)
                # The switch is this side's action for the turn, which is what
                # makes a Protect on the other side fail: nothing is left to act.
                s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) | mask_.TURN_FLAG_MOVED)
                status_.close_action(s, dex, log, b)
                switch_.run_switch(s, dex, log, b, side)
                status_.close_action(s, dex, log, b)
                # Entry hazards can drop the Pokemon that just arrived, and a
                # faint cancels every action still queued, so the other side's
                # switch never happens at all.
                if faint_.stop_here(s, dex, log, b, PHASE_SECOND_ACTION):
                    return

        side = first
        action = act0
        if side == 1:
            action = act1
        action_.run_move_action(s, dex, log, b, side, action)
        if faint_.stop_here(s, dex, log, b, PHASE_SECOND_ACTION):
            return
        if faint_.pause_for_pass(s, b):
            s.phase[b] = wp.uint8(PHASE_SECOND_ACTION)
            return
        phase = PHASE_SECOND_ACTION

    if phase == PHASE_SECOND_ACTION:
        side = 1 - first
        # Read the action back out of state: a drag can have cancelled it since
        # the turn started.
        action_.run_move_action(s, dex, log, b, side, int(s.pending_act[b, side]))
        if faint_.stop_here(s, dex, log, b, PHASE_RESIDUAL):
            return
        if faint_.pause_for_pass(s, b):
            s.phase[b] = wp.uint8(PHASE_RESIDUAL)
            return
        phase = PHASE_RESIDUAL

    if phase == PHASE_RESIDUAL:
        residual_.run_residual(s, dex, log, b)
        # The residual opened with updateSpeed and may have changed a Speed
        # again; the faints it leaves are sorted against what it ended with.
        if faint_.stop_here(s, dex, log, b, PHASE_ENDTURN):
            return

    end_turn(s, dex, log, b)
    s.phase[b] = wp.uint8(PHASE_START)
    faint_.pause_for_replacement(s, dex, log, b)
