"""A move action: the before-move checks, the move and whatever it pulls in
behind it: a Pursuit chase, a drag."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import mon
from advsim.engine import moves as mv
from advsim.engine import order as ord_
from advsim.engine import rng as rng_
from advsim.engine.effects import ability_fx
from advsim.engine.effects import item_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import status as status_
from advsim.engine import slots as slots_
from advsim.engine import execute as execute_
from advsim.engine import switch as switch_
from advsim.engine import residual_fx as rfx_


@wp.func
def use_move(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int, action: int) -> int:
    """The runMove half: turn an action into a move, spend what it costs, and
    note that it was attempted. Everything that happens once a move is known
    lives in `execute_move`, which Sleep Talk calls with a move of its own."""
    foe = 1 - side
    att = int(s.active[b, side])
    dfn = int(s.active[b, foe])
    move = slots_.action_move(s, b, side, action)
    if move == 0:
        return 0
    # A forced action costs no PP: the charge turn already paid, and Struggle
    # has none to pay. Anything else spends the slot the Encore settled on.
    if action != mask_.ACTION_FORCED:
        slot = slots_.action_slot(s, b, side, action)
        cost = 1
        # DeductPP runs over the move's targets, so only a move that reaches
        # the foe costs the extra point: not one aimed at the user, not Heal
        # Bell over its own team, and not a side condition unless it is flagged
        # to pay anyway, which in this vocabulary means Spikes.
        # ModifyMove runs ahead of getMoveTargets, so Curse has already turned
        # itself on its user by the time the targets are read.
        target = dex.move_target[move]
        reaches = target != mv.TARGET_SELF and target != mv.TARGET_ALLY_TEAM and \
            target != mv.TARGET_FOE_SIDE and dex.move_family[move] != ids.MOVEFAM_CURSE
        if ability_fx.has(dex, int(s.ability[b, foe, dfn]), ids.ABILITYFAM_PRESSURE) and \
                (reaches or (dex.move_flags[move] & ids.FLAG_MUSTPRESSURE) != 0):
            cost = 2
        pp = int(s.pp[b, side, att, slot])
        if pp > 0:
            s.pp[b, side, att, slot] = wp.uint8(wp.max(pp - cost, 0))
    # Destiny Bond lasts until its user attempts another move, whatever that
    # move then does. Its own use removes the old one before setting a new one.
    if dex.move_family[move] != ids.MOVEFAM_DESTINYBOND:
        s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_DESTINYBOND)
    s.last_move[b, side] = wp.uint8(move)
    mon.reveal_move(s, b, side, att, move)
    s.last_used[b] = wp.uint8(move)
    # The Choice lock is set by the move's own AfterMove and never updated after,
    # so only the first move a holder attempts matters, failures included.
    # Struggle is the exception Showdown drops outright, its move not being one
    # the Pokemon knows.
    locked_before = int(s.choice_move[b, side]) != 0

    if dex.move_family[move] == ids.MOVEFAM_SLEEPTALK:
        # It works only while asleep, and not at all while a Choice lock or an
        # Encore is already holding the user. Neither refusal costs a draw.
        if int(s.status[b, side, att]) != ids.COND_SLP:
            return 0
        if locked_before or int(s.encore_turns[b, side]) > 0:
            return 0
        move = sleep_talk_pick(s, dex, log, b, side, att)
        mon.reveal_move(s, b, side, att, move)  # the called move gets a line of its own
        if move == 0:
            return 0

    return execute_.execute_move(s, dex, log, b, side, move)


@wp.func
def sleep_talk_pick(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32),
                    b: int, side: int, att: int) -> int:
    """Which move Sleep Talk calls, or 0 for none.

    The pick costs a draw even when only one move is callable, and a move with
    no PP left stays in the list: picking it wastes the turn.
    """
    blocked = ids.FLAG_NOSLEEPTALK | ids.FLAG_CHARGE
    count = int(0)
    for i in range(4):
        m = int(s.moves[b, side, att, i])
        if m != 0 and (dex.move_flags[m] & blocked) == 0:
            count += 1
    if count == 0:
        return 0
    pick = rng_.random_n(rng_.draw(s, log, b), count)
    seen = int(0)
    out = int(0)
    for i in range(4):
        m = int(s.moves[b, side, att, i])
        if m != 0 and (dex.move_flags[m] & blocked) == 0:
            if seen == pick and int(s.pp[b, side, att, i]) > 0:
                out = m
            seen += 1
    return out


@wp.func
def pursuit_intercept(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int,
                      leaving: int):
    """BeforeSwitchOut, where a waiting Pursuit lands on the Pokemon on its way
    out: doubled, never missing, and in place of the move it had queued."""
    chaser = 1 - leaving
    if (int(s.turn_flags[b, chaser]) & mask_.TURN_FLAG_PURSUIT) == 0:
        return
    slot = int(s.active[b, chaser])
    status = int(s.status[b, chaser, slot])
    if int(s.hp[b, chaser, slot]) <= 0 or status == ids.COND_FRZ or status == ids.COND_SLP:
        return
    if (int(s.vflags[b, chaser]) & mask_.VF_TRUANT) != 0:
        return  # loafing; the flag is only ever set on a Truant holder
    action = int(s.pending_act[b, chaser])
    if mask_.is_switch(action) or action == mask_.ACTION_PASS:
        return
    # BeforeSwitchOut spends the PP and notes the move itself, then calls
    # useMove rather than runMove. So there is no BeforeMove event here:
    # paralysis never gets its roll, a flinch cannot stop it, and a Destiny
    # Bond on the chaser survives a move it never formally made. The frozen
    # and sleeping cases are refused above instead, by hand.
    pp_slot = slots_.action_slot(s, b, chaser, action)
    move = int(s.moves[b, chaser, slot, pp_slot])
    if move == 0:
        return
    pp = int(s.pp[b, chaser, slot, pp_slot])
    if pp > 0:
        # The interception still runs useMove, which still asks the Pokemon on
        # its way out about Pressure, so a chase costs two points against one.
        cost = 1
        if ability_fx.has(dex, int(s.ability[b, leaving, int(s.active[b, leaving])]),
                          ids.ABILITYFAM_PRESSURE):
            cost = 2
        s.pp[b, chaser, slot, pp_slot] = wp.uint8(wp.max(pp - cost, 0))
    s.last_move[b, chaser] = wp.uint8(move)
    mon.reveal_move(s, b, chaser, slot, move)
    s.last_used[b] = wp.uint8(move)
    # The chasing flag is what tells the damage path the target is on its way
    # out; the Pursuit flag stays, because its volatile is still a residual
    # handler. Cancelling the action is what stops the chaser moving again.
    s.turn_flags[b, chaser] = wp.uint16(int(s.turn_flags[b, chaser]) | mask_.TURN_FLAG_CHASING)
    s.pending_act[b, chaser] = wp.uint8(mask_.ACTION_PASS)
    s.turn_flags[b, chaser] = wp.uint16(int(s.turn_flags[b, chaser]) | mask_.TURN_FLAG_MOVED)
    target_before = int(s.hp[b, leaving, int(s.active[b, leaving])])
    landed = execute_.execute_move(s, dex, log, b, chaser, move)
    if landed == 1:
        # The Update inside tryMoveHit still runs; the one that closes an action
        # does not, because this was never an action.
        ord_.tie_draw(s, log, b, True)
        status_.run_update(s, dex, b)
    if target_before > 0 and int(s.hp[b, leaving, int(s.active[b, leaving])]) <= 0 and \
            (int(s.vflags[b, leaving]) & mask_.VF_DESTINYBOND) != 0:
        # A chase is still a move, so the Faint event still takes the chaser
        # with it.
        s.hp[b, chaser, slot] = wp.uint16(0)
    item = int(s.item[b, chaser, slot])
    if item != 0 and dex.item_family[item] == ids.ITEMFAM_STAT_ITEM and dex.item_p3[item] == 0 and \
            int(s.choice_move[b, chaser]) == 0:
        s.choice_move[b, chaser] = wp.uint8(move)  # the lock lands after the move, not before


@wp.func
def drag_in(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int, side: int):
    """Pull out whoever is standing and put a random living bench member in.
    The pick is a `sample`, so it draws even when only one is left."""
    alive = int(s.alive_mask[b, side])
    active = int(s.active[b, side])
    count = 0
    for slot in range(6):
        if (alive & (1 << slot)) != 0 and slot != active:
            count += 1
    if count == 0:
        return
    pick = rng_.random_n(rng_.draw(s, log, b), count)
    # The candidates come in Showdown's party order, not the engine's, and the
    # two stop agreeing the moment anything switches.
    chosen = active
    for slot in range(6):
        if (alive & (1 << slot)) != 0 and slot != active:
            seen = 0
            for other in range(6):
                if (alive & (1 << other)) != 0 and other != active and \
                        int(s.party_pos[b, side, other]) < int(s.party_pos[b, side, slot]):
                    seen += 1
            if seen == pick:
                chosen = slot
    if chosen != active:
        switch_.switch_in(s, dex, log, b, side, chosen, False)
        # switchIn cancels whatever the Pokemon it pulled out had queued, so a
        # side dragged before it moves does not move at all.
        s.pending_act[b, side] = wp.uint8(mask_.ACTION_PASS)


@wp.func
def run_move_action(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int,
                    side: int, action: int):
    """One move action, then the sorts that close it. A side that switched this
    turn has no move action at all, and its sorts were spent on the switch; a
    side whose Pokemon fainted has had its action cancelled."""
    if action == mask_.ACTION_PASS or mask_.is_switch(action):
        return
    landed = 0
    acted = False
    # What the action resolves to now: Transform rewrites the move slots under
    # it, so the lock at AfterMove has to name the move that was used, not
    # whatever sits in that slot afterwards.
    used_move = slots_.action_move(s, b, side, action)
    foe_before = int(s.hp[b, 1 - side, int(s.active[b, 1 - side])])
    if int(s.hp[b, side, int(s.active[b, side])]) > 0 and foe_before > 0:
        if status_.can_move(s, dex, log, b, side, used_move):
            s.moved_at[b, side] = wp.uint8(1 + wp.where(int(s.moved_at[b, 1 - side]) != 0, 1, 0))
            s.moved_prio[b, side] = wp.uint8(int(dex.move_priority[used_move]) + 8)
            landed = use_move(s, dex, log, b, side, action)
            acted = True
        else:
            # A move the BeforeMove event stops runs MoveAborted, which throws
            # away a Solar Beam's charge and a Destiny Bond alike: sleep,
            # paralysis, freeze, attraction or a flinch all cancel them.
            s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) &
                                          ~(mask_.VF_TWOTURN | mask_.VF_DESTINYBOND))
            s.twoturn_move[b, side] = wp.uint8(0)
    s.turn_flags[b, side] = wp.uint16(int(s.turn_flags[b, side]) | mask_.TURN_FLAG_MOVED)
    if landed == 1:
        # This sort runs inside the hit, before Showdown processes the faint, so
        # a Pokemon that just dropped to 0 HP still counts as an active here.
        ord_.tie_draw(s, log, b, True)
        status_.run_update(s, dex, b)
    if acted:
        # AfterMove, which is where a White Herb undoes the drop a move just took
        # rather than waiting for the residual phase. Choice Band holds a handler
        # here too, so two of them sort, and tying speeds cost a draw.
        handlers = 0
        item = int(s.item[b, side, int(s.active[b, side])])
        if item != 0 and dex.item_family[item] == ids.ITEMFAM_STAT_ITEM and dex.item_p3[item] == 0:
            handlers += 1
            # The lock is that handler's whole job, and it reads the item the
            # Pokemon is holding now: a Choice item a Trick just handed over
            # locks its new holder to Trick. A forced action locks too when it
            # is a move — the second turn of a Solar Beam, or a Struggle; a
            # recharge turn never gets here, because nothing was used.
            if int(s.choice_move[b, side]) == 0:
                s.choice_move[b, side] = wp.uint8(used_move)
        for k in range(2):
            if rfx_.has_item_residual(s, dex, b, k, rfx_.RESIDUAL_HERB):
                handlers += 1
        if handlers > 1:
            ord_.tie_draw(s, log, b, ord_.both_actives_alive(s, b))
        for k in range(2):
            item_fx.white_herb(s, dex, b, k, int(s.active[b, k]))
    if acted and foe_before > 0 and int(s.hp[b, 1 - side, int(s.active[b, 1 - side])]) <= 0 and \
            (int(s.vflags[b, 1 - side]) & mask_.VF_DESTINYBOND) != 0:
        # faintMessages runs the Faint event after AfterMove and before the
        # action's closing Update, so a White Herb is eaten first and then a
        # Destiny Bond takes the attacker with it. Only
        # a move sets it off, which is exactly what has just happened.
        s.hp[b, side, int(s.active[b, side])] = wp.uint16(0)
    if (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_PASSING) != 0:
        # The request for a recipient goes out before this action's own Update,
        # so a Baton Pass never pays for the sort that closes every other move.
        return
    foe = 1 - side
    dragged = (int(s.turn_flags[b, foe]) & mask_.TURN_FLAG_DRAGGED) != 0
    if dragged:
        s.turn_flags[b, foe] = wp.uint16(int(s.turn_flags[b, foe]) & ~mask_.TURN_FLAG_DRAGGED)
        drag_in(s, dex, log, b, foe)
    status_.close_action(s, dex, log, b)
    if dragged:
        # The runSwitch the drag queued is an action of its own, and ends with
        # an Update like any other.
        switch_.run_switch(s, dex, log, b, foe)
        status_.close_action(s, dex, log, b)
