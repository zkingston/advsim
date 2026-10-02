"""The residual action: Showdown's handler list built, sorted and run, one
handler at a time, with the faint check between them."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import order as ord_
from advsim.engine import rng as rng_
from advsim.engine.effects import ability_fx
from advsim.engine.effects import item_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine import weather as weather_
from advsim.engine import status as status_
from advsim.engine import residual_fx as rfx_
from advsim.engine import faint as faint_

# The residual's handler list, as `fieldEvent` builds it. Few are live at once,
# but the ceiling is what has to fit: one for the weather and N_COLLECTED a side,
# 37. Writing past a Warp vector corrupts the heap rather than failing.
HANDLERS = wp.types.vector(length=40, dtype=wp.int32)

H_WEATHER = wp.constant(0)
H_WISH = wp.constant(1)
H_STATUS = wp.constant(2)
H_LEECH = wp.constant(3)
H_TRAP = wp.constant(4)
H_ENCORE = wp.constant(5)
H_YAWN = wp.constant(6)
H_PERISH = wp.constant(7)
H_TRUANT = wp.constant(8)
H_ABILITY = wp.constant(9)
H_ITEM = wp.constant(10)
H_HERB = wp.constant(11)
H_DURATION = wp.constant(12)      # protect or endure: one turn, always ends here
H_DUR_FLINCH = wp.constant(13)
H_DUR_STALL = wp.constant(14)
H_DUR_TWOTURN = wp.constant(15)
H_DUR_RECHARGE = wp.constant(16)
H_DUR_SIDE = wp.constant(17)      # the beforeTurn side conditions, one turn each


@wp.func
def res_key(order: int, sub: int, speed: int) -> int:
    """`comparePriority` as one sortable integer: order ascending, then speed
    descending, then sub-order ascending. A handler with no order of its own
    sorts behind every numbered one, which is what 63 stands for here."""
    return order * 262144 + (2047 - speed) * 64 + sub


@wp.func
def weather_residual(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int):
    """Order 8: the duration runs down, and when it reaches zero clearWeather
    fires a WeatherChange and the handler stops there, so the upkeep never
    happens that turn. One sort either way, and no sandstorm damage on the turn
    it lifts."""
    raw_weather = int(s.weather[b])
    if raw_weather == 0:
        return
    turns = int(s.weather_turns[b])
    if turns == 1:
        s.weather[b] = wp.uint8(0)
        s.weather_turns[b] = wp.uint8(0)
        ord_.tie_draw(s, log, b, ord_.both_actives_alive(s, b))
        for k in range(2):
            weather_.forecast(s, dex, b, k)
        return
    # Rain and sun sort whatever Air Lock says. Sandstorm asks first whether its
    # weather is in effect, and a suppressed one is not.
    if raw_weather != ids.WEATHER_SAND or weather_.effective_weather(s, dex, b) == ids.WEATHER_SAND:
        ord_.tie_draw(s, log, b, ord_.both_actives_alive(s, b))
    if weather_.effective_weather(s, dex, b) == ids.WEATHER_SAND:
        order = 0
        if ord_.sort_speed(s, b, 1) > ord_.sort_speed(s, b, 0):
            order = 1
        for k in range(2):
            side = order ^ k
            slot = int(s.active[b, side])
            blocked = dex.type_status_immune[int(s.types[b, side, 0])] | \
                dex.type_status_immune[int(s.types[b, side, 1])]
            if int(s.hp[b, side, slot]) > 0 and (blocked & (1 << ids.COND_SANDSTORM)) == 0 and \
                    not ability_fx.blocks_sand(dex, int(s.ability[b, side, slot])):
                s.hp[b, side, slot] = wp.uint16(wp.max(
                    int(s.hp[b, side, slot]) - wp.max(int(s.maxhp[b, side, slot]) / 16, 1), 0))
    if turns > 0:
        s.weather_turns[b] = wp.uint8(turns - 1)


N_COLLECTED = wp.constant(18)


@wp.func
def handler_at(s: State, dex: Dex, b: int, side: int, c: int) -> int:
    """The c-th handler `fieldEvent` can collect from a side, as an H_ kind, or
    -1 when the side carries none there. The order is Showdown's: the Pokemon's
    status, its volatiles in the order they were added, its ability and its
    item, then the slot's Wish.

    The volatiles that carry only a duration all share one key, so they tie,
    and the shuffle permutes them from this order. Protect is added anew each
    turn, so a stall volatile that was already there (its counter past 2) comes
    before it and a fresh one after. Flinch lands on a Pokemon after its own
    move set a charge or a recharge. The order only shows when a handler that
    counts down runs the faint check and that ends the battle.
    """
    slot = int(s.active[b, side])
    alive = int(s.hp[b, side, slot]) > 0
    flags = int(s.vflags[b, side])
    stall = int(s.stall_ctr[b, side])
    if c == 0 and rfx_.has_status_residual(s, b, side):
        return H_STATUS
    if c == 1 and alive and (flags & mask_.VF_LEECHSEED) != 0:
        return H_LEECH
    if c == 2 and alive and int(s.trap_turns[b, side]) > 0:
        return H_TRAP
    if c == 3 and alive and int(s.encore_turns[b, side]) > 0:
        return H_ENCORE
    if c == 4 and alive and int(s.yawn_turns[b, side]) > 0:
        return H_YAWN
    if c == 5 and alive and int(s.perish_count[b, side]) > 0:
        return H_PERISH
    if c == 6 and alive and ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_TRUANT):
        return H_TRUANT
    if c == 7 and stall > 2:
        return H_DUR_STALL
    if c == 8 and (flags & (mask_.VF_PROTECT | mask_.VF_ENDURE)) != 0:
        return H_DURATION
    if c == 9 and stall > 0 and stall <= 2:
        return H_DUR_STALL
    if c == 10 and (flags & mask_.VF_TWOTURN) != 0:
        return H_DUR_TWOTURN
    if c == 11 and (flags & mask_.VF_MUSTRECHARGE) != 0:
        return H_DUR_RECHARGE
    if c == 12 and (flags & mask_.VF_FLINCH) != 0:
        return H_DUR_FLINCH
    if c == 13 and (int(s.turn_flags[b, side]) &
                    (mask_.TURN_FLAG_REFLECT | mask_.TURN_FLAG_FOCUS | mask_.TURN_FLAG_PURSUIT)) != 0:
        return H_DUR_SIDE
    if c == 14 and rfx_.has_ability_residual(s, dex, b, side):
        return H_ABILITY
    if c == 15 and rfx_.has_item_residual(s, dex, b, side, rfx_.RESIDUAL_ITEMS):
        return H_ITEM
    if c == 16 and rfx_.has_item_residual(s, dex, b, side, rfx_.RESIDUAL_HERB):
        return H_HERB
    # A slot condition, collected after the Pokemon's own handlers and carrying
    # the Speed of whoever is standing in the slot.
    if c == 17 and int(s.wish_turns[b, side]) > 0:
        return H_WISH
    return -1


@wp.func
def kind_key(kind: int, spe: int) -> int:
    """Each handler's residual order and sub-order, as the sort key."""
    if kind == H_STATUS:
        return res_key(10, 6, spe)
    if kind == H_LEECH:
        return res_key(10, 5, spe)
    if kind == H_TRAP:
        return res_key(10, 9, spe)
    if kind == H_ENCORE:
        return res_key(10, 14, spe)
    if kind == H_YAWN:
        return res_key(10, 19, spe)
    if kind == H_PERISH:
        return res_key(12, 2, spe)
    if kind == H_TRUANT:
        return res_key(27, 7, spe)
    if kind == H_ABILITY:
        return res_key(10, 3, spe)
    if kind == H_ITEM:
        return res_key(10, 4, spe)
    if kind == H_HERB:
        return res_key(29, 8, spe)
    if kind == H_WISH:
        return res_key(7, 3, spe)
    return res_key(63, 2, spe)  # the volatiles that carry only a duration


@wp.func
def run_residual(s: State, dex: Dex, log: wp.array2d(dtype=wp.uint32), b: int):
    """The residual action: build Showdown's handler list, sort it the way
    Showdown sorts it, and run it.

    `fieldEvent` collects the field's handlers, then each side's: the active
    Pokemon's status, its volatiles, its ability and its item, then the slot
    conditions. `speedSort` is a selection sort that moves each group of tying
    handlers to the front and shuffles it, so the order two sides end up in
    depends on the whole list and not just on the pair. Modelling the pairs one
    at a time got the draw counts right and the answers wrong.
    """
    # The residual action calls updateSpeed first, which puts the real stat back
    # in front of every sort from here on.
    ord_.update_speed(s, dex, b)

    keys = HANDLERS()
    tags = HANDLERS()
    n = int(0)
    if int(s.weather[b]) != 0:
        # A field handler, and the field has no Speed of its own.
        keys[n] = res_key(8, 5, 0)
        tags[n] = H_WEATHER * 2
        n = n + 1
    for side in range(2):
        spe = ord_.sort_speed(s, b, side)
        for c in range(N_COLLECTED):
            kind = handler_at(s, dex, b, side, c)
            if kind >= 0:
                keys[n] = kind_key(kind, spe)
                tags[n] = kind * 2 + side
                n = n + 1

    # speedSort: pick the minimum, move every handler tying with it to the front
    # in list order, shuffle that group, repeat. A group of k costs k-1 draws.
    srt = int(0)
    while srt + 1 < n:
        low = keys[srt]
        for i in range(srt + 1, n):
            if keys[i] < low:
                low = keys[i]
        # The value, not the index: moving a handler forward overwrites what
        # was there, and Showdown collects the tying indices before it swaps.
        pos = int(srt)
        for i in range(srt, n):
            if keys[i] == low:
                if i != pos:
                    kt = keys[pos]
                    keys[pos] = keys[i]
                    keys[i] = kt
                    tt = tags[pos]
                    tags[pos] = tags[i]
                    tags[i] = tt
                pos = pos + 1
        cnt = pos - srt
        if cnt > 1:
            for j in range(srt, srt + cnt - 1):
                r = rng_.random_range(rng_.draw(s, log, b), j, srt + cnt)
                if r != j:
                    kt = keys[j]
                    keys[j] = keys[r]
                    keys[r] = kt
                    tt = tags[j]
                    tags[j] = tags[r]
                    tags[r] = tt
        srt = srt + cnt

    queued = int(0)
    for i in range(n):
        kind = tags[i] / 2
        side = tags[i] - kind * 2
        if kind != H_WEATHER and int(s.hp[b, side, int(s.active[b, side])]) <= 0 and \
                ((queued >> side) & 1) == 0:
            # Showdown passes over a handler whose holder is `fainted`, which
            # faintMessages sets. A Perish Song queues its faint without one, so
            # that Pokemon is still at the table for the handlers behind it.
            continue
        if kind == H_WEATHER:
            weather_residual(s, dex, log, b)
        elif kind == H_WISH:
            rfx_.wish_residual(s, b, side)
        elif kind == H_ABILITY:
            rfx_.ability_residual(s, dex, log, b, side)
        elif kind == H_ITEM:
            slot = int(s.active[b, side])
            item_fx.residual_heal(s, dex, b, side, slot)
            item_fx.pinch_berry(s, dex, b, side, slot)
        elif kind == H_LEECH:
            rfx_.leech_seed(s, dex, b, side)
        elif kind == H_STATUS:
            rfx_.residual_damage(s, b, side)
        elif kind == H_TRAP:
            rfx_.partial_trap(s, b, side)
        elif kind == H_ENCORE:
            rfx_.encore_residual(s, b, side)
        elif kind == H_YAWN:
            rfx_.yawn_residual(s, dex, log, b, side)
        elif kind == H_PERISH:
            # Its duration running out queues a faint and takes the `continue`
            # that skips faintMessages, so the next handler still runs. A count
            # that only goes down runs its callback and then the faint check,
            # which ends the battle if the other side's Perish just queued one.
            rfx_.perish_residual(s, b, side)
            if int(s.hp[b, side, int(s.active[b, side])]) <= 0:
                queued = queued | (1 << side)
                continue
        elif kind == H_TRUANT:
            rfx_.truant_residual(s, dex, b, side)
        elif kind == H_HERB:
            item_fx.white_herb(s, dex, b, side, int(s.active[b, side]))
        elif kind == H_DURATION:
            # Protect and Endure last one turn, so this is where they end. A
            # duration that runs out takes the branch that skips faintMessages,
            # which is why a faint Perish Song queued does not stop the rest.
            s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) &
                                          ~(mask_.VF_PROTECT | mask_.VF_ENDURE))
            continue
        elif kind == H_DUR_FLINCH:
            s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_FLINCH)
            continue
        elif kind == H_DUR_STALL:
            # The stall volatile carries two turns and starts over every time
            # Protect or Endure lands, so on the turn it landed this counts down
            # without ending — and a handler that does not end runs the faint
            # check at the bottom of the loop.
            if (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_STALLED) == 0:
                s.stall_ctr[b, side] = wp.uint8(0)
                continue
        elif kind == H_DUR_TWOTURN:
            if (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_CHARGE_SET) == 0:
                s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_TWOTURN)
                s.twoturn_move[b, side] = wp.uint8(0)
                continue
        elif kind == H_DUR_RECHARGE:
            if (int(s.turn_flags[b, side]) & mask_.TURN_FLAG_RECHARGE_SET) == 0:
                s.vflags[b, side] = wp.uint32(int(s.vflags[b, side]) & ~mask_.VF_MUSTRECHARGE)
                continue
        if faint_.battle_over(s, b):
            return


    # The residual is an action like any other, so it closes with the same sort
    # and the same Update: a Lum Berry cures the sleep a Yawn just delivered.
    status_.close_action(s, dex, log, b)
