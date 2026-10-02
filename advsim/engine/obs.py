"""The observation: one player's view of a battle, laid out by obs_layout.py.

Everything written here is something that player's Showdown stream shows or
lets it count, so the live converter can rebuild the same vector from the
protocol. The foe's Pokemon appear only once seen, and their moves, ability
and item only as `revealed` says; exact HP, stats and PP are the player's own.
"""
import warp as wp

from advsim.engine import legal as legal_
from advsim.engine import mask as mask_
from advsim.engine import moves as mv
from advsim.engine import mon
from advsim.engine import obs_layout as L
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex

MON_SIZE = wp.constant(len(L.MON))
ACTIVE_SIZE = wp.constant(len(L.ACTIVE))
ACTIVE_BASE = wp.constant(L.ACTIVE_BASE)
FIELD_BASE = wp.constant(L.FIELD_BASE)
MATCHUP_BASE = wp.constant(L.MATCHUP_BASE)
MASK_BASE = wp.constant(L.MASK_BASE)
HISTORY_BASE = wp.constant(L.HISTORY_BASE)
STATS_BASE = wp.constant(L.STATS_BASE)
ORDER_BASE = wp.constant(L.ORDER_BASE)
M_PRESENT = wp.constant(L.MON.index('present'))
M_SPECIES = wp.constant(L.MON.index('species'))
M_LEVEL = wp.constant(L.MON.index('level'))
M_HP_PCT = wp.constant(L.MON.index('hp_pct'))
M_HP = wp.constant(L.MON.index('hp'))
M_MAXHP = wp.constant(L.MON.index('maxhp'))
M_STATUS = wp.constant(L.MON.index('status'))
M_TOXIC_STAGE = wp.constant(L.MON.index('toxic_stage'))
M_FAINTED = wp.constant(L.MON.index('fainted'))
M_ACTIVE = wp.constant(L.MON.index('active'))
M_ABILITY = wp.constant(L.MON.index('ability'))
M_ABILITY_KNOWN = wp.constant(L.MON.index('ability_known'))
M_ITEM = wp.constant(L.MON.index('item'))
M_ITEM_KNOWN = wp.constant(L.MON.index('item_known'))
M_MOVE0 = wp.constant(L.MON.index('move0'))
M_PP0 = wp.constant(L.MON.index('pp0'))
A_BOOST_ATK = wp.constant(L.ACTIVE.index('boost_atk'))
A_TYPE0 = wp.constant(L.ACTIVE.index('type0'))
A_TYPE1 = wp.constant(L.ACTIVE.index('type1'))
A_SUBSTITUTE = wp.constant(L.ACTIVE.index('substitute'))
A_CONFUSION = wp.constant(L.ACTIVE.index('confusion'))
A_LEECH_SEED = wp.constant(L.ACTIVE.index('leech_seed'))
A_ENCORE = wp.constant(L.ACTIVE.index('encore'))
A_PARTIAL_TRAP = wp.constant(L.ACTIVE.index('partial_trap'))
A_YAWN = wp.constant(L.ACTIVE.index('yawn'))
A_PERISH = wp.constant(L.ACTIVE.index('perish'))
A_ATTRACT = wp.constant(L.ACTIVE.index('attract'))
A_TRANSFORMED = wp.constant(L.ACTIVE.index('transformed'))
A_FLASH_FIRE = wp.constant(L.ACTIVE.index('flash_fire'))
A_DESTINY_BOND = wp.constant(L.ACTIVE.index('destiny_bond'))
A_CHARGING = wp.constant(L.ACTIVE.index('charging'))
A_RECHARGE = wp.constant(L.ACTIVE.index('recharge'))
A_TRAPPED = wp.constant(L.ACTIVE.index('trapped'))
A_CHOICE_MOVE = wp.constant(L.ACTIVE.index('choice_move'))
F_WEATHER = wp.constant(L.FIELD.index('weather'))
F_WEATHER_TURNS = wp.constant(L.FIELD.index('weather_turns'))
F_TURN = wp.constant(L.FIELD.index('turn'))
F_SPIKES_OWN = wp.constant(L.FIELD.index('spikes_own'))
F_SPIKES_FOE = wp.constant(L.FIELD.index('spikes_foe'))
F_WISH_OWN = wp.constant(L.FIELD.index('wish_own'))
F_WISH_FOE = wp.constant(L.FIELD.index('wish_foe'))
IMMUNE = wp.constant(-8)  # an immunity, below every real effectiveness sum


@wp.func
def put(out: wp.array3d(dtype=wp.int16), b: int, p: int, i: int, v: int):
    out[b, p, i] = wp.int16(v)


@wp.func
def hp_percent(hp: int, maxhp: int) -> int:
    """Showdown's getHealth with reportPercentages: rounded up, and 99 rather
    than 100 for a Pokemon that is not quite full."""
    if hp <= 0 or maxhp <= 0:
        return 0
    pct = (100 * hp + maxhp - 1) / maxhp
    if pct == 100 and hp < maxhp:
        return 99
    return pct


@wp.func
def own_moves(s: State, b: int, side: int, slot: int, i: int) -> int:
    """The Pokemon's own move in slot i: what it had before a Transform."""
    if slot == int(s.active[b, side]) and (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) != 0:
        return int(s.xf_moves[b, side, i])
    return int(s.moves[b, side, slot, i])


@wp.func
def foe_move(s: State, b: int, side: int, slot: int, k: int) -> int:
    """The k-th smallest move id among the ones this Pokemon has revealed, or
    0. The protocol names a move but never its slot, so a foe's moves go in id
    order: the one order the live converter can reproduce."""
    revealed = int(s.revealed[b, side, slot])
    last = int(0)
    out = int(0)
    for _ in range(k + 1):
        best = int(256)
        for i in range(4):
            m = own_moves(s, b, side, slot, i)
            if (revealed & (1 << (1 + i))) != 0 and m > last and m < best:
                best = m
        if best == 256:
            return 0
        out = best
        last = best
    return out


@wp.func
def shown_species(s: State, b: int, side: int, slot: int) -> int:
    """The species a Pokemon shows as: its own, even while transformed."""
    if slot == int(s.active[b, side]) and (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) != 0:
        return int(s.xf_species[b, side])
    return int(s.species[b, side, slot])


@wp.func
def foe_rank(s: State, b: int, side: int, slot: int) -> int:
    """How many seen Pokemon on this side have a smaller species id."""
    mine = shown_species(s, b, side, slot)
    rank = int(0)
    for i in range(6):
        if (int(s.revealed[b, side, i]) & mon.REVEAL_SEEN) != 0 and shown_species(s, b, side, i) < mine:
            rank += 1
    return rank


@wp.func
def write_mon(s: State, out: wp.array3d(dtype=wp.int16), b: int, p: int, side: int, slot: int, token: int,
              pp_listed: bool):
    base = token * MON_SIZE
    own = side == p
    species = int(s.species[b, side, slot])
    seen = (int(s.revealed[b, side, slot]) & mon.REVEAL_SEEN) != 0
    if species == 0 or (not own and not seen):
        return  # the vector starts zeroed
    active = slot == int(s.active[b, side])
    species = shown_species(s, b, side, slot)
    hp = int(s.hp[b, side, slot])
    maxhp = int(s.maxhp[b, side, slot])
    status = int(s.status[b, side, slot])
    revealed = int(s.revealed[b, side, slot])
    put(out, b, p, base + M_PRESENT, 1)
    put(out, b, p, base + M_SPECIES, species)
    put(out, b, p, base + M_LEVEL, int(s.level[b, side, slot]))
    put(out, b, p, base + M_HP_PCT, hp_percent(hp, maxhp))
    if own:
        put(out, b, p, base + M_HP, hp)
        put(out, b, p, base + M_MAXHP, maxhp)
    if hp <= 0:
        status = 0  # the stream shows a fainted Pokemon as `0 fnt`, whatever it had
    put(out, b, p, base + M_STATUS, status)
    if status == ids.COND_TOX:
        put(out, b, p, base + M_TOXIC_STAGE, int(s.status_ctr[b, side, slot]))
    if hp <= 0:
        put(out, b, p, base + M_FAINTED, 1)
    if active:
        put(out, b, p, base + M_ACTIVE, 1)
    ability = int(s.ability[b, side, slot])
    if active and (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) != 0:
        ability = int(s.xf_ability[b, side])  # a copy shows as itself: the copied ability is not news
    if own or (revealed & mon.REVEAL_ABILITY) != 0:
        put(out, b, p, base + M_ABILITY, ability)
        put(out, b, p, base + M_ABILITY_KNOWN, 1)
    if own or (revealed & mon.REVEAL_ITEM) != 0:
        put(out, b, p, base + M_ITEM, int(s.item[b, side, slot]))
        put(out, b, p, base + M_ITEM_KNOWN, 1)
    for i in range(4):
        if own:
            # The request lists the moves in play, a copy included, and PP
            # only for the active Pokemon.
            put(out, b, p, base + M_MOVE0 + i, int(s.moves[b, side, slot, i]))
            if active and pp_listed:
                put(out, b, p, base + M_PP0 + i, int(s.pp[b, side, slot, i]))
        else:
            put(out, b, p, base + M_MOVE0 + i, foe_move(s, b, side, slot, i))


@wp.func
def write_active(s: State, out: wp.array3d(dtype=wp.int16), b: int, p: int, side: int, token: int):
    base = ACTIVE_BASE + token * ACTIVE_SIZE
    boosts = int(s.boosts[b, side])
    for i in range(7):
        put(out, b, p, base + A_BOOST_ATK + i, mv.get_boost(boosts, i))
    flags = int(s.vflags[b, side])
    put(out, b, p, base + A_TYPE0, int(s.types[b, side, 0]))
    put(out, b, p, base + A_TYPE1, int(s.types[b, side, 1]))
    put(out, b, p, base + A_SUBSTITUTE, wp.where(int(s.sub_hp[b, side]) > 0, 1, 0))
    put(out, b, p, base + A_CONFUSION, wp.where(int(s.confusion_turns[b, side]) > 0, 1, 0))
    put(out, b, p, base + A_LEECH_SEED, wp.where((flags & mask_.VF_LEECHSEED) != 0, 1, 0))
    put(out, b, p, base + A_ENCORE, wp.where(int(s.encore_turns[b, side]) > 0, 1, 0))
    put(out, b, p, base + A_PARTIAL_TRAP, wp.where(int(s.trap_turns[b, side]) > 0, 1, 0))
    put(out, b, p, base + A_YAWN, wp.where(int(s.yawn_turns[b, side]) > 0, 1, 0))
    # The count starts at 4 and the stream shows it from `perish3`.
    put(out, b, p, base + A_PERISH, wp.min(int(s.perish_count[b, side]), 3))
    put(out, b, p, base + A_ATTRACT, wp.where((flags & mask_.VF_ATTRACT) != 0, 1, 0))
    put(out, b, p, base + A_TRANSFORMED, wp.where((flags & mask_.VF_TRANSFORMED) != 0, 1, 0))
    put(out, b, p, base + A_FLASH_FIRE, wp.where((flags & mask_.VF_FLASHFIRE) != 0, 1, 0))
    put(out, b, p, base + A_DESTINY_BOND, wp.where((flags & mask_.VF_DESTINYBOND) != 0, 1, 0))
    put(out, b, p, base + A_CHARGING, wp.where((flags & mask_.VF_TWOTURN) != 0, 1, 0))
    put(out, b, p, base + A_RECHARGE, wp.where((flags & mask_.VF_MUSTRECHARGE) != 0, 1, 0))
    put(out, b, p, base + A_TRAPPED, wp.where((flags & mask_.VF_TRAPPED) != 0, 1, 0))
    # The lock only while the band is held: Showdown keeps the volatile after
    # a Trick hands the band away until its next check, but it binds nothing
    # then, and the player cannot tell it is there.
    if side == p and int(s.item[b, side, int(s.active[b, side])]) == ids.ITEM_CHOICEBAND:
        put(out, b, p, base + A_CHOICE_MOVE, int(s.choice_move[b, side]))


@wp.func
def write_matchup(s: State, dex: Dex, out: wp.array3d(dtype=wp.int16), b: int, p: int,
                  side: int, token: int):
    """Each move `side`'s active shows the player, against the other active's
    current types: the effectiveness sum (IMMUNE for an immunity) and STAB."""
    other = 1 - side
    slot = int(s.active[b, side])
    for i in range(4):
        move = int(s.moves[b, side, slot, i])
        if side != p:
            move = foe_move(s, b, side, slot, i)
        # A foe's Hidden Power shows only its name, never its type.
        hidden = side != p and move != 0 and dex.move_type_from_mon[move] != 0
        if move != 0 and not hidden and dex.move_category[move] != ids.CATEGORY_STATUS:
            t = mv.type_of(dex, move, int(s.hp_type[b, side, slot]))
            eff = mv.type_multiplier_of(dex, t, int(s.types[b, other, 0]), int(s.types[b, other, 1]))
            if eff == -99:
                eff = IMMUNE
            put(out, b, p, MATCHUP_BASE + token * 8 + 2 * i, eff)
            if t == int(s.types[b, side, 0]) or t == int(s.types[b, side, 1]):
                put(out, b, p, MATCHUP_BASE + token * 8 + 2 * i + 1, 1)


@wp.func
def observe(s: State, dex: Dex, out: wp.array3d(dtype=wp.int16), b: int, p: int):
    """Player p's observation of battle b. `out` must be zeroed first."""
    foe = 1 - p
    legal = legal_.legal_actions(s, dex, b, p)
    # The request lists PP only in a move request that offers every move: a
    # lock, a recharge or Struggle lists one, and then there is nothing to read.
    pp_listed = int(s.request[b, p]) == mask_.REQUEST_MOVE and ((legal >> mask_.ACTION_FORCED) & 1) == 0
    for slot in range(6):
        write_mon(s, out, b, p, p, slot, slot, pp_listed)
        # The foe's party order is never shown, so its seen Pokemon go in
        # species-id order: the one order the converter can reproduce.
        write_mon(s, out, b, p, foe, slot, 6 + foe_rank(s, b, foe, slot), False)
    write_active(s, out, b, p, p, 0)
    write_active(s, out, b, p, foe, 1)
    put(out, b, p, FIELD_BASE + F_WEATHER, int(s.weather[b]))
    put(out, b, p, FIELD_BASE + F_WEATHER_TURNS, int(s.weather_turns[b]))
    put(out, b, p, FIELD_BASE + F_TURN, wp.min(int(s.turn[b]), 1000))
    put(out, b, p, FIELD_BASE + F_SPIKES_OWN, int(s.spikes[b, p]))
    put(out, b, p, FIELD_BASE + F_SPIKES_FOE, int(s.spikes[b, foe]))
    put(out, b, p, FIELD_BASE + F_WISH_OWN, wp.where(int(s.wish_turns[b, p]) > 0, 1, 0))
    put(out, b, p, FIELD_BASE + F_WISH_FOE, wp.where(int(s.wish_turns[b, foe]) > 0, 1, 0))
    write_matchup(s, dex, out, b, p, p, 0)
    write_matchup(s, dex, out, b, p, foe, 1)
    for a in range(12):
        put(out, b, p, MASK_BASE + a, (legal >> a) & 1)
    # The move each active last used since it came in: the protocol names
    # every one, so both sides' are the player's to see.
    put(out, b, p, HISTORY_BASE, int(s.last_move[b, p]))
    put(out, b, p, HISTORY_BASE + 1, int(s.last_move[b, foe]))
    # Own stats as the request lists them: a copy's are the original's, since
    # Transform overwrites the live stats and the request shows the stored ones.
    xf = (int(s.vflags[b, p]) & mask_.VF_TRANSFORMED) != 0
    for slot in range(6):
        if int(s.species[b, p, slot]) != 0:
            for k in range(5):
                v = int(s.stats[b, p, slot, k])
                if xf and slot == int(s.active[b, p]):
                    v = int(s.xf_stats[b, p, k])
                put(out, b, p, STATS_BASE + 5 * slot + k, v)
    # Who moved first, when both sides used a move of their own at the same
    # priority: each move line shows, and its priority is the move's.
    mine = int(s.moved_at[b, p])
    theirs = int(s.moved_at[b, foe])
    order = 0
    if mine != 0 and theirs != 0 and int(s.moved_prio[b, p]) == int(s.moved_prio[b, foe]):
        order = wp.where(mine < theirs, 1, -1)
    put(out, b, p, ORDER_BASE, order)
