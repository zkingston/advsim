"""JAX baseline for the gate mini-step.

Same state, same draw order and same arithmetic as warp_bench, written the way
JAX wants it: one scalar function per battle under `vmap`, `lax.switch` for the
family dispatch, and masks instead of early returns. Both branches of every
conditional run; that is the cost this baseline exists to measure.
"""
import functools

import jax

# The draw mapping (u64(x) * n) >> 32 and the 64-bit checksum need x64; JAX
# silently truncates them to 32 bits otherwise. Must run before any array is made.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax

import tables as T

SEED_KEY = jnp.uint32(0x9E3779B9)
HP_SCALE = 8
NSTATS = 5
U32 = jnp.uint32
CHART = jnp.array(T.CHART, dtype=jnp.int32)
MOVE = jnp.array(T.MOVE_COLS, dtype=jnp.int32)


def mix(x):
    x = U32(x)
    x ^= x >> U32(16)
    x *= U32(0x7FEB352D)
    x ^= x >> U32(15)
    x *= U32(0x846CA68B)
    x ^= x >> U32(16)
    return x


def draw(lane, gen, c):
    k = mix(U32(lane) ^ SEED_KEY) ^ mix(U32(gen) * U32(2654435761))
    return mix(k ^ mix(U32(c)))


def rand_n(v, n):
    return ((v.astype(jnp.uint64) * jnp.uint64(n)) >> jnp.uint64(32)).astype(jnp.int32)


def boosted(stat, stage):
    up = stat * (2 + stage) // 2
    down = stat * 2 // (2 - stage)
    return jnp.where(stage >= 0, up, down)


def init_lane(lane, gen):
    sides = jnp.arange(2, dtype=jnp.int32)
    h = mix(U32(lane * 2 + sides) ^ mix(U32(gen) ^ SEED_KEY))
    level = 70 + rand_n(mix(h ^ U32(1)), 31)
    mx = (200 + rand_n(mix(h ^ U32(2)), 200)) * HP_SCALE
    stats = jnp.stack([120 + rand_n(mix(h ^ U32(10 + j)), 260) for j in range(NSTATS)], axis=-1)
    t0 = rand_n(mix(h ^ U32(3)), 17)
    t1 = (t0 + 1 + rand_n(mix(h ^ U32(4)), 16)) % 17
    mvs = jnp.stack([rand_n(mix(h ^ U32(20 + j)), 20) for j in range(4)], axis=-1)
    z = jnp.zeros(2, dtype=jnp.int32)
    return dict(hp=mx, maxhp=mx, stats=stats, types=jnp.stack([t0, t1], axis=-1), boosts=jnp.zeros((2, NSTATS), jnp.int32),
                mvs=mvs, status=z, toxctr=z, level=level, ctr=jnp.int32(0), gen=jnp.int32(gen), result=jnp.int32(0))


def type_mod(mtype, dt0, dt1):
    return CHART[mtype, dt0] + jnp.where(dt1 != dt0, CHART[mtype, dt1], 0)


def one_hit(s, lane, att, dfn, mi, c):
    """Damage for one hit; consumes a crit draw then a roll draw."""
    power, cat, mtype = MOVE[mi, 2], MOVE[mi, 1], MOVE[mi, 0]
    ai = jnp.where(cat == 1, 2, 0)
    di = jnp.where(cat == 1, 3, 1)
    a = boosted(s['stats'][att, ai], s['boosts'][att, ai])
    d = boosted(s['stats'][dfn, di], s['boosts'][dfn, di])
    a = jnp.where((s['status'][att] == 4) & (cat == 0), a // 2, a)
    dmg = ((2 * s['level'][att] // 5 + 2) * power * a // d) // 50 + 2
    dmg = jnp.where(rand_n(draw(lane, s['gen'], c), 16) < 1, dmg * 2, dmg)
    stab = (mtype == s['types'][att, 0]) | (mtype == s['types'][att, 1])
    dmg = jnp.where(stab, dmg * 3 // 2, dmg)
    tm = type_mod(mtype, s['types'][dfn, 0], s['types'][dfn, 1])
    for k in range(1, 7):  # fixed unroll: the same doubling/halving ladder, selected
        dmg = jnp.where(tm >= k, dmg * 2, dmg)
    for k in range(1, 7):
        dmg = jnp.where(tm <= -k, dmg // 2, dmg)
    r = rand_n(draw(lane, s['gen'], c + 1), 16)
    dmg = dmg * (100 - r) // 100
    return jnp.where(tm < -6, 0, jnp.maximum(dmg, 1))


def _damage_family(s, lane, att, dfn, mi, c, fam, p0, p1):
    hits = jnp.where(fam == T.F_MULTI, p0, 1)
    total = one_hit(s, lane, att, dfn, mi, c)
    second = one_hit(s, lane, att, dfn, mi, c + 2)
    total = total + jnp.where(hits > 1, second, 0)
    c = c + 2 * hits
    hp = s['hp'].at[dfn].add(-total)
    self_hp = jnp.where(fam == T.F_RECOIL, hp[att] - jnp.maximum(total * p0 // p1, 1),
                        jnp.where(fam == T.F_DRAIN, jnp.minimum(hp[att] + total * p0 // p1, s['maxhp'][att]), hp[att]))
    hp = hp.at[att].set(self_hp)
    sec = (fam == T.F_SEC_STATUS) | (fam == T.F_SEC_DROP) | (fam == T.F_FLINCH)
    fired = (rand_n(draw(lane, s['gen'], c), 100) < p0) & (hp[dfn] > 0) & sec
    status = s['status'].at[dfn].set(jnp.where(fired & (fam == T.F_SEC_STATUS) & (s['status'][dfn] == 0), p1, s['status'][dfn]))
    toxctr = s['toxctr'].at[dfn].set(jnp.where(fired & (fam == T.F_SEC_STATUS) & (s['status'][dfn] == 0), 1, s['toxctr'][dfn]))
    boosts = s['boosts'].at[dfn, p1].set(jnp.where(fired & (fam == T.F_SEC_DROP),
                                                   jnp.maximum(s['boosts'][dfn, p1] - 1, -6), s['boosts'][dfn, p1]))
    flinch = jnp.where(fired & (fam == T.F_FLINCH), dfn, -1)
    return {**s, 'hp': hp, 'status': status, 'toxctr': toxctr, 'boosts': boosts}, c + sec.astype(jnp.int32), flinch


def use_move(s, lane, att, dfn, act, c):
    mi = s['mvs'][att, act]
    fam, acc, p0, p1 = MOVE[mi, 4], MOVE[mi, 3], MOVE[mi, 5], MOVE[mi, 6]
    checks_acc = acc != T.NEVER_MISS
    missed = checks_acc & (rand_n(draw(lane, s['gen'], c), 100) >= acc)
    c_hit = c + checks_acc.astype(jnp.int32)

    boosts = s['boosts']
    for j in range(NSTATS):
        boosts = boosts.at[att, j].set(jnp.where((fam == T.F_BOOST) & (((p0 >> j) & 1) == 1),
                                                 jnp.minimum(boosts[att, j] + p1, 6), boosts[att, j]))
    fresh = s['status'][dfn] == 0
    status = s['status'].at[dfn].set(jnp.where((fam == T.F_STATUS) & fresh, p0, s['status'][dfn]))
    toxctr = s['toxctr'].at[dfn].set(jnp.where((fam == T.F_STATUS) & fresh, 1, s['toxctr'][dfn]))
    hp = s['hp'].at[dfn].add(jnp.where(fam == T.F_FIXED, -s['level'][att], 0))
    simple = {**s, 'boosts': boosts, 'status': status, 'toxctr': toxctr, 'hp': hp}
    dmg_state, c_dmg, flinch = _damage_family(s, lane, att, dfn, mi, c_hit, fam, p0, p1)

    is_simple = (fam == T.F_BOOST) | (fam == T.F_STATUS) | (fam == T.F_FIXED)
    out = jax.tree.map(lambda a, b: jnp.where(is_simple, a, b), simple, dmg_state)
    c_out = jnp.where(is_simple, c_hit, c_dmg)
    flinch = jnp.where(is_simple, -1, flinch)
    out = jax.tree.map(lambda a, b: jnp.where(missed, a, b), s, out)
    return out, jnp.where(missed, c + 1, c_out), jnp.where(missed, -1, flinch)


def advance(s, lane):
    c = s['ctr']
    g = s['gen']
    acts = jnp.stack([rand_n(draw(lane, g, c), 4), rand_n(draw(lane, g, c + 1), 4)])
    c = c + 2
    sp = boosted(s['stats'][:, 4], s['boosts'][:, 4])
    sp = jnp.where(s['status'] == 1, sp // 4, sp)
    tie = sp[0] == sp[1]
    first = jnp.where(sp[1] > sp[0], 1, jnp.where(tie, rand_n(draw(lane, g, c), 2), 0))
    c = c + tie.astype(jnp.int32)

    flinched = jnp.int32(-1)
    for k in range(2):
        att = jnp.where(k == 0, first, 1 - first)
        dfn = 1 - att
        # A side that cannot act consumes no draws at all, so the counter only
        # moves for sides that reach their status check.
        can_act = (s['hp'][att] > 0) & (s['hp'][dfn] > 0) & (flinched != att)
        st = s['status'][att]
        par_stop = can_act & (st == 1) & (rand_n(draw(lane, g, c), 4) < 1)
        thaw = can_act & (st == 5) & (rand_n(draw(lane, g, c), 5) < 1)
        frz_stop = can_act & (st == 5) & ~thaw
        c = c + (can_act & ((st == 1) | (st == 5))).astype(jnp.int32)
        s2 = {**s, 'status': s['status'].at[att].set(jnp.where(thaw, 0, st))}
        acts_ok = can_act & ~par_stop & ~frz_stop
        moved, c_moved, flinch = use_move(s2, lane, att, dfn, acts[att], c)
        s = jax.tree.map(lambda a, b: jnp.where(acts_ok, a, b), moved, s2)
        c = jnp.where(acts_ok, c_moved, c)
        flinched = jnp.where(acts_ok, flinch, flinched)

    st = s['status']
    alive = s['hp'] > 0
    resid = jnp.where((st == 4) | (st == 2), jnp.maximum(s['maxhp'] // 8, 1),
                      jnp.where(st == 3, jnp.maximum(s['maxhp'] * s['toxctr'] // 16, 1), 0))
    s = {**s, 'hp': s['hp'] - jnp.where(alive, resid, 0),
         'toxctr': s['toxctr'] + jnp.where(alive & (st == 3), 1, 0), 'ctr': c}
    down0, down1 = s['hp'][0] <= 0, s['hp'][1] <= 0
    result = jnp.where(down0 & down1, 3, jnp.where(down0, 2, jnp.where(down1, 1, 0)))
    reset = jax.tree.map(lambda a, b: jnp.where(result != 0, a, b), init_lane(lane, g + 1), {**s, 'result': result})
    return reset


@functools.partial(jax.jit, static_argnums=(1,))
def init_batch(lanes, batch):
    return jax.vmap(lambda i: init_lane(i, 0))(lanes)


@jax.jit
def step_batch(state, lanes):
    return jax.vmap(advance)(state, lanes)


@jax.jit
def step_batch_idx(state, lanes, idx):
    """Search-shaped variant: gather the slots to advance, scatter them back."""
    sub = jax.tree.map(lambda a: a[idx], state)
    sub = jax.vmap(advance)(sub, lanes[idx])
    return jax.tree.map(lambda a, b: a.at[idx].set(b), state, sub)


@jax.jit
def fork_batch(state, src, dst):
    return jax.tree.map(lambda a: a.at[dst].set(a[src]), state)


@jax.jit
def checksum(state):
    h = jnp.full(state['hp'].shape[0], 0xCBF29CE484222325, dtype=jnp.uint64)
    prime = jnp.uint64(0x100000001B3)
    for side in range(2):
        for w, v in enumerate([state['hp'][:, side], state['status'][:, side],
                               state['boosts'][:, side, 0] + 8 * state['boosts'][:, side, 3],
                               state['toxctr'][:, side]]):
            h = (h ^ v.astype(jnp.uint32).astype(jnp.uint64)) * prime
    for v in (state['ctr'], state['gen']):
        h = (h ^ v.astype(jnp.uint32).astype(jnp.uint64)) * prime
    return jnp.sum(h)


def make_state(batch):
    lanes = jnp.arange(batch, dtype=jnp.int32)
    return init_batch(lanes, batch), lanes
