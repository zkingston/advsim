"""Warp implementation of the gate mini-step.

One thread per battle: pick both moves, order by speed, resolve two attacks
through the Gen 3 integer damage pipeline, apply residual damage, and auto-reset
finished battles. Every random draw goes through a counter hash, in a fixed
order, so the JAX and CUDA versions can reproduce the same final state.
"""
import numpy as np
import warp as wp

import tables as T

wp.set_module_options({"enable_backward": False})

SEED_KEY = wp.constant(wp.uint32(0x9E3779B9))
NSTATS = 5
# HP is scaled up so a lane faints roughly every 15 steps, the rate a real battle
# ends at. Unscaled, a KO lands every second step and the reset path, which is
# uniform work, would dilute the divergence this benchmark exists to measure.
HP_SCALE = wp.constant(8)


@wp.struct
class St:
    hp: wp.array2d(dtype=wp.int32)
    maxhp: wp.array2d(dtype=wp.int32)
    stats: wp.array3d(dtype=wp.int32)
    types: wp.array3d(dtype=wp.int32)
    boosts: wp.array3d(dtype=wp.int32)
    mvs: wp.array3d(dtype=wp.int32)
    status: wp.array2d(dtype=wp.int32)
    toxctr: wp.array2d(dtype=wp.int32)
    level: wp.array2d(dtype=wp.int32)
    ctr: wp.array(dtype=wp.int32)
    gen: wp.array(dtype=wp.int32)
    result: wp.array(dtype=wp.int32)
    chart: wp.array2d(dtype=wp.int32)
    move: wp.array2d(dtype=wp.int32)


@wp.func
def mix(x: wp.uint32) -> wp.uint32:
    x ^= x >> wp.uint32(16)
    x *= wp.uint32(0x7FEB352D)
    x ^= x >> wp.uint32(15)
    x *= wp.uint32(0x846CA68B)
    x ^= x >> wp.uint32(16)
    return x


@wp.func
def draw(lane: int, gen: int, c: int) -> wp.uint32:
    """Counter hash: the draw depends only on (lane, reset generation, counter)."""
    k = mix(wp.uint32(lane) ^ SEED_KEY) ^ mix(wp.uint32(gen) * wp.uint32(2654435761))
    return mix(k ^ mix(wp.uint32(c)))


@wp.func
def rand_n(v: wp.uint32, n: int) -> int:
    return int((wp.uint64(v) * wp.uint64(n)) >> wp.uint64(32))


@wp.func
def boosted(stat: int, stage: int) -> int:
    if stage >= 0:
        return stat * (2 + stage) / 2
    return stat * 2 / (2 - stage)


@wp.func
def init_lane(s: St, lane: int, gen: int):
    for side in range(2):
        h = mix(wp.uint32(lane * 2 + side) ^ mix(wp.uint32(gen) ^ SEED_KEY))
        s.level[lane, side] = 70 + rand_n(mix(h ^ wp.uint32(1)), 31)
        mx = (200 + rand_n(mix(h ^ wp.uint32(2)), 200)) * HP_SCALE
        s.maxhp[lane, side] = mx
        s.hp[lane, side] = mx
        for j in range(NSTATS):
            s.stats[lane, side, j] = 120 + rand_n(mix(h ^ wp.uint32(10 + j)), 260)
            s.boosts[lane, side, j] = 0
        t0 = rand_n(mix(h ^ wp.uint32(3)), 17)
        s.types[lane, side, 0] = t0
        s.types[lane, side, 1] = (t0 + 1 + rand_n(mix(h ^ wp.uint32(4)), 16)) % 17
        for j in range(4):
            s.mvs[lane, side, j] = rand_n(mix(h ^ wp.uint32(20 + j)), 20)
        s.status[lane, side] = 0
        s.toxctr[lane, side] = 0
    s.ctr[lane] = 0
    s.gen[lane] = gen
    s.result[lane] = 0


@wp.func
def type_mod(s: St, mtype: int, lane: int, dfn: int) -> int:
    m = s.chart[mtype, s.types[lane, dfn, 0]]
    t1 = s.types[lane, dfn, 1]
    if t1 != s.types[lane, dfn, 0]:
        m += s.chart[mtype, t1]
    return m


@wp.func
def one_hit(s: St, lane: int, att: int, dfn: int, mi: int, c: int) -> int:
    """Damage for one hit; consumes a crit draw and a roll draw."""
    power = s.move[mi, 2]
    cat = s.move[mi, 1]
    mtype = s.move[mi, 0]
    ai = 0
    di = 1
    if cat == 1:
        ai = 2
        di = 3
    a = boosted(s.stats[lane, att, ai], s.boosts[lane, att, ai])
    d = boosted(s.stats[lane, dfn, di], s.boosts[lane, dfn, di])
    if s.status[lane, att] == 4 and cat == 0:
        a = a / 2
    dmg = int(((2 * s.level[lane, att] / 5 + 2) * power * a / d) / 50 + 2)
    if rand_n(draw(lane, s.gen[lane], c), 16) < 1:
        dmg *= 2
    if mtype == s.types[lane, att, 0] or mtype == s.types[lane, att, 1]:
        dmg = dmg * 3 / 2
    tm = type_mod(s, mtype, lane, dfn)
    if tm < -6:
        return 0
    for _ in range(tm):
        dmg *= 2
    for _ in range(-tm):
        dmg = dmg / 2
    r = rand_n(draw(lane, s.gen[lane], c + 1), 16)
    dmg = dmg * (100 - r) / 100
    return wp.max(dmg, 1)


@wp.func
def use_move(s: St, lane: int, att: int, dfn: int, act: int, c: int) -> int:
    """Resolve one move. Returns the new counter packed with the flinch flag: c * 2 + flinched."""
    mi = s.mvs[lane, att, act]
    fam = s.move[mi, 4]
    acc = s.move[mi, 3]
    p0 = s.move[mi, 5]
    p1 = s.move[mi, 6]
    if acc != 255:  # a never-miss move draws nothing, so the counter must not move
        if rand_n(draw(lane, s.gen[lane], c), 100) >= acc:
            return (c + 1) * 2
        c += 1
    if fam == 4:  # self boost
        for j in range(NSTATS):
            if (p0 >> j) & 1:
                s.boosts[lane, att, j] = wp.min(s.boosts[lane, att, j] + p1, 6)
        return c * 2
    if fam == 3:  # status move
        if s.status[lane, dfn] == 0:
            s.status[lane, dfn] = p0
            s.toxctr[lane, dfn] = 1
        return c * 2
    if fam == 8:  # fixed damage
        s.hp[lane, dfn] -= s.level[lane, att]
        return c * 2
    hits = 1
    if fam == 5:
        hits = p0
    total = int(0)
    cc = int(c)
    for _ in range(hits):
        total += one_hit(s, lane, att, dfn, mi, cc)
        cc += 2
    c = cc
    s.hp[lane, dfn] -= total
    if fam == 6:  # recoil
        s.hp[lane, att] -= wp.max(total * p0 / p1, 1)
    elif fam == 7:  # drain
        s.hp[lane, att] = wp.min(s.hp[lane, att] + total * p0 / p1, s.maxhp[lane, att])
    fl = int(0)
    if fam == 1 or fam == 2 or fam == 9:  # secondary effect
        if rand_n(draw(lane, s.gen[lane], c), 100) < p0 and s.hp[lane, dfn] > 0:
            if fam == 1 and s.status[lane, dfn] == 0:
                s.status[lane, dfn] = p1
                s.toxctr[lane, dfn] = 1
            elif fam == 2:
                s.boosts[lane, dfn, p1] = wp.max(s.boosts[lane, dfn, p1] - 1, -6)
            elif fam == 9:
                fl = 1
        c += 1
    return c * 2 + fl


@wp.func
def advance(s: St, lane: int):
    c = s.ctr[lane]
    g = s.gen[lane]
    a0 = rand_n(draw(lane, g, c), 4)
    a1 = rand_n(draw(lane, g, c + 1), 4)
    c += 2
    sp0 = boosted(s.stats[lane, 0, 4], s.boosts[lane, 0, 4])
    sp1 = boosted(s.stats[lane, 1, 4], s.boosts[lane, 1, 4])
    if s.status[lane, 0] == 1:
        sp0 = sp0 / 4
    if s.status[lane, 1] == 1:
        sp1 = sp1 / 4
    first = int(0)
    if sp1 > sp0:
        first = 1
    elif sp1 == sp0:
        first = rand_n(draw(lane, g, c), 2)
        c += 1
    flinched = int(-1)
    c = int(c)
    for k in range(2):
        att = first
        if k == 1:
            att = 1 - first
        dfn = 1 - att
        if s.hp[lane, att] <= 0 or s.hp[lane, dfn] <= 0 or flinched == att:
            continue
        st = s.status[lane, att]
        if st == 1:  # paralysis: 25% full stop
            if rand_n(draw(lane, g, c), 4) < 1:
                c += 1
                continue
            c += 1
        elif st == 5:  # freeze: 20% thaw, else no move
            if rand_n(draw(lane, g, c), 5) < 1:
                s.status[lane, att] = 0
            else:
                c += 1
                continue
            c += 1
        act = a0
        if att == 1:
            act = a1
        packed = use_move(s, lane, att, dfn, act, c)
        c = packed / 2
        if packed % 2 == 1:
            flinched = dfn
    for side in range(2):
        st = s.status[lane, side]
        if s.hp[lane, side] > 0:
            if st == 4 or st == 2:  # burn, poison
                s.hp[lane, side] -= wp.max(s.maxhp[lane, side] / 8, 1)
            elif st == 3:  # toxic
                s.hp[lane, side] -= wp.max(s.maxhp[lane, side] * s.toxctr[lane, side] / 16, 1)
                s.toxctr[lane, side] += 1
    s.ctr[lane] = c
    down0 = s.hp[lane, 0] <= 0
    down1 = s.hp[lane, 1] <= 0
    if down0 or down1:
        r = 1
        if down0 and down1:
            r = 3
        elif down0:
            r = 2
        s.result[lane] = r
        init_lane(s, lane, g + 1)  # auto-reset keeps every lane busy, as `step` does


@wp.kernel
def k_init(s: St):
    init_lane(s, wp.tid(), 0)


@wp.kernel
def k_step(s: St):
    advance(s, wp.tid())


@wp.kernel
def k_step_idx(s: St, idx: wp.array(dtype=wp.int32)):
    advance(s, idx[wp.tid()])


@wp.kernel
def k_fork(s: St, src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    """Search-shaped copy: clone a parent slot into a child slot."""
    i = wp.tid()
    a = src[i]
    b = dst[i]
    for side in range(2):
        s.hp[b, side] = s.hp[a, side]
        s.maxhp[b, side] = s.maxhp[a, side]
        s.status[b, side] = s.status[a, side]
        s.toxctr[b, side] = s.toxctr[a, side]
        s.level[b, side] = s.level[a, side]
        for j in range(NSTATS):
            s.stats[b, side, j] = s.stats[a, side, j]
            s.boosts[b, side, j] = s.boosts[a, side, j]
        for j in range(2):
            s.types[b, side, j] = s.types[a, side, j]
        for j in range(4):
            s.mvs[b, side, j] = s.mvs[a, side, j]
    s.ctr[b] = s.ctr[a]
    s.gen[b] = s.gen[a]
    s.result[b] = s.result[a]


@wp.kernel
def k_checksum(s: St, out: wp.array(dtype=wp.uint64)):
    lane = wp.tid()
    h = wp.uint64(0xCBF29CE484222325)
    for side in range(2):
        for w in range(4):
            v = s.hp[lane, side]
            if w == 1:
                v = s.status[lane, side]
            elif w == 2:
                v = s.boosts[lane, side, 0] + 8 * s.boosts[lane, side, 3]
            elif w == 3:
                v = s.toxctr[lane, side]
            h = (h ^ wp.uint64(wp.uint32(v))) * wp.uint64(0x100000001B3)
    h = (h ^ wp.uint64(wp.uint32(s.ctr[lane]))) * wp.uint64(0x100000001B3)
    h = (h ^ wp.uint64(wp.uint32(s.gen[lane]))) * wp.uint64(0x100000001B3)
    out[lane] = h


def make_state(batch: int, device) -> St:
    s = St()
    z2 = lambda: wp.zeros((batch, 2), dtype=wp.int32, device=device)
    s.hp, s.maxhp, s.status, s.toxctr, s.level = z2(), z2(), z2(), z2(), z2()
    s.stats = wp.zeros((batch, 2, NSTATS), dtype=wp.int32, device=device)
    s.boosts = wp.zeros((batch, 2, NSTATS), dtype=wp.int32, device=device)
    s.types = wp.zeros((batch, 2, 2), dtype=wp.int32, device=device)
    s.mvs = wp.zeros((batch, 2, 4), dtype=wp.int32, device=device)
    s.ctr = wp.zeros(batch, dtype=wp.int32, device=device)
    s.gen = wp.zeros(batch, dtype=wp.int32, device=device)
    s.result = wp.zeros(batch, dtype=wp.int32, device=device)
    s.chart = wp.array(np.array(T.CHART, dtype=np.int32), dtype=wp.int32, device=device)
    s.move = wp.array(np.array(T.MOVE_COLS, dtype=np.int32), dtype=wp.int32, device=device)
    wp.launch(k_init, dim=batch, inputs=[s], device=device)
    return s


def checksum(s: St, batch: int, device) -> int:
    out = wp.zeros(batch, dtype=wp.uint64, device=device)
    wp.launch(k_checksum, dim=batch, inputs=[s, out], device=device)
    return int(np.add.reduce(out.numpy(), dtype=np.uint64))
