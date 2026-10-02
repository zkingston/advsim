"""The leak test (SPEC §Determinization): scramble everything `layout.py` marks
hidden from one side, and that side's observation must not move, and
`determinize` under one key must give the same world for both states. Worlds
must also show that side exactly what the true state does, and play on.

The scrambler reads only `layout.hidden` and the `revealed` bits, never
`determinize.py`, so a field the kernel forgets is caught here.
"""
import numpy as np
import pytest

from advsim import artifacts, fileio
from advsim.engine import layout, obs_layout
from conftest import make_env
from test_env_api import as_np, play, random_legal

B = 24
TRANSFORMED = 1 << 11
SLP = 18


def hp_pct(hp, maxhp):
    if hp <= 0 or maxhp <= 0:
        return 0
    pct = (100 * hp + maxhp - 1) // maxhp
    return 99 if pct == 100 and hp < maxhp else pct


def same_pct_hp(hp, maxhp, rng):
    """A new (hp, maxhp) that shows the same percent."""
    pct = hp_pct(hp, maxhp)
    if pct in (0, 100):
        new_max = int(rng.integers(150, 400))
        return (0 if pct == 0 else new_max), new_max
    while True:
        new_max = int(rng.integers(150, 400))
        options = [h for h in range(1, new_max) if hp_pct(h, new_max) == pct]
        if options:
            return int(rng.choice(options)), new_max


def scramble(w: dict, src: int, dst: int, side: int, rng) -> None:
    """Battle dst becomes battle src with every field hidden from `side` changed."""
    for name in w:
        w[name][dst] = w[name][src]
    foe = 1 - side
    perm = rng.permutation(6)
    for f in layout.FIELDS:
        if f.scope == layout.MON:
            w[f.name][dst, foe, perm] = w[f.name][src, foe].copy()
    remap = lambda bits: sum(1 << int(perm[i]) for i in range(6) if bits >> i & 1)
    w['active'][dst, foe] = perm[w['active'][src, foe]]
    for name in ('alive_mask', 'knocked_mask', 'truant_mask'):
        w[name][dst, foe] = remap(int(w[name][src, foe]))
    if w['trap_source'][src, side]:
        w['trap_source'][dst, side] = perm[w['trap_source'][src, side] - 1] + 1

    active = int(w['active'][dst, foe])
    xf = bool(int(w['vflags'][dst, foe]) & TRANSFORMED)
    w['party_pos'][dst, foe] = rng.permutation(6)
    for j in range(6):
        rev = int(w['revealed'][dst, foe, j])
        seen = rev & 1
        if not seen:
            for f in layout.FIELDS:
                if f.scope == layout.MON and f.hidden in ('unseen', 'reveal', 'foe'):
                    w[f.name][dst, foe, j] = rng.integers(1, 100, size=w[f.name][dst, foe, j].shape)
            continue
        hp, maxhp = same_pct_hp(int(w['hp'][dst, foe, j]), int(w['maxhp'][dst, foe, j]), rng)
        w['hp'][dst, foe, j], w['maxhp'][dst, foe, j] = hp, maxhp
        own = xf and j == active
        if not own:  # a copy's live stats are the viewer's own
            w['stats'][dst, foe, j] = rng.integers(20, 400, size=5)
            w['hp_type'][dst, foe, j] = rng.integers(1, 17)
            w['max_pp'][dst, foe, j] = rng.integers(1, 64, size=4)
        w['cached_spe'][dst, foe, j] = rng.integers(20, 400)
        moves, pp = ('xf_moves', 'xf_pp') if own else ('moves', 'pp')
        for k in range(4):
            if not rev >> (1 + k) & 1:
                at = (dst, foe, k) if own else (dst, foe, j, k)
                w[moves][at] = rng.integers(1, 114)
                w[pp][at] = rng.integers(1, 64)
        if not rev & (1 << 5) and not (j == active and w['base_ability'][dst, foe]):
            if own:
                w['xf_ability'][dst, foe] = rng.integers(1, 71)
            else:
                w['ability'][dst, foe, j] = rng.integers(1, 71)
        if not rev & (1 << 6):
            w['item'][dst, foe, j] = rng.integers(0, 14)
            if j == active:
                w['choice_move'][dst, foe] = rng.integers(0, 114)
    if xf:
        w['xf_stats'][dst, foe] = rng.integers(20, 400, size=5)
        w['xf_max_pp'][dst, foe] = rng.integers(1, 64, size=4)
        w['xf_hp_type'][dst, foe] = rng.integers(1, 17)
    if w['sub_hp'][dst, foe]:
        w['sub_hp'][dst, foe] = rng.integers(1, 100)
    w['dmg_taken'][dst, foe] = rng.integers(0, 300)

    w['rng_key'][dst] = rng.integers(0, 2 ** 32, dtype=np.uint64)
    w['rng_ctr'][dst] = rng.integers(0, 1000)
    for p in range(2):
        for name in ('confusion_turns', 'encore_turns', 'trap_turns'):
            if w[name][dst, p]:
                w[name][dst, p] = rng.integers(1, 9)
        for j in range(6):
            if w['status'][dst, p, j] == SLP and w['slept_by_foe'][dst, p, j]:
                w['status_ctr'][dst, p, j] = rng.integers(1, 7)


def inject(env) -> None:
    """Hidden durations random play rarely leaves standing at a decision: a
    confusion, a partial trap and a foe-caused sleep, spread over the battles."""
    w = env.words()
    for b in range(B):
        for side in range(2):
            active = int(w['active'][b, side])
            if w['hp'][b, side, active] == 0 or w['result'][b]:
                continue
            if b % 3 == 0:
                w['confusion_turns'][b, side] = 3
            elif b % 3 == 1 and not w['trap_turns'][b, side]:
                w['trap_turns'][b, side] = 3
                w['trap_source'][b, side] = w['active'][b, 1 - side] + 1
            elif not w['status'][b, side, active]:
                w['status'][b, side, active] = SLP
                w['status_ctr'][b, side, active] = 3
                w['slept_by_foe'][b, side, active] = 1
    # A transformed Pokemon and a traced one: their own sets sit where the
    # engine keeps them while the live fields carry the copy.
    trace = artifacts.load_ids()['abilities'].index('trace')
    for b in range(0, B, 4):
        side, other = b // 4 % 2, 1 - b // 4 % 2
        me, them = int(w['active'][b, side]), int(w['active'][b, other])
        if w['result'][b] or not w['hp'][b, side, me] or w['vflags'][b, side] & TRANSFORMED:
            continue
        if b % 8 == 0:
            w['vflags'][b, side] |= TRANSFORMED
            w['xf_species'][b, side] = w['species'][b, side, me]
            w['xf_ability'][b, side] = w['ability'][b, side, me]
            w['xf_hp_type'][b, side] = w['hp_type'][b, side, me]
            w['xf_stats'][b, side] = w['stats'][b, side, me]
            w['xf_moves'][b, side] = w['moves'][b, side, me]
            w['xf_pp'][b, side] = w['pp'][b, side, me]
            w['xf_max_pp'][b, side] = w['max_pp'][b, side, me]
            for name in ('species', 'ability', 'hp_type', 'stats', 'moves'):
                w[name][b, side, me] = w[name][b, other, them]
            w['pp'][b, side, me] = np.where(w['moves'][b, side, me] > 0, 5, 0)
            w['max_pp'][b, side, me] = w['pp'][b, side, me]
            w['types'][b, side] = w['types'][b, other]
        elif not w['base_ability'][b, side]:
            w['base_ability'][b, side] = trace + 1
            w['ability'][b, side, me] = w['ability'][b, other, them]
            w['revealed'][b, side, me] |= 1 << 5
    for name, arr in w.items():
        env.arrays[name].assign(arr)


def test_the_hidden_kinds_are_the_known_ones():
    assert {f.hidden for f in layout.FIELDS} == {'', 'unseen', 'reveal', 'foe', 'both', 'party'}


@pytest.mark.parametrize('steps', [10, 40, 120])
def test_determinize_sees_only_what_the_side_sees(steps):
    if not (fileio.ARTIFACTS / 'setdist.npz').exists():
        pytest.skip('artifacts/setdist.npz missing; run `advsim build`')
    env = make_env(3 * B, seed=11)
    play(env, steps, seed=steps)
    rng = np.random.default_rng(steps)
    # Every column but the legal mask, which can hang on a hidden trap.
    keep = np.r_[0:obs_layout.MASK_BASE, obs_layout.HISTORY_BASE:obs_layout.OBS_DIM]
    inject(env)
    for side in range(2):
        w = env.words()
        for b in range(B):
            scramble(w, b, B + b, side, rng)
        for name, arr in w.items():
            env.arrays[name].assign(arr)
        obs = as_np(env.observe()[0]).copy()
        assert (obs[:B, side][:, keep] == obs[B:2 * B, side][:, keep]).all(), 'the scrambler moved a visible field'

        # Into the same slots: a world's draws are a function of its slot.
        worlds = list(range(2 * B, 3 * B))
        env.determinize(list(range(B, 2 * B)), worlds, side, key=77)
        scrambled = env.words()
        env.determinize(list(range(B)), worlds, side, key=77)
        got = env.words()
        for name, arr in got.items():
            diff = np.nonzero((arr[2 * B:3 * B] != scrambled[name][2 * B:3 * B]).reshape(B, -1).any(1))[0]
            assert not len(diff), f'{name} leaks in battles {diff.tolist()} (side {side})'
        worlds = as_np(env.observe()[0])
        bad = np.nonzero((worlds[2 * B:3 * B, side][:, keep] != obs[:B, side][:, keep]).any(1))[0]
        if len(bad):
            names = obs_layout.names()
            b = int(bad[0])
            cols = keep[np.nonzero(worlds[2 * B + b, side][keep] != obs[b, side][keep])[0]]
            detail = [(names[c], int(obs[b, side, c]), int(worlds[2 * B + b, side, c])) for c in cols]
            raise AssertionError(f'a world looks different to side {side}: battles {bad.tolist()}; {detail}')
        assert not got['err'][2 * B:].any()

    # Worlds are real battles: they play on without an error bit.
    for _ in range(30):
        env.step(random_legal(as_np(env.observe()[1]), rng))
        assert not as_np(env.err)[2 * B:].any()


def test_worlds_differ_in_what_is_hidden():
    if not (fileio.ARTIFACTS / 'setdist.npz').exists():
        pytest.skip('artifacts/setdist.npz missing; run `advsim build`')
    env = make_env(3 * B, seed=12)
    play(env, 20)
    env.determinize(list(range(B)), list(range(B, 2 * B)), 0, key=1)
    env.determinize(list(range(B)), list(range(2 * B, 3 * B)), 0, key=2)
    w = env.words()
    assert (w['species'][B:2 * B, 1] != w['species'][2 * B:, 1]).any()
    assert (w['species'][B:2 * B, 0] == w['species'][2 * B:, 0]).all(), 'own side untouched'
