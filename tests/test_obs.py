"""The observation kernel: what each player sees, and what it must not."""
import numpy as np
import warp as wp

from advsim.engine import kernels, obs_layout as L
from conftest import make_env, poke, step

BATCH = 8
MON = len(L.MON)


def observe(env):
    out = wp.zeros((env.batch, 2, L.OBS_DIM), dtype=wp.int16, device='cpu')
    wp.launch(kernels.obs, dim=(env.batch, 2), device='cpu', inputs=[env.state, env.dex, out])
    return out.numpy().copy()


def col(token, name):
    return token * MON + L.MON.index(name)


def test_the_foe_bench_is_hidden_and_the_own_party_is_not():
    env = make_env(BATCH, seed=5)
    o = observe(env)
    for p in (0, 1):
        own_species = [o[0, p, col(t, 'species')] for t in range(6)]
        assert all(own_species), 'the own party is always in view'
        assert o[0, p, col(6, 'present')] == 1, 'the foe lead switched in, so it is seen'
        assert o[0, p, col(6, 'active')] == 1
        assert not o[0, p, 7 * MON:12 * MON].any(), 'an unseen foe Pokemon is all zeros'
        assert o[0, p, col(6, 'hp')] == 0 and o[0, p, col(6, 'ability_known')] == 0


def test_the_mask_block_is_the_legal_mask():
    env = make_env(BATCH, seed=5)
    o = observe(env)
    legal = wp.zeros((BATCH, 2), dtype=wp.int32, device='cpu')
    wp.launch(kernels.legal_mask, dim=(BATCH, 2), device='cpu', inputs=[env.state, env.dex, legal])
    bits = (legal.numpy()[..., None] >> np.arange(12)) & 1
    assert (o[:, :, L.MASK_BASE:L.MASK_BASE + 12] == bits).all()


def test_foe_hp_is_showdowns_percentage():
    env = make_env(BATCH, seed=5)
    maxhp = int(env.numpy('maxhp')[0, 1, 0])
    for hp, want in ((maxhp, 100), (maxhp - 1, 99), (1, 1), (0, 0)):
        poke(env, 'hp', (0, 1, 0), hp)
        assert observe(env)[0, 0, col(6, 'hp_pct')] == want


def test_a_move_the_foe_used_is_revealed():
    env = make_env(BATCH, seed=5)
    step(env, [[0, 0]] * BATCH)
    o = observe(env)
    moves = env.numpy('moves')
    for b in range(BATCH):
        if env.numpy('last_move')[b, 1] == moves[b, 1, 0, 0]:
            assert o[b, 0, col(6, 'move0')] == moves[b, 1, 0, 0]
