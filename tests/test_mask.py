"""Legal actions the replay compare rarely reaches: Struggle, the two move
locks, and traps. Every other mask is compared at every sweep decision point."""
import pytest
import warp as wp

from advsim.artifacts import load_ids
from advsim.engine import mask as mask_
from advsim.engine import kernels
from conftest import make_env, poke

BATCH = 8
MOVES, SWITCHES = 0xF, 0x3F0


@pytest.fixture
def env():
    return make_env(BATCH, seed=2)


def masks(env):
    out = wp.zeros((BATCH, 2), dtype=wp.int32, device='cpu')
    wp.launch(kernels.legal_mask, dim=(BATCH, 2), device='cpu', inputs=[env.state, env.dex, out])
    return out.numpy()


def test_struggle_replaces_an_empty_moveset(env):
    for slot in range(4):
        poke(env, 'pp', (0, 0, 0, slot), 0)
    m = masks(env)[0, 0]
    assert m & MOVES == 0
    assert m & (1 << mask_.ACTION_FORCED)
    assert m & SWITCHES, 'a Pokemon out of PP can still switch'


def test_choice_lock_and_encore_pin_one_move(env):
    move_id = int(env.numpy('moves')[0, 0, 0, 2])
    poke(env, 'choice_move', (0, 0), move_id)
    assert masks(env)[0, 0] & MOVES == 0b0100
    poke(env, 'choice_move', (0, 0), 0)
    poke(env, 'encore_move', (0, 0), move_id)
    assert masks(env)[0, 0] & MOVES == 0b0100


def test_traps_remove_switches_but_not_moves(env):
    # A partial trap holds only while the Pokemon that set it is still across
    # the field, which is what trap_source names.
    poke(env, 'trap_turns', (0, 0), 3)
    poke(env, 'trap_source', (0, 0), int(env.numpy('active')[0, 1]) + 1)
    m = masks(env)[0, 0]
    assert m & SWITCHES == 0 and m & MOVES == 0xF
    poke(env, 'trap_source', (0, 0), (int(env.numpy('active')[0, 1]) + 2) % 6 + 1)
    assert masks(env)[0, 0] & SWITCHES != 0, 'the Pokemon that wrapped it is gone'
    poke(env, 'trap_turns', (0, 0), 0)
    # Shadow Tag across the field holds anything in.
    poke(env, 'ability', (0, 1, int(env.numpy('active')[0, 1])),
         load_ids()['abilities'].index('shadowtag'))
    assert masks(env)[0, 0] & SWITCHES == 0


