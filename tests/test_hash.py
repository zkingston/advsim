"""The canonical hash: the Warp and NumPy sides must agree word for word.

They are the two halves of replay diffing — the engine hashes its own state on
the device, the mirror hashes what `export_state.js` says Showdown holds — so a
disagreement here would read as a parity bug everywhere else.
"""
import numpy as np
import pytest

from advsim import statehash
from advsim.engine import layout
from conftest import make_env, poke, step

BATCH = 8


@pytest.fixture
def env():
    return make_env(BATCH, seed=11)


def test_the_word_order_covers_every_field_once():
    offsets = layout.word_offsets()
    assert sorted(offsets.values()) == list(offsets.values()), 'offsets must follow declaration order'
    at = 0
    for f in layout.FIELDS:
        assert offsets[f.name] == at, f.name
        at += f.elems()
    assert at == layout.N_WORDS


def test_the_masks_are_what_the_spec_says():
    full, position = {f.name for f in layout.FULL}, {f.name for f in layout.POSITION}
    assert 'err' not in full and len(full) == len(layout.FIELDS) - 1
    for name in ('turn', 'rng_key', 'rng_ctr', 'rng_mode', 'revealed', 'err'):
        assert name not in position, name
    assert position < full


def test_warp_matches_the_numpy_mirror(env):
    for turn in range(4):
        words = env.words()
        assert np.array_equal(env.hash(), statehash.state_hash(words)), f'full hash, turn {turn}'
        assert np.array_equal(env.hash(position=True),
                              statehash.state_hash(words, position=True)), f'position hash, turn {turn}'
        step(env, [[turn % 4, (turn + 1) % 4]] * BATCH)


def test_battles_in_different_states_hash_differently(env):
    assert len(set(env.hash().tolist())) == BATCH, 'eight different teams, eight hashes'
    before = env.hash()
    step(env, [[0, 0]] * BATCH)
    assert not np.array_equal(before, env.hash())


def test_the_position_hash_ignores_the_rng_and_the_turn(env):
    full, position = env.hash(), env.hash(position=True)
    poke(env, 'turn', 0, 300)
    poke(env, 'rng_ctr', 0, 77)
    poke(env, 'revealed', (0, 0, 0), 0xFFFF)
    assert env.hash(position=True)[0] == position[0], 'none of those change the position'
    assert env.hash()[0] != full[0], 'but the full hash sees them'


def test_a_changed_field_changes_the_hash(env):
    """Every hashed field, one word each: a silent word is a word the hash forgets."""
    offsets = layout.word_offsets()
    for f in layout.FULL:
        before = env.hash()[0]
        flat = env.arrays[f.name].numpy().copy()
        view = flat.reshape(BATCH, -1)
        old = int(view[0, 0])
        view[0, 0] = old + 1 if old < 3 else old - 1
        env.arrays[f.name].assign(view.reshape(flat.shape))
        assert env.hash()[0] != before, f'{f.name} word {offsets[f.name]} is not in the hash'
        view[0, 0] = old
        env.arrays[f.name].assign(view.reshape(flat.shape))
        assert env.hash()[0] == before, f'{f.name} did not restore'


def test_a_single_word_updates_the_hash_in_place(env):
    """The Zobrist property search relies on: h ^= f(i, old) ^ f(i, new)."""
    words = env.words()
    before = statehash.state_hash(words)[0]
    i = layout.word_offsets()['sub_hp']
    old, new = int(words['sub_hp'][0, 0]), 61
    words['sub_hp'][0, 0] = new
    incremental = before ^ statehash.splitmix64(np.uint64((i << 32) | old)) \
        ^ statehash.splitmix64(np.uint64((i << 32) | new))
    assert statehash.state_hash(words)[0] == incremental
