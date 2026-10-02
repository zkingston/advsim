"""The engine's Emerald chooser picks what the oracle's picked.

Every position where Showdown asked a side for a replacement, in battles the
oracle played with `policy.js`, is loaded into the engine; the kernel has to
name the Pokemon that came in next.
"""
import pytest
import numpy as np
import warp as wp

from advsim import artifacts, oracle
from advsim.engine import kernels, mask
from advsim.env import Gen3Env


def replacement_points(cases: list[dict]) -> list[tuple[dict, int, int]]:
    """(state, side, the action Showdown's chooser took) at every replacement."""
    out = []
    for c in cases:
        for prev, seg in zip(c['segments'], c['segments'][1:]):
            for side in (0, 1):
                if prev['after']['request'][side] == int(mask.REQUEST_SWITCH):
                    out.append((prev['after'], side, seg['choices'][side]))
    return out


def engine_picks(states: list[dict]) -> np.ndarray:
    env = Gen3Env(batch=len(states), device='cpu', rng_mode='replay')
    env.reset(seed=0)
    env.load_words(states)
    out = wp.zeros((len(states), 2), dtype=wp.int32, device='cpu')
    wp.launch(kernels.emerald_replacements, dim=(len(states), 2), device='cpu', inputs=[env.state, env.dex, out])
    return out.numpy().copy()


@pytest.mark.oracle
def test_engine_matches_oracle():
    cases = oracle.turn_cases(150, seed=11, turns=100, policy='emerald')
    points = replacement_points(cases)
    assert len(points) > 100
    picks = engine_picks([state for state, _, _ in points])
    wrong = [(i, side, int(picks[i, side]), want) for i, (_, side, want) in enumerate(points)
             if picks[i, side] != want]
    assert not wrong, f'{len(wrong)} of {len(points)} differ, first {wrong[:5]}'


@pytest.mark.oracle
def test_single_type_hits_twice():
    """Typhlosion's Fire counts twice, so Corsola scores 0 rather than 2 and
    Section 2 takes over. There the status move counts by type, and the tie
    goes to the earlier slot. Random battles never reached this."""
    ids = artifacts.load_ids()
    dex = artifacts.load_dex()
    sp = {n: ids['species'].index(n) for n in ('typhlosion', 'snorlax', 'corsola')}
    mv = {n: ids['moves'].index(n) for n in ('raindance', 'surf')}
    env = Gen3Env(batch=1, device='cpu', rng_mode='replay')
    env.reset(seed=0)
    w = env.words()
    w['request'][0] = [int(mask.REQUEST_SWITCH), 0]
    w['active'][0] = [0, 0]
    w['hp'][0, 0, 0] = 0
    w['hp'][0, 0, 3:] = 0
    w['types'][0, 0] = [ids['types'].index('Normal'), 0]
    w['types'][0, 1] = [dex['species_type1'][sp['typhlosion']], dex['species_type2'][sp['typhlosion']]]
    w['ability'][0, 1, 0] = 0
    w['last_used'][0] = 0
    for slot, (name, move) in ((1, ('snorlax', 'raindance')), (2, ('corsola', 'surf'))):
        w['species'][0, 0, slot] = sp[name]
        w['moves'][0, 0, slot] = [mv[move], 0, 0, 0]
    for name, arr in w.items():
        env.arrays[name].assign(arr)
    out = wp.zeros((1, 2), dtype=wp.int32, device='cpu')
    wp.launch(kernels.emerald_replacements, dim=(1, 2), device='cpu', inputs=[env.state, env.dex, out])
    assert out.numpy()[0, 0] == int(mask.SWITCH_BASE) + 1
