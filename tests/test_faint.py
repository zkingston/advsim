"""What the replay compare cannot reach: a faint on a hand-set position, a
finished battle stepped again, and the turn limit, which no 100-turn sweep hits.
Replacements and the last faint are compared at every sweep decision point.
"""
import pytest

from advsim.engine import mask as mask_
from conftest import make_env, poke, step

BATCH = 8


@pytest.fixture
def env():
    return make_env(BATCH, seed=4)


def test_a_fainted_active_asks_for_a_replacement(env):
    poke(env, 'hp', (0, 1, 0), 1)       # p2's lead is one hit from fainting
    poke(env, 'status', (0, 1, 0), 2)   # and burned
    step(env, [[0, 0]] * BATCH)
    assert env.numpy('hp')[0, 1, 0] == 0
    assert env.numpy('request')[0, 1] == mask_.REQUEST_SWITCH
    assert env.numpy('alive_mask')[0, 1] == 0b111110, 'the fainted slot must leave the mask'
    assert env.numpy('status')[0, 1, 0] == 0, 'a fainted Pokemon keeps no status'
    assert env.numpy('result')[0] == 0, 'the battle continues while the bench is alive'


def test_a_finished_battle_ignores_further_steps(env):
    poke(env, 'result', 0, 1)
    before = env.numpy('turn')[0]
    step(env, [[0, 0]] * BATCH)
    assert env.numpy('turn')[0] == before


def test_the_turn_limit_ties(env):
    """Turn 1,000 ends the battle, and the check comes before the Quick Claw
    roll, so the turn that ties costs one draw fewer than any other."""
    poke(env, 'turn', 0, 1000)
    before = env.numpy('rng_ctr')[0]
    step(env, [[0, 0]] * BATCH)
    assert env.numpy('result')[0] == 3, 'turn 1,001 is a tie'
    assert env.numpy('result')[1] == 0, 'the other battles carry on'
    after = env.numpy('rng_ctr')[0]
    assert after - before < env.numpy('rng_ctr')[1], 'no Quick Claw roll on the turn that ties'
