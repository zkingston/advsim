"""Allocation and reset: the state a battle starts from."""
import numpy as np
import pytest
import warp as wp

from advsim.env import Gen3Env
from advsim.engine import layout
from conftest import make_env

BATCH = 256


@pytest.fixture(scope='module')
def env():
    return make_env(BATCH, seed=11)


def test_allocation_follows_the_layout(env):
    for f in layout.FIELDS:
        assert env.arrays[f.name].shape == f.shape(BATCH), f.name
    assert env.bytes_allocated() // BATCH == layout.bytes_per_battle()


def test_every_battle_starts_at_full_health(env):
    hp, maxhp = env.numpy('hp'), env.numpy('maxhp')
    assert (hp == maxhp).all()
    assert (maxhp > 0).all()
    assert (env.numpy('alive_mask') == 63).all()


def test_teams_are_dealt_without_repeats_inside_a_batch(env):
    species = env.numpy('species').reshape(BATCH * 2, 6)
    assert len(np.unique(species, axis=0)) == BATCH * 2
    assert species.min() >= 1


def test_volatile_and_field_state_starts_zero(env):
    for name in ('boosts', 'vflags', 'sub_hp', 'confusion_turns', 'stall_ctr',
                 'weather', 'weather_turns', 'turn', 'result', 'err', 'spikes', 'status'):
        assert not env.numpy(name).any(), f'{name} must be zero at reset'


def test_the_lead_has_its_species_types(env):
    """Types live in the active slot, so reset has to fill them from the species."""
    types = env.numpy('types')
    assert (types[:, :, 0] > 0).all(), 'a type ID of 0 is the ??? type, not a real lead type'
    assert types.shape[-1] == 2


def test_pp_starts_at_max_and_pads(env):
    pp, max_pp, moves = env.numpy('pp'), env.numpy('max_pp'), env.numpy('moves')
    assert np.array_equal(pp, max_pp)
    assert np.array_equal(moves == 0, pp == 0)


def test_rng_keys_differ_per_battle(env):
    keys = env.numpy('rng_key')
    assert len(np.unique(keys)) > BATCH * 0.99
    assert not env.numpy('rng_ctr').any()


@pytest.mark.built
def test_reset_is_reproducible_for_a_seed():
    a, b = Gen3Env(batch=64, device='cpu'), Gen3Env(batch=64, device='cpu')
    a.reset(seed=5)
    b.reset(seed=5)
    for name in ('species', 'hp', 'rng_key', 'stats'):
        assert np.array_equal(a.numpy(name), b.numpy(name)), name


@pytest.mark.built
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason='no CUDA device')
def test_cpu_and_gpu_reset_agree():
    cpu, gpu = Gen3Env(batch=64, device='cpu'), Gen3Env(batch=64, device='cuda:0')
    cpu.reset(seed=3)
    gpu.reset(seed=3)
    for name in ('species', 'hp', 'stats', 'moves', 'rng_key'):
        assert np.array_equal(cpu.numpy(name), gpu.numpy(name)), name
