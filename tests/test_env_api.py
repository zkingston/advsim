"""The training API: observe, step with auto-reset, rewards, the step cap, and
a captured CUDA graph that must do exactly what plain launches do."""
import numpy as np
import pytest
import warp as wp

from advsim.env import Gen3Env
from conftest import make_env

BATCH = 32


def as_np(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def random_legal(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniformly random legal action per battle per side."""
    out = np.zeros(mask.shape, dtype=np.int32)
    for idx in np.ndindex(mask.shape):
        choices = [a for a in range(12) if (mask[idx] >> a) & 1]
        out[idx] = rng.choice(choices)
    return out


def play(env, steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    ends, rewards = 0, []
    for _ in range(steps):
        _, mask = env.observe()
        env.step(random_legal(as_np(mask), rng))
        assert not as_np(env.err).any(), 'a legal action set an error bit'
        done = as_np(env.done).astype(bool)
        ends += int(done.sum())
        rewards += as_np(env.reward)[done].tolist()
    return ends, rewards


def test_battles_end_score_and_are_dealt_again():
    env = make_env(BATCH, seed=3)
    ends, rewards = play(env, 400)
    assert ends > 0, 'some battle should finish in 400 decisions'
    assert set(rewards) <= {-1.0, 0.0, 1.0}
    assert (as_np(env.reward)[~as_np(env.done).astype(bool)] == 0).all(), 'reward only at a game end'


def test_the_step_cap_truncates_and_deals_again():
    env = make_env(BATCH, seed=3)
    env.max_steps = 5
    rng = np.random.default_rng(1)
    for _ in range(5):
        _, mask = env.observe()
        env.step(random_legal(as_np(mask), rng))
    trunc = as_np(env.truncated).astype(bool)
    assert trunc.sum() + as_np(env.done).sum() > 0
    assert (env.numpy('turn')[trunc] == 0).all(), 'a truncated battle starts over'


def test_needs_decision_is_false_only_where_a_side_can_only_pass():
    env = make_env(BATCH, seed=3)
    _, mask = env.observe()
    decide = as_np(env.needs_decision)
    assert (decide == (as_np(mask) != (1 << 10))).all()


@pytest.mark.skipif(not wp.is_cuda_available(), reason='needs a GPU')
def test_a_captured_graph_steps_exactly_like_plain_launches():
    torch = pytest.importorskip('torch')
    runs = []
    for use_graph in (False, True):
        env = Gen3Env(BATCH, device='cuda:0')
        env.reset(seed=9)
        if use_graph:
            env.capture()
            env.reset(seed=9)  # capture ran a step; start over
        rng = np.random.default_rng(4)
        hashes = []
        for _ in range(60):
            _, mask = env.observe()
            acts = torch.as_tensor(random_legal(as_np(mask), rng), device='cuda:0')
            if use_graph:
                wp.to_torch(env.actions).copy_(acts)
                env.step_graph()
            else:
                env.step(acts)
            hashes.append(env.hash())
        runs.append(np.array(hashes))
    assert (runs[0] == runs[1]).all()
