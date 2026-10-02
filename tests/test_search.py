"""The device search: regret matching solves known matrix games, a search
leaves its roots alone and spends every lane of every iteration, and on CUDA
an iteration captured as one graph does what plain launches do."""
import numpy as np
import pytest
import warp as wp

from advsim import fileio
from advsim.engine import kernels_search
from advsim.search.mcts import Search
from conftest import make_env
from test_env_api import play


def solve(matrices, masks, iters=4000):
    n = len(matrices)
    q = np.zeros((n, 12, 12), dtype=np.float32)
    for i, m in enumerate(matrices):
        m = np.asarray(m, dtype=np.float32)
        q[i, :m.shape[0], :m.shape[1]] = m
    out = wp.zeros((n, 6, 12), dtype=wp.float32, device='cpu')
    wp.launch(kernels_search.solve_matrix, dim=n, device='cpu', inputs=[
        wp.array(q, dtype=wp.float32, device='cpu'), wp.array(np.array(masks, np.int32), dtype=wp.int32, device='cpu'),
        iters, out])
    return out.numpy()[:, :2]


def test_regret_matching_finds_the_equilibria():
    pennies = [[1, -1], [-1, 1]]
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    dominated = [[1, 1], [0, 0]]  # row 0 dominates; the column player is indifferent
    pol = solve([pennies, rps, dominated], [[3, 3], [7, 7], [3, 3]])
    assert np.allclose(pol[0, :, :2], 0.5, atol=0.02)
    assert np.allclose(pol[1, :, :3], 1 / 3, atol=0.02)
    assert pol[2, 0, 0] > 0.98
    assert np.allclose(pol.sum(-1), 1, atol=1e-4)


def needs_setdist():
    if not (fileio.ARTIFACTS / 'setdist.npz').exists():
        pytest.skip('artifacts/setdist.npz missing; run `advsim build`')


@pytest.mark.parametrize('hidden', [False, True])
def test_a_search_spends_its_lanes_and_leaves_the_roots_alone(hidden):
    needs_setdist()
    games, lanes, iters, cap = 6, 4, 12, 4
    env = make_env(games + cap * lanes, seed=21)
    play(env, 8)
    before = env.hash()[:games]
    search = Search(env, cap, lane_base=games, lanes=lanes, nodes=256, depth=3, rollout=4, hidden=hidden, key=5)
    # Roots anywhere in the arena, each tree for its own side, in waves.
    roots, sides = np.array([5, 1, 3, 0]), np.array([1, 0, 1, 0])
    search.run(roots, sides, iters)
    assert (env.hash()[:games] == before).all(), 'a search wrote to its roots'
    live = env.numpy('result')[roots] == 0
    assert (search.visits(4).sum(1)[live] == iters * lanes).all()
    assert (search.tree.count.numpy()[:4][live] > 1).all()
    pol, value = search.policy(4)
    assert np.allclose(pol[live].sum(-1), 1, atol=1e-4)
    assert (np.abs(value) <= 1).all()
    search.run(roots[:2], sides[:2], iters)  # a smaller wave: the rest sit out
    assert (search.visits(4)[2:] == 0).all()
    assert not env.numpy('err')[games:].any()
    legal = np.full(6, 0b11, np.int32)
    acts = search.decide(np.arange(6), np.zeros(6, np.int32), legal, 4, np.random.default_rng(0))
    assert set(acts.tolist()) <= {0, 1}


@pytest.mark.skipif(not wp.is_cuda_available(), reason='needs a GPU')
def test_a_graphed_iteration_is_the_plain_one():
    """Over two waves, the second under a new key: a captured graph must read
    the key at run time, not keep the one it was captured with."""
    needs_setdist()
    from advsim.env import Gen3Env
    runs = []
    for graph in (False, True):
        env = Gen3Env(8 * 2, device='cuda:0')
        env.reset(seed=4)
        search = Search(env, 8, lane_base=8, lanes=1, nodes=512, depth=4, rollout=8, key=3)
        run = []
        for key in (3, 11):
            search.key = key
            search.run(np.arange(8), np.arange(8) % 2, 40, graph=graph)
            run += [search.tree.cell_n.numpy().copy(), search.tree.cell_w.numpy().copy(),
                    search.tree.regret.numpy().copy()]
        runs.append(run)
    for a, b in zip(*runs):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('kind', ['mlp', 'v2'])
def test_a_value_network_scores_the_leaves(kind):
    """With a network, leaves are its (V(p1) - V(p2)) / 2; the search still
    spends every lane, and the root values differ from the rollout search's."""
    torch = pytest.importorskip('torch')
    needs_setdist()
    from advsim.net import Policy, PolicyV2
    torch.manual_seed(0)
    net = Policy(hidden=32) if kind == 'mlp' else PolicyV2(width=32, d=16)
    games, lanes, cap = 4, 4, 4
    env = make_env(games + cap * lanes, seed=22)
    play(env, 6)
    search = Search(env, cap, lane_base=games, lanes=lanes, nodes=128, depth=2, rollout=4, key=1)
    roots, sides = np.arange(4), np.array([0, 1, 0, 1])
    search.run(roots, sides, 8, net=net)
    live = env.numpy('result')[roots] == 0
    assert (search.visits(4).sum(1)[live] == 8 * lanes).all()
    _, with_net = search.policy(4)
    values = search.tree.value.numpy()
    assert (np.abs(values) <= 1).all()
    search.run(roots, sides, 8)
    _, rollouts = search.policy(4)
    assert not np.allclose(with_net[live], rollouts[live])


def test_the_policy_is_the_prior():
    """With a prior, every root and every node a descent made holds both
    players' policies, normalised over what was legal there; and a fresh
    node's strategy is its prior."""
    torch = pytest.importorskip('torch')
    needs_setdist()
    from advsim.net import PolicyV2
    torch.manual_seed(0)
    net = PolicyV2(width=32, d=16).eval()
    games, lanes, cap = 4, 4, 4
    env = make_env(games + cap * lanes, seed=23)
    play(env, 6)
    search = Search(env, cap, lane_base=games, lanes=lanes, nodes=128, depth=2, key=1)
    roots = np.arange(4)
    search.run(roots, np.array([0, 1, 0, 1]), 6, net=net, prior=True)
    prior = search.tree.prior.numpy()
    count = search.tree.count.numpy()
    live = env.numpy('result')[roots] == 0
    for tree in np.nonzero(live)[0]:
        root = tree * 128
        assert np.allclose(prior[root].sum(-1), 1, atol=1e-4), 'root prior'
        made = prior[root + 1:root + count[tree]].sum(-1)
        assert (np.isclose(made, 1, atol=1e-4) | (made == 0)).all()
        assert np.isclose(made, 1, atol=1e-4).any(), 'no node got a prior'
    assert (search.visits(4).sum(1)[live] == 6 * lanes).all()


@pytest.mark.parametrize('final', ['eq', 'mix'])
def test_decide_spans_waves(final):
    """More roots than the capacity: every wave searches, every root gets a
    legal action, whatever the final move is made of."""
    torch = pytest.importorskip('torch')
    needs_setdist()
    from advsim.net import PolicyV2
    torch.manual_seed(0)
    games, lanes, cap = 10, 2, 4
    env = make_env(games + cap * lanes, seed=24)
    play(env, 4)
    search = Search(env, cap, lane_base=games, lanes=lanes, nodes=64, depth=2, key=2)
    legal = np.full(games, 0b111, np.int32)
    acts = search.decide(np.arange(games), np.arange(games) % 2, legal, 3, np.random.default_rng(0),
                         net=PolicyV2(width=32, d=16).eval(), prior=True, temperature=0.25, final=final)
    assert len(acts) == games and set(acts.tolist()) <= {0, 1, 2}


def favour_first(torch):
    """A stand-in network: each side's first legal action gets ~90% of the policy; value 0."""
    class Favour(torch.nn.Module):
        def forward(self, obs, legal):
            first = torch.nn.functional.one_hot(legal.float().argmax(-1), 12).bool()
            logits = torch.where(legal, 0.0, -1e9) + 5.0 * first
            return torch.distributions.Categorical(logits=logits), torch.zeros(len(obs))

        def value(self, obs):
            return torch.zeros(len(obs))
    return Favour()


def test_pruning_keeps_only_the_favoured_action():
    torch = pytest.importorskip('torch')
    needs_setdist()
    games, lanes, cap, iters = 4, 4, 4, 8
    env = make_env(games + cap * lanes, seed=25)
    play(env, 6)
    search = Search(env, cap, lane_base=games, lanes=lanes, nodes=128, depth=2, key=3)
    roots, sides = np.arange(4), np.array([0, 1, 0, 1])
    live = env.numpy('result')[roots] == 0
    net = favour_first(torch)
    search.run(roots, sides, iters, net=net, prior=True)
    assert (search.visits(4).sum(1)[live] == iters * lanes).all()
    spread = (search.visits(4) > 0).sum(1)
    search.tree.knobs.assign(np.array([0.1, 0.5], np.float32))  # prune: only the favoured action
    search.run(roots, sides, iters, net=net, prior=True)
    assert ((search.visits(4) > 0).sum(1)[live] == 1).all() and (spread[live] > 1).any()
