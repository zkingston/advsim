"""PolicyTF: reads real observations, its switch logits follow the Pokemon
they send in, its prediction reaches the policy only with `react`,
checkpoints load back as what they were, the damage inputs compile without
a graph break, and a trained network widens to read them unchanged."""
import pytest

torch = pytest.importorskip('torch')
from advsim import net as net_  # noqa: E402
from advsim.engine import obs_layout as L  # noqa: E402
from advsim.net_tf import N_PRED, PolicyTF, with_damage  # noqa: E402
from test_net import observations  # noqa: E402


@pytest.mark.parametrize('react', [False, True])
def test_it_reads_observations(react):
    torch.manual_seed(0)
    obs, mask = observations()
    net = PolicyTF(d=32, layers=2, heads=2, react=react).eval()
    legal = net_.legal_bits(mask)
    with torch.no_grad():
        dist, v, extras = net(obs, legal, opp_action=True)
    assert dist.probs.shape == (len(obs), 12) and v.shape == (len(obs),)
    assert (dist.probs[~legal] < 1e-6).all() and torch.isfinite(v).all()
    assert (extras['opp_action'] is None) != react
    if react:
        assert extras['opp_action'].shape == (len(obs), N_PRED)
    assert torch.allclose(net.value(obs), v, atol=1e-5)


@pytest.mark.parametrize('react', [False, True])
def test_switch_logits_follow_the_pokemon(react):
    torch.manual_seed(1)
    obs, _ = observations()
    net = PolicyTF(d=32, layers=2, heads=2, react=react).eval()
    M = len(L.MON)
    active = obs[:, L.MON.index('active'):6 * M:M]
    rows = torch.nonzero((active[:, 2] == 0) & (active[:, 3] == 0)).squeeze(-1)[:8]
    swapped = obs[rows].clone()
    a, b = swapped[:, 2 * M:3 * M].clone(), swapped[:, 3 * M:4 * M].clone()
    swapped[:, 2 * M:3 * M], swapped[:, 3 * M:4 * M] = b, a
    everything = torch.ones(len(rows), 12, dtype=torch.bool)
    with torch.no_grad():
        before = net(obs[rows], everything)[0].logits
        after = net(swapped, everything)[0].logits
    perm = list(range(12))
    perm[6], perm[7] = 7, 6
    assert torch.allclose(before[:, perm], after, atol=1e-4)


def test_the_prediction_feeds_the_policy_only_when_reacting():
    torch.manual_seed(2)
    obs, mask = observations(8, 2)
    legal = net_.legal_bits(mask)
    for react in (False, True):
        net = PolicyTF(d=32, layers=2, heads=2, react=react).eval()
        with torch.no_grad():
            before = net(obs, legal)[0].logits
            if react:
                net.opp_action[1].bias[:5] += 10.0  # a different prediction
            after = net(obs, legal)[0].logits
        assert torch.allclose(before, after) != react


def test_checkpoints_round_trip(tmp_path):
    obs, mask = observations(8, 2)
    for react in (False, True):
        net = PolicyTF(d=32, layers=3, heads=2, react=react).eval()
        path = tmp_path / f'tf{react}.npz'
        net_.save(net, path)
        back = net_.load(path, 'cpu')
        assert type(back) is PolicyTF and back.config == net.config
        with torch.no_grad():
            assert torch.allclose(net(obs, net_.legal_bits(mask))[0].logits,
                                  back(obs, net_.legal_bits(mask))[0].logits)


def reference_block(block, x, keep):
    """The first Block: fused attention with a boolean mask, the per-type
    feed-forward on slices. Faster versions must compute the same thing."""
    import torch.nn.functional as F
    from advsim.net_tf import GROUPS
    n, T, d = x.shape
    q, k, v = block.qkv(block.n1(x)).view(n, T, 3, block.heads, d // block.heads).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
    x = x + block.out(a.transpose(1, 2).reshape(n, T, d))
    h = block.n2(x)
    return x + torch.cat([ff(h[:, g]) for ff, g in zip(block.ff, GROUPS)], 1)


def test_the_block_matches_the_reference():
    from advsim.net_tf import Block
    torch.manual_seed(3)
    block = Block(32, 4).eval()
    x = torch.randn(16, 25, 32)
    keep = torch.rand(16, 25) > 0.3
    keep[:, 23] = True  # the summary always takes part
    with torch.no_grad():
        assert torch.allclose(block(x, keep), reference_block(block, x, keep), atol=1e-5)


def test_the_compact_layout_reads_points_and_round_trips(tmp_path):
    torch.manual_seed(4)
    obs, mask = observations()
    legal = net_.legal_bits(mask)
    net = PolicyTF(d=32, layers=2, heads=2, react=True, compact=True).eval()
    assert net.tokens(obs)[0].shape[1] == 17
    with torch.no_grad():
        dist, v, extras = net(obs, legal, opp_action=True)
    assert dist.probs.shape == (len(obs), 12) and (dist.probs[~legal] < 1e-6).all()
    assert extras['opp_action'].shape == (len(obs), N_PRED)
    M = len(L.MON)
    active = obs[:, L.MON.index('active'):6 * M:M]
    rows = torch.nonzero((active[:, 2] == 0) & (active[:, 3] == 0)).squeeze(-1)[:8]
    swapped = obs[rows].clone()
    a, b = swapped[:, 2 * M:3 * M].clone(), swapped[:, 3 * M:4 * M].clone()
    swapped[:, 2 * M:3 * M], swapped[:, 3 * M:4 * M] = b, a
    everything = torch.ones(len(rows), 12, dtype=torch.bool)
    with torch.no_grad():
        before, after = net(obs[rows], everything)[0].logits, net(swapped, everything)[0].logits
    perm = list(range(12))
    perm[6], perm[7] = 7, 6
    assert torch.allclose(before[:, perm], after, atol=1e-4)
    net_.save(net, tmp_path / 'compact.npz')
    back = net_.load(tmp_path / 'compact.npz', 'cpu')
    assert back.compact and back.config == net.config
    with torch.no_grad():
        assert torch.allclose(net(obs, legal)[0].logits, back(obs, legal)[0].logits)


def test_damage_inputs_read_point_and_round_trip(tmp_path):
    torch.manual_seed(5)
    obs, mask = observations()
    legal = net_.legal_bits(mask)
    for compact in (False, True):
        net = PolicyTF(d=32, layers=2, heads=2, react=True, compact=compact, damage=True).eval()
        with torch.no_grad():
            dist, v = net(obs, legal)
        assert torch.isfinite(dist.logits[legal]).all() and torch.isfinite(v).all()
        net_.save(net, tmp_path / f'd{compact}.npz')
        back = net_.load(tmp_path / f'd{compact}.npz', 'cpu')
        assert back.damage and back.config == net.config
        with torch.no_grad():
            assert torch.allclose(net(obs, legal)[0].logits, back(obs, legal)[0].logits)
    M = len(L.MON)
    active = obs[:, L.MON.index('active'):6 * M:M]
    rows = torch.nonzero((active[:, 2] == 0) & (active[:, 3] == 0)).squeeze(-1)[:8]
    swapped = obs[rows].clone()
    a, b = swapped[:, 2 * M:3 * M].clone(), swapped[:, 3 * M:4 * M].clone()
    swapped[:, 2 * M:3 * M], swapped[:, 3 * M:4 * M] = b, a
    s0, s1 = L.STATS_BASE + 10, L.STATS_BASE + 15  # the stats block follows the party order too
    sa, sb = swapped[:, s0:s0 + 5].clone(), swapped[:, s1:s1 + 5].clone()
    swapped[:, s0:s0 + 5], swapped[:, s1:s1 + 5] = sb, sa
    everything = torch.ones(len(rows), 12, dtype=torch.bool)
    with torch.no_grad():
        before, after = net(obs[rows], everything)[0].logits, net(swapped, everything)[0].logits
    perm = list(range(12))
    perm[6], perm[7] = 7, 6
    assert torch.allclose(before[:, perm], after, atol=1e-4)


def test_damage_inputs_compile_as_one_graph():
    """Loading the tables inside the forward once split it in ten graphs and cost
    14-15% of training throughput; the network holds them as buffers now."""
    torch._dynamo.reset()
    obs, mask = observations()
    explained = torch._dynamo.explain(PolicyTF(d=32, layers=2, heads=2, react=True, compact=True, damage=True))(
        obs, net_.legal_bits(mask))
    assert explained.graph_break_count == 0, [str(b.reason)[:120] for b in explained.break_reasons]


def test_widening_for_damage_keeps_the_function_and_learns_the_inputs():
    torch.manual_seed(6)
    obs, mask = observations()
    legal = net_.legal_bits(mask)
    net = PolicyTF(d=32, layers=2, heads=2, react=True, compact=True).eval()
    wide = with_damage(net)
    assert wide.damage and not wide.training
    with torch.no_grad():
        (d0, v0), (d1, v1) = net(obs, legal), wide(obs, legal)
    assert torch.allclose(d0.logits, d1.logits) and torch.allclose(v0, v1)
    wide(obs, legal)[1].sum().backward()
    assert wide.mon_in.weight.grad[:, -2:].abs().sum() > 0 and wide.move_in.weight.grad[:, -2:].abs().sum() > 0
