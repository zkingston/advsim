"""The policy networks: PolicyV2 reads real observations, its switch logits
follow the Pokemon they send in, and checkpoints of either kind load back."""
import pytest

torch = pytest.importorskip('torch')
from advsim import net as net_  # noqa: E402
from advsim.engine import obs_layout as L  # noqa: E402
from conftest import make_env  # noqa: E402
from test_env_api import as_np, play  # noqa: E402


def observations(n=32, steps=6):
    env = make_env(n, seed=4)
    play(env, steps)
    obs, mask = env.observe()
    return torch.as_tensor(as_np(obs)).view(n * 2, -1), torch.as_tensor(as_np(mask)).view(-1)


def test_v2_reads_observations():
    torch.manual_seed(0)
    obs, mask = observations()
    net = net_.PolicyV2(width=64, d=32).eval()
    with torch.no_grad():
        dist, v = net(obs, net_.legal_bits(mask))
    assert dist.probs.shape == (len(obs), 12) and v.shape == (len(obs),)
    illegal = ~net_.legal_bits(mask)
    assert (dist.probs[illegal] < 1e-6).all()
    assert torch.isfinite(v).all()


def test_switch_logits_follow_the_pokemon():
    """Swap two benched Pokemon in the observation: their switch logits swap
    and every other logit stays put."""
    torch.manual_seed(1)
    obs, _ = observations()
    net = net_.PolicyV2(width=64, d=32).eval()
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


def test_checkpoints_round_trip(tmp_path):
    obs, mask = observations(8, 2)
    for net in (net_.Policy(32), net_.PolicyV2(width=64, d=32)):
        net.eval()
        path = tmp_path / f'{type(net).__name__}{id(net)}.npz'
        net_.save(net, path)
        back = net_.load(path, 'cpu')
        assert type(back) is type(net)
        with torch.no_grad():
            assert torch.allclose(net(obs, net_.legal_bits(mask))[0].logits,
                                  back(obs, net_.legal_bits(mask))[0].logits)


def test_split_heads_are_the_concatenated_ones():
    """The pointer heads compute the trunk's half of their first layer once;
    it must be exactly head(concat(h, item))."""
    torch.manual_seed(2)
    net = net_.PolicyV2(width=64, d=32).eval()
    h, items = torch.randn(5, 64), torch.randn(5, 6, 32)
    direct = net.switch_head(torch.cat([h.unsqueeze(1).expand(-1, 6, 64), items], -1)).squeeze(-1)
    assert torch.allclose(net.point(net.switch_head, h, items), direct, atol=1e-5)


def test_older_checkpoints_read_the_current_observation():
    """Observation versions only append columns, so networks trained on an
    earlier one (the MLP league, PolicyV2 long) still load and play."""
    from advsim import players
    obs, mask = observations(8, 2)
    assert obs.shape[1] == L.OBS_DIM
    paths = [players.checkpoint(tag) for tag in ('league', 'long') if players.checkpoint(tag).exists()]
    if not paths:
        pytest.skip('ppo_league.npz and ppo_long.npz missing (ppo_long.npz is a release download)')
    for path in paths:
        net = net_.load(path, 'cpu')
        with torch.no_grad():
            dist, v = net(obs, net_.legal_bits(mask))
        assert torch.isfinite(v).all() and dist.probs.shape == (len(obs), 12)
