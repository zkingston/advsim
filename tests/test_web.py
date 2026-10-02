"""The static page's halves against the Python ones: the JS converter
(web/js/infostate.js, observation.js) must give live/observation.py's vector
at every decision, and the JS network (web/js/policy.js) PolicyV2's logits
and value. Regenerate web/ with tools/export_web.py after a checkpoint or
obs_layout change."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from advsim import fileio, net, oracle  # noqa: E402
from advsim.engine import obs_layout as L  # noqa: E402
from advsim.live.infostate import InfoState  # noqa: E402
from advsim.live.observation import Vocab, observe  # noqa: E402

pytestmark = pytest.mark.oracle


def python_obs(case: dict, vocab: Vocab) -> list:
    states = [InfoState(p, vocab.species_types) for p in ('p1', 'p2')]
    out = []
    for view in [case['start']] + [s['view'] for s in case['segments']]:
        row = []
        for p, st in enumerate(states):
            st.feed(case['log'], view['cursor'])
            st.take_request(view['requests'][p])
            row.append(observe(st, vocab))
        out.append(row)
    return out


@pytest.fixture(scope='module')
def observations():
    cases = oracle.turn_cases(20, seed=4401, turns=80, policy='mix', protocol=True)
    vocab = Vocab()
    return [(python_obs(c, vocab), js) for c, js in zip(cases, oracle.web_observe(cases))]


def test_the_js_converter_matches_the_python_one(observations):
    names, bad, n = L.names(), {}, 0
    for want, got in observations:
        for w_view, g_view in zip(want, got):
            for w, g in zip(w_view, g_view):
                n += 1
                for i in np.flatnonzero(w != np.array(g)):
                    bad[names[i]] = bad.get(names[i], 0) + 1
    assert n > 1000 and not bad, bad


def forward(policy, rows):
    obs = torch.from_numpy(rows)
    legal = obs[:, L.MASK_BASE:L.MASK_BASE + 12].bool()
    with torch.no_grad():
        dist, value = policy(obs, legal)
    return dist.logits.numpy(), value.numpy(), legal.numpy()  # logits normalized


def half(policy):
    """The network as the page has it: every weight rounded to float16 (tools/export_web.py)."""
    with torch.no_grad():
        for p in policy.parameters():
            p.copy_(p.half().float())
    return policy


@pytest.fixture(scope='module')
def rows(observations):
    rows = np.array([o for want, _ in observations for view in want for o in view][:400])
    return rows[rows[:, L.MASK_BASE:L.MASK_BASE + 12].any(1)]


MODELS = fileio.read_json(fileio.ROOT / 'web' / 'models' / 'index.json')


def needs_checkpoint(model):
    if not (fileio.ARTIFACTS / f'ppo_{model}.npz').exists():
        pytest.skip(f'artifacts/ppo_{model}.npz missing; it is a release download (README)')


@pytest.mark.parametrize('model', MODELS)
def test_the_js_network_matches_torch(rows, model):
    needs_checkpoint(model)
    got = oracle.web_forward(model, rows.tolist())
    want, value, keep = forward(half(net.load(fileio.ARTIFACTS / f'ppo_{model}.npz', 'cpu')), rows)
    js = np.array(got['logits'])
    js = js - np.log(np.exp(js - js.max(1, keepdims=True)).sum(1, keepdims=True)) - js.max(1, keepdims=True)
    assert np.abs(js[keep] - want[keep]).max() < 1e-3
    assert np.abs(np.array(got['value']) - value).max() < 1e-3


@pytest.mark.parametrize('model', MODELS)
def test_float16_weights_barely_move_the_network(rows, model):
    needs_checkpoint(model)
    path = fileio.ARTIFACTS / f'ppo_{model}.npz'
    (full, v_full, keep), (halved, v_half, _) = forward(net.load(path, 'cpu'), rows), forward(half(net.load(path, 'cpu')), rows)
    p_full, p_half = np.exp(full) * keep, np.exp(halved) * keep
    # long: 0.998 of top actions agree (a near-tie flips), probabilities move 0.004 at most, the value
    # 0.0007. dmg46: every top action agrees, probabilities move 0.0003 on average and 0.005 at the
    # 99th percentile, one of 400 positions 0.023, the value 0.002.
    change = np.abs(p_full - p_half)[keep]
    assert (p_full.argmax(1) == p_half.argmax(1)).mean() >= 0.99
    assert np.quantile(change, 0.99) < 0.01 and change.max() < 0.03
    assert np.abs(v_full - v_half).max() < 0.01
