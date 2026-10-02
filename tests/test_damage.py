"""advsim/damage.py: the damage ranges networks read, against Gen 3's formula
worked by hand on a built observation."""
import pytest

torch = pytest.importorskip('torch')
from advsim import artifacts, damage  # noqa: E402
from advsim.engine import obs_layout as L  # noqa: E402
from advsim.engine._generated import ids  # noqa: E402

M, A = len(L.MON), len(L.ACTIVE)


def board(move=ids.MOVE_EARTHQUAKE, own_types=(ids.TYPE_NORMAL, ids.TYPE_NORMAL),
          foe_types=(ids.TYPE_NORMAL, ids.TYPE_NORMAL), atk=300):
    """Own level-100 active (Atk `atk`) with `move` in slot 0 against a
    level-100 foe Snorlax at full HP."""
    obs = torch.zeros(1, L.OBS_DIM, dtype=torch.int16)
    snorlax = artifacts.load_ids()['species'].index('snorlax')
    put = lambda token, name, v: obs[0].__setitem__(L.MON_BASE + token * M + L.MON.index(name), v)
    for name, v in (('present', 1), ('species', snorlax), ('level', 100), ('active', 1), ('hp', 300), ('maxhp', 300),
                    ('hp_pct', 100), ('move0', move)):
        put(0, name, v)
    for name, v in (('present', 1), ('species', snorlax), ('level', 100), ('active', 1), ('hp_pct', 100)):
        put(6, name, v)
    for token, (t0, t1) in enumerate((own_types, foe_types)):
        obs[0, L.ACTIVE_BASE + token * A + L.ACTIVE.index('type0')] = t0
        obs[0, L.ACTIVE_BASE + token * A + L.ACTIVE.index('type1')] = t1
    obs[0, L.STATS_BASE:L.STATS_BASE + 5] = torch.tensor([atk, 200, 200, 200, 200], dtype=torch.int16)
    return obs


def test_the_range_is_gen_3s_formula():
    # Snorlax at 100 is estimated at Def (2*65 + 52) + 5 = 187 and HP (2*160 + 52) + 110 = 482.
    # Base damage: floor(floor(42 * 100 * 300 / 187) / 50) + 2 = 136; the 85% roll floors to 115.
    lo, hi = damage.features(board())['own_moves'][0, 0].tolist()
    assert (lo, hi) == pytest.approx((115 / 482, 136 / 482))
    stab = damage.features(board(own_types=(ids.TYPE_GROUND, ids.TYPE_GROUND)))['own_moves'][0, 0].tolist()
    assert stab == pytest.approx((172 / 482, 204 / 482))  # floor(115 * 1.5), floor(136 * 1.5)
    immune = damage.features(board(foe_types=(ids.TYPE_FLYING, ids.TYPE_FLYING)))['own_moves'][0, 0].tolist()
    assert immune == [0.0, 0.0]
    toss = damage.features(board(move=ids.MOVE_SEISMICTOSS))['own_moves'][0, 0].tolist()
    assert toss == pytest.approx((100 / 482, 100 / 482))  # the user's level, whatever the stats
    f = damage.features(board())
    assert f['party'][0, 0].tolist() == pytest.approx((136 / 482, 0.0))  # dealt; the foe has shown no moves
    assert (f['own_moves'][0, 1:] == 0).all()  # empty slots


def test_real_observations_give_sane_ranges():
    import sys
    sys.path.insert(0, 'tests')
    from conftest import make_env
    from test_env_api import play
    env = make_env(64, seed=31)
    play(env, 5)
    obs = env.observe()[0].clone().view(128, -1)
    f = damage.features(obs)
    for v in f.values():
        assert torch.isfinite(v).all() and (v >= 0).all() and (v <= 3).all()
    for key in ('own_moves', 'foe_moves'):
        assert (f[key][..., 0] <= f[key][..., 1]).all()
    assert (f['own_moves'][..., 1] > 0).any() and (f['party'][..., 0] > 0).any()
