"""L1: the damage path against Showdown's own getDamage, case for case, with
the Pokemon's own abilities and items. `tools/parity.py damage` runs it at scale."""
import numpy as np
import pytest

from advsim import oracle
from advsim.replay import damage

CASES = 4000
pytestmark = pytest.mark.oracle


@pytest.fixture(scope='module')
def cases():
    return oracle.damage_cases(CASES, seed=20260918)


def test_damage_matches_showdown(cases):
    got, want = damage(cases)
    bad = np.flatnonzero(got != want)
    if len(bad):
        i = int(bad[0])
        pytest.fail(f'{len(bad)}/{len(cases)} mismatched; first: engine {got[i]} vs showdown {want[i]} '
                    f'for {cases[i]}')


def test_cases_cover_the_interesting_axes(cases):
    assert {c['crit'] for c in cases} == {True, False}
    assert len({c['words']['weather'] for c in cases}) == 4
    assert len({c['roll'] for c in cases}) == 16
    assert any(c['damage'] == -1 for c in cases), 'no immunity case generated'
    assert len({c['words']['ability'][0][c['words']['active'][0]] for c in cases}) > 30, 'abilities stay live'
    assert len({c['words']['item'][0][c['words']['active'][0]] for c in cases}) > 6, 'items stay live'
