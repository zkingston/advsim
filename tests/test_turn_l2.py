"""Battles played in Showdown and replayed in the engine from the same draws.

Each case runs several decision points, so counters that take turns to matter
(weather, toxic, a switch after a boost) actually run down. The engine is in
replay mode off Showdown's logged `rng.next()` outputs and every decision point
is judged by `advsim/replay.py`: draw counts first, then the canonical hash over every
field Showdown can speak to, then the legal-action mask.
"""
import collections

import pytest

from advsim import oracle
from advsim.replay import replay

CASES = 60
TURNS = 6  # decision points per battle, so a weather or toxic counter can run down


@pytest.fixture(scope='module')
def cases():
    try:
        return oracle.turn_cases(CASES, seed=31337, turns=TURNS)
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_every_decision_point_matches(cases):
    bad, first = collections.Counter(), None
    for index, b, problems in replay(cases):
        bad.update(problems.keys())
        if problems and first is None:
            first = f'case {b} segment {index} choices {cases[b]["segments"][index]["choices"]}: ' + \
                '; '.join(f'{k} engine {e} showdown {w}' for k, (e, w) in problems.items())
    assert not bad, f'{dict(bad)}\nfirst: {first}'


def test_cases_exercise_moves_and_effects(cases):
    assert len(cases) == CASES
    segs = [seg for c in cases for seg in c['segments']]
    assert len(segs) >= CASES * 2, 'battles should run several turns'
    assert any(any(s for row in seg['after']['status'] for s in row) for seg in segs), 'no status inflicted'
    assert len({sum(len(seg['draws']) for seg in c['segments']) for c in cases}) > 2, \
        'draw counts are suspiciously uniform'
    assert any(a >= 4 for seg in segs for a in seg['choices']), 'no switch exercised'
    assert any(any(seg['after']['sub_hp']) for seg in segs), 'no Substitute stood up'
    assert any(2 in seg['after']['request'] for seg in segs), 'no faint and replacement exercised'
