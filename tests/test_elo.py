"""The bracket's ratings recover known strengths from synthetic games."""
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip('torch')  # elo imports advsim.players
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
import elo  # noqa: E402


def test_bradley_terry_recovers_known_ratings():
    true = {'random': 1000, 'maxdamage': 1300, 'search-32': 1600}
    rng = np.random.default_rng(0)
    rows, couple = [], 0
    for p, (a, b) in enumerate([('random', 'maxdamage'), ('random', 'search-32'), ('maxdamage', 'search-32')]):
        pa = 1 / (1 + 10 ** ((true[b] - true[a]) / 400))
        for g in range(4000):
            win = rng.random() < pa
            side = g % 2
            result = 1 if win == (side == 0) else 2
            rows.append({'pair': p, 'couple': couple + g // 2, 'a': a, 'b': b, 'a_side': side, 'result': result})
        couple += 2000
    s = elo.ratings(rows, boot=20)
    got = dict(zip(s['players'], s['elo']))
    for n, want in true.items():
        assert abs(got[n] - want) < 25, (n, got[n], want)
    assert all(lo <= e <= hi for e, lo, hi in zip(s['elo'], s['lo'], s['hi']))


def test_add_against_plays_only_the_named_opponents(tmp_path):
    import json
    import elo
    out = tmp_path / 'games.jsonl'
    common = ['--device', 'cpu', '--games', '4', '--capacity', '64', '--max-decisions', '300', '--out', str(out)]
    elo.main(common + ['--players', 'random,maxdamage'])
    elo.main(common + ['--add', 'status', '--against', 'random', '--anchor', 'maxdamage'])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {(r['a'], r['b']) for r in rows} == {('random', 'maxdamage'), ('status', 'random')}
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['elo'][summary['players'].index('maxdamage')] == 1000
