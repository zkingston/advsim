"""The build's job is to fail on anything it cannot account for; these check that."""
import json

import pytest

from advsim import artifacts, fileio
from advsim.build import build_dex, codegen, tables
from advsim.build import vocab as V
from advsim.build import families as F

LIMIT = 5000  # the vocabulary saturates around team 2,600


@pytest.fixture(scope='module')
def built():
    return build_dex.build(limit=LIMIT, write=False)


def test_vocabulary_is_enumerated_not_sampled(built):
    """The generator's own tables, so these counts are exhaustive rather than observed."""
    counts = {k: len(v) - 1 for k, v in built['ids'].items()}
    # 113 from the sets, plus Struggle, which is in no set but is what a
    # Pokemon with nothing selectable uses.
    assert (counts['species'], counts['moves'], counts['abilities'], counts['items']) == (247, 114, 71, 13)
    assert 'struggle' in built['ids']['moves'] and 'struggle' not in \
        fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')['randbats']['moves']
    rb = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')['randbats']
    assert len(rb['species']) == 247 and len(rb['items']) == 13
    assert sum(1 for m in rb['moves'] if m.startswith('hiddenpower')) == 13


def test_hidden_power_variants_are_known_exactly(built):
    """13, enumerated. Nothing is sized to that count: hp_type stays a type ID."""
    rb = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')['randbats']
    hp = sorted(m for m in rb['moves'] if m.startswith('hiddenpower'))
    assert len(hp) == 13
    assert 'hiddenpowernormal' not in hp
    assert len(built['ids']['types']) == 18


def test_pool_stays_inside_the_enumerated_vocabulary():
    raw = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')
    vocab = V.vocabulary(raw)
    pool = fileio.ARTIFACTS / 'pool.jsonl'
    if not pool.exists():
        pytest.skip('artifacts/pool.jsonl missing')
    assert V.cross_check_pool(raw, vocab, pool, LIMIT) == []


def test_cross_check_rejects_a_pool_outside_the_vocabulary(tmp_path):
    """The vocabulary is a claim about the generator; the pool is the evidence."""
    raw = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')
    vocab = V.vocabulary(raw)
    bad = tmp_path / 'pool.jsonl'
    bad.write_text(json.dumps({'s': 0, 'i': 0, 'mons': [
        {'species': 'pikachu', 'level': 99, 'gender': 'M', 'ability': 'static', 'item': 'lightball',
         'hpType': 'Dark', 'maxhp': 200, 'stats': [1, 2, 3, 4, 5], 'moves': ['tackle'], 'maxpp': [56]}]}) + '\n')
    problems = V.cross_check_pool(raw, vocab, bad, None)
    assert any('tackle' in p for p in problems), problems
    assert any('level' in p for p in problems), problems


def test_deoxys_formes_keep_their_own_levels():
    """Cosmetic formes inherit a level; real formes do not, and Deoxys has four."""
    raw = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')
    levels = {name: V.level_of(raw, name)
              for name in ('deoxys', 'deoxysattack', 'deoxysdefense', 'deoxysspeed')}
    assert len(set(levels.values())) == 4, levels
    assert V.level_of(raw, 'unownp') == V.level_of(raw, 'unown')


def test_every_callback_entry_is_mapped_or_declared_dead(built):
    raw = fileio.read_json(fileio.ARTIFACTS / 'dex_raw.json')
    for table, mapping in (('moves', F.MOVES), ('abilities', F.ABILITIES), ('items', F.ITEMS)):
        for name in built['ids'][table][1:]:
            if V.has_callbacks(raw[table][name]):
                assert name in mapping or name in F.DEAD, f'{table}.{name} has callbacks but no family'


def test_unmapped_entry_fails_the_build(monkeypatch):
    """Drop a mapping and the build must refuse, not quietly emit family 0."""
    monkeypatch.delitem(F.MOVES, 'substitute')
    with pytest.raises(SystemExit, match='substitute'):
        build_dex.build(limit=LIMIT, write=False)


def test_hidden_power_is_one_row_with_room_for_every_type(built):
    ids, arrays = built['ids'], built['arrays']
    hp = ids['moves'].index('hiddenpower')
    assert sum(1 for m in ids['moves'] if m.startswith('hiddenpower')) == 1
    assert arrays['move_power'][hp] == 70
    assert arrays['move_type_from_mon'][hp] == 1
    # Only 13 Hidden Power types appear in a 1M pool, but nothing here is sized to
    # that sample: hp_type is a type ID, so all 18 remain expressible.
    assert len(ids['types']) == 18


def test_return_resolves_to_fixed_power(built):
    ids, arrays = built['ids'], built['arrays']
    assert arrays['move_power'][ids['moves'].index('return')] == 102


def test_ids_are_stable_and_zero_is_none(built):
    ids = built['ids']
    for table in ('species', 'moves', 'abilities', 'items', 'conditions'):
        assert ids[table][0] == ''
        assert ids[table][1:] == sorted(ids[table][1:])


def test_build_is_reproducible(built):
    again = build_dex.build(limit=LIMIT, write=False)
    assert fileio.hash_arrays(again['arrays']) == fileio.hash_arrays(built['arrays'])
    assert fileio.hash_json(again['ids']) == fileio.hash_json(built['ids'])


def test_written_artifacts_match_their_manifest_hashes():
    if not (fileio.ARTIFACTS / 'dex.npz').exists():
        pytest.skip('run `advsim build` first')
    assert artifacts.load_dex()
    assert artifacts.load_ids()


def test_generated_modules_are_up_to_date(built, tmp_path, monkeypatch):
    """The committed engine/_generated/ is what the build writes from the dump."""
    current = {p.name: p.read_text() for p in codegen.GENERATED.glob('*.py')}
    monkeypatch.setattr(codegen, 'GENERATED', tmp_path)
    monkeypatch.setattr(fileio, 'ARTIFACTS', tmp_path)  # it also writes layout.json
    for path in codegen.generate(built['ids']):
        assert path.read_text() == current[path.name], f'{path.name} is stale; run `advsim build`'


def test_boost_packing_is_zero_when_empty():
    assert tables.pack_boosts(None) == 0
    assert tables.pack_boosts({}) == 0
    packed = tables.pack_boosts({'atk': 2, 'spe': -1})
    assert packed & 0xF == 2 and (packed >> 16) & 0xF == 0xF  # -1 in two's complement
