"""L2: crafted positions, played in both engines with the same forced draws.

A scenario is a JSON file with no expected values in it. Showdown builds the
battle, `export_state.js` hands its state to the engine, the scripted RNG steers
the outcomes the scenario cares about, and the two are compared by canonical
hash at every decision point. A mismatch is then diffed field by field, because
a hash says only that something moved.
"""
import re

import pytest

from advsim import artifacts, fileio
from advsim.build import families
from advsim.engine import mask as mask_
from advsim.oracle import Oracle
from advsim.replay import replay

SCENARIOS = sorted((fileio.ROOT / 'tests' / 'scenarios').glob('*.json'))


@pytest.fixture(scope='module')
def oracle():
    if not (fileio.ARTIFACTS / 'ids.json').exists():
        pytest.skip('artifacts missing; run `advsim build`')
    with Oracle() as o:
        yield o


def play(oracle, spec):
    # Drain the generator: an abandoned request leaves its `done` line in the
    # pipe, and the next command reads that instead of its own answer.
    return list(oracle.request(cmd='scenario', scenario=spec))[0]


def run(oracle, spec):
    """Play one scenario in both engines; returns the failures and the case."""
    case = play(oracle, spec)
    if case['unused']:
        return [f'forced sites that never fired: {case["unused"]}'], case
    for index, _, problems in replay([case]):
        if problems:
            seg = case['segments'][index]
            return [f'segment {index} (choices {seg["choices"]}, draws {seg["names"]}):'] + \
                [f'{k}: engine {e} != showdown {w}' for k, (e, w) in problems.items()], case
    return [], case


@pytest.mark.parametrize('path', SCENARIOS, ids=lambda p: p.stem)
def test_scenario(oracle, path):
    spec = fileio.read_json(path)
    failures, case = run(oracle, spec)
    assert not failures, f'{spec["name"]}: ' + '\n  '.join(failures)
    assert case['segments'], 'a scenario with no decision points tests nothing'


def test_scenarios_are_named_after_their_files():
    for path in SCENARIOS:
        assert fileio.read_json(path)['name'] == path.stem, path.name


@pytest.mark.built
def test_scenario_sets_and_forces_are_in_the_vocabulary():
    """A scenario may only use what the engine was built for, and may only steer
    call sites the catalog names. Both fail here rather than inside Showdown."""
    ids = artifacts.load_ids()
    catalog = fileio.read_json(fileio.ARTIFACTS / 'catalog.json')
    sites = {entry['name'] for entry in catalog.values() if entry.get('name')}
    for path in SCENARIOS:
        spec = fileio.read_json(path)
        assert spec['turns'], f'{path.name}: no turns'
        for name in spec.get('force', {}):
            assert name in sites, f'{path.name}: {name} is not a call site in the catalog'
        for side in ('p1', 'p2'):
            for mon in spec[side]:
                assert mon['species'] in ids['species'], f'{path.name}: {mon["species"]}'
                assert mon['ability'] in ids['abilities'], f'{path.name}: {mon["ability"]}'
                assert not mon.get('item') or mon['item'] in ids['items'], f'{path.name}: {mon["item"]}'
                for move in mon['moves']:
                    assert move in ids['moves'], f'{path.name}: {move}'


# The Showdown volatile that sets each engine bit. The names differ on both
# sides, so this is the one place the two vocabularies are tied together.
VOLATILES = {'mustrecharge': 'VF_MUSTRECHARGE', 'twoturnmove': 'VF_TWOTURN', 'trapped': 'VF_TRAPPED',
             'flashfire': 'VF_FLASHFIRE', 'attract': 'VF_ATTRACT', 'protect': 'VF_PROTECT',
             'endure': 'VF_ENDURE', 'leechseed': 'VF_LEECHSEED', 'flinch': 'VF_FLINCH',
             'destinybond': 'VF_DESTINYBOND', 'truant': 'VF_TRUANT'}


def test_the_exporter_carries_the_engine_s_volatile_bits():
    """export_state.js keeps its own copy of the bit numbers; renumbering one in
    mask.py without it would export a volatile as a different volatile."""
    js = (fileio.ROOT / 'showdown' / 'lib' / 'export_state.js').read_text()
    block = re.search(r'const VFLAGS = \{(.*?)\};', js, re.S).group(1)
    exported = {name: int(bit) for name, bit in re.findall(r'(\w+): (\d+)', block)}
    assert set(exported) == set(VOLATILES), 'the two lists of volatiles have drifted'
    assert int(re.search(r'const VF_TRANSFORMED = (\d+);', js).group(1)) == mask_.VF_TRANSFORMED.bit_length() - 1
    for name, const in VOLATILES.items():
        assert 1 << exported[name] == int(getattr(mask_, const)), name


def missing_coverage(oracle):
    """Families and one-offs that no scenario reaches, from Showdown's own log.

    The scenario says what it plays; this says what that actually touched, which
    is the only version that cannot drift from the position.
    """
    seen = set()
    for path in SCENARIOS:
        seen |= set(play(oracle, fileio.read_json(path))['touched'])
    return families.uncovered(seen)


def test_every_family_and_one_off_has_a_scenario(oracle):
    """L2's bar: every family and one-off in the vocabulary is crafted for."""
    gap = missing_coverage(oracle)
    assert not gap, f'{len(gap)} with no scenario: {gap}'
