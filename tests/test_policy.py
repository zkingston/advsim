"""The Emerald replacement chooser against its write-up's own worked examples.

Section 1 returns the party index it switches in, or -1 when it finds none and
Section 2 takes over. Species and moves only set types; nothing here plays.
"""
import pytest
from advsim import artifacts, oracle


def mon(species: str, *moves: str) -> dict:
    return {'species': species, 'moves': list(moves)}


CASES = [
    # Section 1, example 1: Corsola 4, Yanma 5.
    ({'foe': {'species': 'combusken'}, 'party': [mon('corsola', 'surf'), mon('yanma', 'wingattack')]}, 1, None),
    # Example 2: Magnemite scores 40 but has nothing super effective.
    ({'foe': {'species': 'combusken'},
      'party': [mon('rhyhorn', 'earthquake'), mon('yanma', 'wingattack'), mon('magnemite', 'thunderbolt')]}, 0, None),
    # Example 3: Fighting on Psychic comes before Fighting on Rock, so Lunatone is 4.
    ({'foe': {'species': 'combusken'}, 'party': [mon('lunatone', 'psychic'), mon('yanma', 'wingattack')]}, 1, None),
    # Example 4: a single type hits twice, Corsola rounds to 0 and is never chosen.
    ({'foe': {'species': 'typhlosion'}, 'party': [mon('charizard', 'flamethrower'), mon('corsola', 'surf')]}, -1, None),
    # Levitate is the one ability Section 1 knows.
    ({'foe': {'species': 'gengar', 'ability': 'levitate'}, 'party': [mon('marowak', 'earthquake')]}, -1, None),
    ({'foe': {'species': 'gengar'}, 'party': [mon('marowak', 'earthquake')]}, 0, None),
    # A status move is never super effective here.
    ({'foe': {'species': 'gyarados'}, 'party': [mon('pikachu', 'thunderwave')]}, -1, None),
    ({'foe': {'species': 'gyarados'}, 'party': [mon('pikachu', 'thunderbolt')]}, 0, None),
    # The table stops before Fighting on Ghost, so Gengar scores 2, not 0.
    ({'foe': {'species': 'machamp'}, 'party': [mon('gengar', 'psychic')]}, 0, None),
    # Section 2, examples 1-3: effectiveness, status moves by type, STAB from the Pokemon going out.
    ({'foe': {'species': 'vaporeon'}, 'party': [mon('rattata', 'tackle'), mon('pikachu', 'thunderbolt')],
      'base': 20}, None, 1),
    ({'foe': {'species': 'vaporeon'}, 'party': [mon('rattata', 'tackle'), mon('pikachu', 'thunderwave')],
      'base': 20}, None, 1),
    ({'foe': {'species': 'snorlax'}, 'party': [mon('rattata', 'hyperbeam'), mon('pikachu', 'thunderbolt')],
      'base': 50, 'stab': ['Electric']}, None, 1),
    # Example 4: 31 halves to 15 on Bug then doubles to 30 on Steel; Rock stays 31.
    ({'foe': {'species': 'scizor'}, 'party': [mon('machop', 'brickbreak'), mon('geodude', 'rockslide')],
      'base': 31}, None, 1),
    ({'foe': {'species': 'scizor'}, 'party': [mon('geodude', 'rockslide'), mon('machop', 'brickbreak')],
      'base': 31}, None, 0),
    # 300 is stored as 44, so a 200 behind it wins.
    ({'foe': {'species': 'snorlax'}, 'party': [mon('pikachu', 'thunderbolt'), mon('rattata', 'tackle')],
      'base': 200, 'stab': ['Electric']}, None, 1),
    # Moves with non-standard damage are skipped.
    ({'foe': {'species': 'snorlax'}, 'party': [mon('machop', 'seismictoss'), mon('rattata', 'tackle')],
      'base': 20}, None, 1),
    # Immune to everything: neither section picks.
    ({'foe': {'species': 'golem'}, 'party': [mon('pikachu', 'thunderbolt')], 'base': 20}, -1, -1),
]


@pytest.mark.oracle
def test_write_up_examples():
    picks, _ = oracle.emerald([case for case, _, _ in CASES])
    for (case, s1, s2), got in zip(CASES, picks, strict=True):
        if s1 is not None:
            assert got['section1'] == s1, case
        if s2 is not None:
            assert got['section2'] == s2, case


@pytest.mark.oracle
def test_table_matches_showdown():
    """Apart from the two Ghost rows the AI never reaches, the cartridge's table
    is Showdown's gen 3 chart in another order."""
    _, mismatches = oracle.emerald([])
    assert mismatches == []


def decisions(cases: list[dict]):
    """(state before, side, action) for every choice a side made."""
    for c in cases:
        states = [c['before']] + [seg['after'] for seg in c['segments']]
        for before, seg in zip(states, c['segments']):
            for side in (0, 1):
                yield before, side, seg['choices'][side]


def voluntary_switches(cases: list[dict]) -> int:
    """Switches chosen while the side could have moved instead."""
    return sum(1 for before, side, a in decisions(cases) if 4 <= a < 10 and before['request'][side] != 2)


def status_share(cases: list[dict], category) -> float:
    """The share of chosen moves that are status moves."""
    moves = [before['moves'][side][before['active'][side]][a] for before, side, a in decisions(cases) if a < 4]
    return sum(1 for m in moves if category[m] == 2) / len(moves)


@pytest.mark.oracle
def test_the_heuristics_play_as_named():
    """maxdamage and switchaverse switch only when made to, status plays more
    status moves than random play, and mix gives the sides their own."""
    category = artifacts.load_dex()['move_category']
    runs = {p: oracle.turn_cases(20, seed=3, turns=40, policy=p)
            for p in ('random', 'maxdamage', 'switchaverse', 'status', 'mix')}
    assert voluntary_switches(runs['random']) > 0
    assert voluntary_switches(runs['maxdamage']) == 0
    assert voluntary_switches(runs['switchaverse']) == 0
    assert status_share(runs['status'], category) > 2 * status_share(runs['random'], category)
    assert status_share(runs['maxdamage'], category) < status_share(runs['random'], category) / 2
    assert len({tuple(c['policies']) for c in runs['mix']}) > 5
