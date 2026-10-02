"""What the build covers, and the checks that it covers all of it.

Vocabulary is enumerated from the generator's set table and cross-checked
against the pool; anything outside it gets no code. Every callback-bearing
entry must be mapped in `families.py` or listed there as dead, so the build
fails rather than silently dropping behaviour.
"""
from __future__ import annotations

import itertools
import re


from advsim import fileio
from advsim.build import families as F
from advsim.build.tables import SENTINELS


def vocabulary(raw: dict) -> dict[str, list[str]]:
    """Everything the team generator can produce, enumerated from its own tables.

    Not sampled: the generator's set table lists every species, movepool and
    ability, and the 13 items come from the literals in getItem. The Hidden
    Power variants collapse to one move, as they do in a battle.
    """
    rb = raw['randbats']
    moves = {'hiddenpower' if m.startswith('hiddenpower') else m for m in rb['moves']}
    # Struggle is in no set, but a Choice Band holder whose locked move runs out
    # of PP has nothing else to use, so the engine needs it in the table.
    moves.add('struggle')
    return {'species': sorted(rb['species']), 'moves': sorted(moves),
            'abilities': sorted(rb['abilities']), 'items': sorted(rb['items'])}


def level_of(raw: dict, name: str) -> int | None:
    """Levels come from the set table. Deoxys formes have their own entries, so
    only a cosmetic forme with no entry of its own falls back to its base."""
    levels = raw['randbats']['levels']
    if name in levels:
        return levels[name]
    base = raw['species'].get(name, {}).get('baseSpecies', '')
    return levels.get(''.join(c for c in base.lower() if c.isalnum()))


def cross_check_pool(raw: dict, vocab: dict[str, list[str]], pool_path, limit: int | None) -> list[str]:
    """The generated pool must stay inside the enumerated vocabulary.

    The vocabulary is a claim about the generator; this is the evidence. A pool
    entry outside it means the generator changed or the enumeration is wrong,
    and either way the build should stop.
    """
    sets = {k: set(v) for k, v in vocab.items()}
    problems: list[str] = []
    seen = {k: set() for k in sets}
    rows = fileio.iter_jsonl(pool_path)
    for team in (itertools.islice(rows, limit) if limit else rows):
        for mon in team['mons']:
            seen['species'].add(mon['species'])
            seen['abilities'].add(mon['ability'])
            if mon['item']:
                seen['items'].add(mon['item'])
            seen['moves'].update(mon['moves'])
            want = level_of(raw, mon['species'])
            if want is not None and mon['level'] != want and len(problems) < 20:
                problems.append(f'level: {mon["species"]} is level {mon["level"]}, the set table says {want}')
    for table, found in seen.items():
        for name in sorted(found - sets[table]):
            problems.append(f'{table}: pool has {name}, which the generator vocabulary does not list')
    return problems


def has_callbacks(entry: dict) -> bool:
    if entry.get('callbacks'):
        return True
    for sub in ('condition', 'self'):
        if (entry.get(sub) or {}).get('callbacks'):
            return True
    return any(s.get('callbacks') for s in entry.get('secondaries', []))


def check_coverage(raw: dict, vocab: dict[str, list[str]], conditions: list[str]) -> list[str]:
    """Fail on any callback-bearing entry that families.py neither maps nor declares dead."""
    problems = []
    for table, mapping in (('moves', F.MOVES), ('abilities', F.ABILITIES), ('items', F.ITEMS)):
        for name in vocab[table]:
            entry = raw[table].get(name)
            if entry is None:
                problems.append(f'{table}: {name} is in the pool but not in the dump')
                continue
            if has_callbacks(entry) and name not in mapping and name not in F.DEAD:
                problems.append(f'{table}: {name} has callbacks but no family')
        for name in mapping:
            if name not in vocab[table]:
                problems.append(f'{table}: {name} is mapped but not in the vocabulary')
    for name in conditions:
        if name not in F.CONDITIONS:
            problems.append(f'conditions: {name} is referenced by a move but has no family')
    # An ability can carry no callbacks of its own and still do something: Early
    # Bird lives entirely inside the sleep condition. has_callbacks cannot see
    # those, so look for the names other entries read instead.
    for name in sorted(named_elsewhere(raw, set(vocab['abilities']))):
        if name not in F.ABILITIES and name not in F.DEAD:
            problems.append(f'abilities: {name} is read by another entry but has no family')
    return problems


def named_elsewhere(raw: dict, vocab: set[str]) -> set[str]:
    """Vocabulary abilities that some callback source asks about by name."""
    found, pattern = set(), re.compile(r"""hasAbility\(\s*['"]([a-z ]+)['"]""")
    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            found.update(n.replace(' ', '') for n in pattern.findall(node))
    for table in ('moves', 'abilities', 'items', 'conditions', 'scripts'):
        walk(raw[table])
    return found & vocab


def generator_assumptions(raw: dict, vocab: dict[str, list[str]]) -> list[str]:
    """Facts about the set table that let the engine leave code out.

    Curse is two different moves: a Ghost pays half its HP to curse the foe, and
    anyone else takes a self-boost. No Ghost in this format's set table learns
    it, so the engine only carries the second half. Sleep Talk is likewise the
    only move that acts through sleep. If either ever changes the build has to
    say so rather than the parity sweep.
    """
    problems = []
    for name in vocab['moves']:
        if raw['moves'][name].get('sleepUsable') and name != 'sleeptalk':
            problems.append(f'assumption: {name} is usable while asleep; the engine only lets '
                            'Sleep Talk through the sleep check')
    for name in vocab['moves']:
        if raw['moves'][name].get('ohko'):
            problems.append(f'assumption: {name} is a OHKO move; Sturdy is mapped dead because none existed')
    for name, entry in raw['randbats']['sets'].items():
        if not any('curse' in st['movepool'] for st in entry['sets']):
            continue
        if 'Ghost' in raw['species'][name]['types']:
            problems.append(f'assumption: {name} is a Ghost that learns Curse; '
                            'the engine implements only the self-boost half')
    return problems


def referenced_conditions(raw: dict, moves: list[str], abilities: list[str]) -> list[str]:
    out = set()
    for name in moves:
        m = raw['moves'][name]
        for field in ('status', 'volatileStatus', 'sideCondition', 'weather', 'slotCondition'):
            if m.get(field):
                out.add(str(m[field]).lower())
        for sec in m.get('secondaries', []):
            for field in ('status', 'volatileStatus'):
                if sec.get(field):
                    out.add(str(sec[field]).lower())
        if (m.get('self') or {}).get('volatileStatus'):
            out.add(m['self']['volatileStatus'].lower())
    for name in abilities:
        fam, params = F.ABILITIES.get(name, ('', ()))
        if fam == 'weather_setter':
            out.add({'sun': 'sunnyday', 'rain': 'raindance', 'sand': 'sandstorm'}[params[0]])
        if fam in ('status_immune', 'contact_punish'):
            sym = params[-1]
            if sym not in SENTINELS:
                out.add(sym)
    return sorted(out)
