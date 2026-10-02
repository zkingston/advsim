"""Parity diagnostics against Showdown.

The suite proves the engine matches on a small sample; this is what you run
when it does not, and what measures a wider sample than the suite can afford.

    uv run python tools/parity.py sweep --seeds 6 --turns 8
    uv run python tools/parity.py sweep --seed 200001 --seeds 200 --battles 500 --turns 100 --jobs 24
    uv run python tools/parity.py first --seeds 101,102
    uv run python tools/parity.py damage --seeds 200 --battles 50000 --jobs 20   # L1, 10M cases
    uv run python tools/parity.py perft --seed 7000001 --seeds 500 --battles 3000 --jobs 20
    uv run python tools/parity.py devices --policy mix --seeds 4 --battles 500 --turns 100
    uv run python tools/parity.py sweep --policy emerald --turns 100

`sweep` reports mismatch counts by field. `first` prints the earliest segment
where a battle diverges, with its call sites.

Every decision point is judged by `advsim/replay.py`, the same code the suite
uses: draw counts, then the canonical hash over every field Showdown can speak
to, then the legal-action mask. Compare draw counts before anything else: a
wrong count names the missing or extra call site, while equal counts with
wrong state mean a value was read as the wrong roll.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import functools
import multiprocessing
import sys

import numpy as np

from advsim import artifacts, oracle
from advsim.build import families as F
from advsim.replay import damage as damage_l1, replay


def fetch(seed: int, n: int, turns: int, policy: str = 'random') -> list[dict]:
    return oracle.turn_cases(n, seed=seed, turns=turns, policy=policy)


def compare(cases: list[dict]) -> collections.Counter:
    """Mismatch counts by field over every decision point, not just the last.

    Checking only where a battle ends hides anything a later turn puts back:
    three Trick bugs sat in the middle of their battles and the counts did not
    move when they were fixed.
    """
    bad = collections.Counter()
    for _, _, problems in replay(cases):
        bad.update(problems.keys())
    return bad


def active_line(words: dict, ids: dict, side: int) -> str:
    a = words['active'][side]
    status = ids['conditions'][words['status'][side][a]] or 'healthy'
    item = ids['items'][words['item'][side][a]] or 'no item'
    return (f'{ids["species"][words["species"][side][a]]} {words["hp"][side][a]}/{words["maxhp"][side][a]} '
            f'{ids["abilities"][words["ability"][side][a]]} {item} {status} '
            f'{[ids["types"][t] for t in words["types"][side]]}')


def describe(words: dict, ids: dict, choices: list[int]) -> list[str]:
    out = []
    for side, action in enumerate(choices):
        a = words['active'][side]
        if action < 4:
            out.append(ids['moves'][words['moves'][side][a][action]])
        elif action < 10:
            out.append(f'switch to {ids["species"][words["species"][side][action - 4]]}')
        else:
            out.append('pass' if action == 10 else 'forced')
    return out


def show(seed: int, b: int, index: int, case: dict, problems: dict, ids: dict) -> None:
    seg = case['segments'][index]
    before = case['before'] if index == 0 else case['segments'][index - 1]['after']
    print(f'--- seed {seed} case {b} segment {index}')
    print(f'    actions {seg["choices"]} -> {describe(before, ids, seg["choices"])}')
    print(f'    draws: {len(seg["draws"])} this segment' +
          (', log cursor engine {} showdown {}'.format(*problems['draws']) if 'draws' in problems else ''))
    print('    sites: ' + ' | '.join(x.replace('Battle.', '').replace('BattleActions.', '') for x in seg['sites']))
    for side in (0, 1):
        print(f'    p{side + 1}: {active_line(before, ids, side)} weather {ids["weather"][before["weather"]]}')
    for name, (engine, showdown) in problems.items():
        if name == 'legal':
            engine = [f'{m:#014b}' for m in engine]
            showdown = [f'{m:#014b}' for m in showdown]
        print(f'      {name}: engine {engine}')
        print(f'      {" " * len(name)}  showdown {showdown}')


def sweep_seed(seed: int, battles: int, turns: int, policy: str) -> tuple[int, int, int, dict, dict]:
    """One seed, start to finish: its own Showdown process, its own engine.

    A seed that blows up is reported as one, rather than taking a run of
    hundreds down with it.
    """
    try:
        cases = fetch(seed, battles, turns, policy)
        hits = collections.Counter(name for c in cases for name in c['touched'])
        return seed, len(cases), sum(len(c['segments']) for c in cases), dict(compare(cases)), dict(hits)
    except Exception as e:  # noqa: BLE001 - the whole point is not to care which
        return seed, 0, 0, {f'crashed ({e})': 1}, {}


def cmd_sweep(args, ids):
    """Seeds are independent, so --jobs runs them side by side, each in a fresh
    process: Showdown is the bottleneck and it runs one battle at a time."""
    total = points = 0
    bad, hits = collections.Counter(), collections.Counter()
    work = functools.partial(sweep_seed, battles=args.battles, turns=args.turns, policy=args.policy)
    if args.jobs > 1:
        # spawn, not fork: a forked Warp runtime is not safe to use.
        pool = concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn'))
        results = pool.map(work, args.seed_list)
    else:
        pool, results = None, map(work, args.seed_list)
    for seed, n, p, found, touched in results:
        total += n
        points += p
        bad.update(found)
        hits.update(touched)
        if args.jobs > 1:
            # Progress, so a long run shows where it is and which seed to rerun.
            print(f'seed {seed}: {n} battles, {p} points, ' +
                  (', '.join(f'{k} {v}' for k, v in sorted(found.items())) or 'clean'), flush=True)
    if pool:
        pool.shutdown()
    print(f'{total} battles, {points} decision points, full hash at every one')
    print('mismatches: ' + (', '.join(f'{k} {v}' for k, v in sorted(bad.items())) if bad else 'none'))
    report_coverage(hits, ids)
    return 1 if bad else 0


def report_coverage(hits: collections.Counter, ids: dict) -> None:
    """SPEC §Verification: count what L3 touched, from Showdown's own log and the
    actives it saw. Anything with zero hits needs an L2 scenario."""
    vocab = [(t, n) for t in ('moves', 'abilities', 'items', 'conditions') for n in ids[t][1:]]
    zero = [f'{t[:-1] if t != "abilities" else "ability"} {n}' for t, n in vocab if not hits[n]]
    rare = sorted({(hits[n], n) for _, n in vocab if hits[n]})[:5]
    print(f'coverage: {len(vocab) - len(zero)} of {len(vocab)} vocabulary entries hit; '
          f'rarest {", ".join(f"{n} {k}" for k, n in rare)}')
    print('zero-hit: ' + (', '.join(zero) or 'none'))
    gap = F.uncovered(set(hits))
    print('families and one-offs with no hit: ' + (', '.join(gap) or 'none'))


def cmd_first(args, ids):
    """The earliest diverging segment of each battle, which is the only one worth reading."""
    shown = 0
    for seed in args.seed_list:
        cases = fetch(seed, args.battles, args.turns, args.policy)
        first = {}
        for index, b, problems in replay(cases):
            if problems and b not in first:
                first[b] = (index, problems)
        for b in sorted(first):
            if shown >= args.limit:
                return 0
            show(seed, b, first[b][0], cases[b], first[b][1], ids)
            shown += 1
    return 0


def damage_seed(seed: int, n: int) -> tuple[int, int, int, str]:
    """One L1 shard: n getDamage cases from Showdown, computed by the engine."""
    try:
        got, want = damage_l1(oracle.damage_cases(n, seed=seed))
        bad = np.flatnonzero(got != want)
        first = f'case {int(bad[0])}: engine {got[bad[0]]} showdown {want[bad[0]]}' if len(bad) else ''
        return seed, n, len(bad), first
    except Exception as e:  # noqa: BLE001
        return seed, 0, 1, f'crashed ({e})'


def cmd_damage(args, ids):
    """SPEC L1: getDamage case for case, abilities and items live. --battles is
    the cases per seed here."""
    work = functools.partial(damage_seed, n=args.battles)
    pool = concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn'))
    total = bad = 0
    for seed, n, wrong, first in pool.map(work, args.seed_list):
        total += n
        bad += wrong
        print(f'seed {seed}: {n} cases, {wrong} mismatched {first}'.rstrip(), flush=True)
    pool.shutdown()
    print(f'{total} damage cases: {bad} mismatched')
    return 1 if bad else 0


def perft_seed(seed: int, depth: int, stride: int, budget: int) -> tuple[int, str, int, int, dict]:
    """One perft position, its leaves replayed through the engine leaf by leaf,
    each from the engine's own fork of the position.
    Every leaf matching (draw count, full hash, legal masks) means the
    successor multisets are identical too. The status is 'done', 'too big'
    (an expansion passed `budget` leaves), 'ended' (the battle ended before
    the position) or the crash."""
    bad = collections.Counter()
    leaves = depth1 = 0
    try:
        for cases in oracle.perft(seed, turns=1 + seed % 30, depth=depth, budget=budget, stride=stride):
            if isinstance(cases, int):
                depth1 = cases
                continue
            leaves += len(cases)
            for _, _, problems in replay(cases, fork=True):
                bad.update(problems.keys())
    except Exception as e:  # noqa: BLE001
        return seed, 'too big' if 'leaves' in str(e) and 'over' in str(e) else f'crashed ({e})', 0, leaves, dict(bad)
    return seed, 'done' if depth1 else 'ended', depth1, leaves, dict(bad)


def cmd_perft(args, ids):
    """SPEC perft: every joint action and chance outcome at depth 1, and depth 2
    below every --battles-th depth-1 leaf (the first always), damage rolls min,
    mid and max, from --seeds positions. A position that passes --budget leaves
    in all (speed ties and multi-hit moves multiply) is skipped and counted,
    and seeds are drawn on until --seeds positions are done. At ~800 leaves a
    second per Showdown process the budget bounds a position at ~40 s, so 500
    positions on 16 jobs finish within ~20 minutes at worst."""
    work = functools.partial(perft_seed, depth=2, stride=args.battles, budget=args.budget)
    pool = concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn'))
    status, bad = collections.Counter(), collections.Counter()
    depth1 = leaves = 0
    next_seed = args.seed
    running = set()
    while status['done'] < args.seeds or running:
        while len(running) < args.jobs and status['done'] + len(running) < args.seeds:
            running.add(pool.submit(work, next_seed))
            next_seed += 1
        finished, running = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
        for f in finished:
            seed, st, d1, n, found = f.result()
            status[st if not st.startswith('crashed') else 'crashed'] += 1
            if st == 'done':
                depth1 += d1
                leaves += n
            bad.update(found)
            if found or st.startswith('crashed') or st == 'too big':
                print(f'seed {seed}: {st}, {d1} depth-1 leaves, {n} cases ' +
                      ', '.join(f'{k} {v}' for k, v in sorted(found.items())), flush=True)
            if status['done'] % 50 == 0 and st == 'done':
                print(f'... {status["done"]} positions done', flush=True)
    pool.shutdown()
    print(f'{status["done"]} positions, {depth1} depth-1 leaves, {leaves} leaf cases replayed; '
          f'skipped: {status["too big"]} over {args.budget} leaves, {status["ended"]} ended first, '
          f'{status["crashed"]} crashed')
    print('mismatches: ' + (', '.join(f'{k} {v}' for k, v in sorted(bad.items())) if bad else 'none'))
    return 1 if bad or status['crashed'] else 0


def cmd_devices(args, ids):
    """SPEC §Verification: integer-only math means the CPU and the GPU reach the
    same hash at every decision point. Replays each seed on both and compares."""
    points = differ = bad = 0
    for seed in args.seed_list:
        cases = fetch(seed, args.battles, args.turns, args.policy)
        streams = []
        for device in ('cpu', args.device):
            hashes = {}
            bad += sum(1 for _, _, problems in replay(cases, device, hashes) if problems)
            streams.append(hashes)
        points += len(streams[0])
        differ += sum(1 for key, h in streams[0].items() if streams[1].get(key) != h)
        print(f'seed {seed}: {len(streams[0])} points, {differ} differ so far', flush=True)
    print(f'{points} decision points on cpu and {args.device}: {differ} hashes differ, '
          f'{bad} points disagree with Showdown')
    return 1 if differ or bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('sweep', 'first', 'devices', 'damage', 'perft'))
    ap.add_argument('--seed', type=int, default=101, help='the first seed')
    ap.add_argument('--seeds', type=int, default=4, help='how many consecutive seeds')
    ap.add_argument('--battles', type=int, default=200, help='per seed: battles (sweep, first, devices), cases (damage), or the stride between positions (perft)')
    ap.add_argument('--turns', type=int, default=8, help='turns per battle; long battles are where PP bugs live (try 100)')
    ap.add_argument('--limit', type=int, default=3, help='first: how many divergences to print')
    ap.add_argument('--budget', type=int, default=30000, help='perft: leaves one position may reach, depth 2 included; bounds each at ~40 s')
    ap.add_argument('--device', default='cuda:0', help='devices: the device compared against the CPU')
    ap.add_argument('--jobs', type=int, default=1, help='sweep, damage, perft: seeds to run at once, one process each')
    ap.add_argument('--policy', choices=('random', 'emerald', 'maxdamage', 'status', 'switchaverse', 'mix'),
                    default='random', help='how both sides play (showdown/lib/play.js); mix draws one per side')
    args = ap.parse_args(argv)
    args.seed_list = list(range(args.seed, args.seed + args.seeds))
    ids = artifacts.load_ids()
    return {'sweep': cmd_sweep, 'first': cmd_first, 'devices': cmd_devices, 'damage': cmd_damage, 'perft': cmd_perft}[args.command](args, ids)


if __name__ == '__main__':
    sys.exit(main())
