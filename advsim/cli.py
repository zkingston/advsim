"""The `advsim` command line: `advsim build` turns Showdown's dump and the team pool into the engine's data."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='advsim')
    sub = ap.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build', help='dump -> dex.npz, ids.json, pool.npz, generated modules')
    build.add_argument('--limit', type=int, default=None, help='read only the first N pool teams')
    build.add_argument('--pool', default=None, help='the team pool (default artifacts/pool.jsonl)')
    build.add_argument('--no-pool', action='store_true', help='skip pool.npz, which takes a minute over 1M teams')
    args = ap.parse_args(argv)

    if args.command == 'build':
        from advsim.build import build_dex, build_pool, codegen
        out = build_dex.build(args.pool, args.limit)
        counts = {k: len(v) - 1 for k, v in out['ids'].items()
                  if k in ('species', 'moves', 'abilities', 'items', 'conditions')}
        paths = codegen.generate(out['ids'])
        pool = ''
        if not args.no_pool:
            from advsim.build import setdist
            teams = build_pool.build(args.pool, args.limit)['teams']
            sets = setdist.build(teams, len(out['ids']['species']))['sets']
            pool = f' | pool {teams.shape[0]} teams, {sets.shape[0]} distinct sets'
        print(f'built {len(out["arrays"])} arrays | vocabulary {counts} | '
              f'dead callbacks {len(out["dead"])} | generated {len(paths)} modules{pool}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
