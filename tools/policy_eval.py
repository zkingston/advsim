"""Score a network against opponents on fixed mirrored couples, no training.

    uv run --group rl python tools/policy_eval.py v2 --temps 1,0.5,0.25,0 --against maxdamage,v2

A network is a checkpoint tag (artifacts/ppo_<tag>.npz); temperature 0 takes
the top action. Each opponent is a baseline name or a tag, played at
temperature 1. Prints the network's score against each.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from advsim import net as net_, players  # noqa: E402
from league import Evaluator  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tag', help='the network: a checkpoint tag or .npz path')
    ap.add_argument('--temps', default='1', help='comma-separated temperatures to score it at (0: the top action)')
    ap.add_argument('--against', default='maxdamage', help='comma-separated baselines or checkpoint tags')
    ap.add_argument('--games', type=int, default=4096, help='per opponent and temperature')
    ap.add_argument('--seed', type=int, default=11, help='the deals')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args(argv)
    torch.set_float32_matmul_precision('high')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda'))
    ev = Evaluator(args.games, args.device, args.seed, amp)
    net = net_.load(players.checkpoint(args.tag), args.device)
    opponents = {o: o if o in players.BASELINES else net_.load(players.checkpoint(o), args.device)
                 for o in args.against.split(',')}
    for temp in (float(x) for x in args.temps.split(',')):
        me = net if temp == 1 else net_.Tempered(net, max(temp, 1e-3))
        scores = '  '.join(f'{o} {ev.score(me, opp):.3f}' for o, opp in opponents.items())
        print(f'{args.tag} at temperature {temp:g}: {scores}', flush=True)


if __name__ == '__main__':
    main()
