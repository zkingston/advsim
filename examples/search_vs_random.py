"""The device search playing p1 against uniformly random p2, every game at once.

Each decision runs `--iters` graph-launched iterations over every game's tree,
then p1 samples its root equilibrium strategy, restricted to its true legal
actions (a world can disagree about them under a hidden trap). Prints p1's
win rate over finished games.

    uv run --group rl python examples/search_vs_random.py --games 256 --iters 128
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from advsim.engine import kernels
from advsim.env import Gen3Env
from advsim.search.mcts import Search, sample


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--games', type=int, default=256, help='battles played at once')
    ap.add_argument('--iters', type=int, default=128, help='search iterations per decision')
    ap.add_argument('--lanes', type=int, default=8, help='descents per tree per iteration')
    ap.add_argument('--depth', type=int, default=6, help='the longest descent')
    ap.add_argument('--rollout', type=int, default=16, help='random decisions past the tree')
    ap.add_argument('--nodes', type=int, default=1024, help='nodes per tree')
    ap.add_argument('--seed', type=int, default=0, help='the deals and the search key')
    ap.add_argument('--max-decisions', type=int, default=300, help='stop after this many decisions')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args(argv)

    T = args.games
    cap = min(T, 4096)
    env = Gen3Env(T + cap * args.lanes, device=args.device)
    env.reset(seed=args.seed)
    search = Search(env, cap, lane_base=T, lanes=args.lanes, nodes=args.nodes, depth=args.depth,
                    rollout=args.rollout, hidden=True, key=args.seed)
    rng = np.random.default_rng(args.seed)
    acts = wp.zeros((env.batch, 2), dtype=wp.int32, device=args.device)
    log = wp.zeros((env.batch, 1), dtype=wp.uint32, device=args.device)
    legal_out = wp.zeros((env.batch, 2), dtype=wp.int32, device=args.device)
    t0 = time.time()
    for decision in range(args.max_decisions):
        result = env.numpy('result')[:T]
        live = np.nonzero(result == 0)[0]
        if not len(live):
            break
        wp.launch(kernels.legal_mask, dim=(env.batch, 2), device=args.device, inputs=[env.state, env.dex, legal_out])
        legal = legal_out.numpy()[:T]
        search.key = args.seed * 1000003 + decision
        host = np.zeros((env.batch, 2), dtype=np.int32)
        if args.iters:
            host[live, 0] = search.decide(live, np.zeros(len(live)), legal[live, 0], args.iters, rng)
        else:
            host[live, 0] = sample(np.ones((len(live), 12)), legal[live, 0], rng)
        host[live, 1] = sample(np.ones((len(live), 12)), legal[live, 1], rng)
        acts.assign(host)
        idx = wp.array(live.astype(np.int32), dtype=wp.int32, device=args.device)
        wp.launch(kernels.step_idx, dim=len(live), device=args.device, inputs=[env.state, env.dex, acts, log, idx])
        if decision % 10 == 0:
            print(f'decision {decision:3d}: {len(live)} games live, {time.time() - t0:.0f}s', flush=True)
    result = env.numpy('result')[:T]
    done = result != 0
    wins = int((result == 1).sum())
    assert not env.numpy('err')[:T].any(), 'an illegal action reached a game'
    print(f'{done.sum()} games finished: p1 (search) won {wins}, lost {int((result == 2).sum())}, '
          f'tied {int((result == 3).sum())}; win rate {wins / max(done.sum(), 1):.3f}; {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
