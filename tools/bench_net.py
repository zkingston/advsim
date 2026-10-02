"""Time a policy network on real observations, as training runs it.

    uv run --group rl python tools/bench_net.py                # the training recipe
    uv run --group rl python tools/bench_net.py --arch v2 --kernels 20

Prints forward+backward rows/s on --rows-row chunks with an Adam step (the
PPO update), no-grad forward rows/s (the rollout), and peak memory; with
--kernels N, the N costliest GPU kernels of one training step.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from advsim import net as net_  # noqa: E402
from advsim.env import Gen3Env  # noqa: E402
from advsim.net_tf import PolicyTF  # noqa: E402


def timed(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', default='tf', choices=['tf', 'v2'])
    ap.add_argument('--d', type=int, default=96, help='tf: width')
    ap.add_argument('--layers', type=int, default=3, help='tf: layers')
    ap.add_argument('--react', action=argparse.BooleanOptionalAction, default=True, help='tf: the opponent-action token')
    ap.add_argument('--compact', action=argparse.BooleanOptionalAction, default=True, help='tf: 17 tokens')
    ap.add_argument('--damage', action=argparse.BooleanOptionalAction, default=True, help='tf: damage inputs')
    ap.add_argument('--rows', type=int, default=8192, help='rows per forward')
    ap.add_argument('--no-compile', action='store_true', help='eager PyTorch')
    ap.add_argument('--mode', default=None, help='torch.compile mode, e.g. max-autotune-no-cudagraphs (static shapes)')
    ap.add_argument('--kernels', type=int, default=0, help='print the N costliest GPU kernels of a training step')
    args = ap.parse_args(argv)
    if args.arch == 'v2':
        args.react = False
    torch.set_float32_matmul_precision('high')
    env = Gen3Env(args.rows // 2, device='cuda:0')
    env.reset(seed=3)
    obs, mask = env.observe()
    obs, legal = obs.clone().view(args.rows, -1), net_.legal_bits(mask).view(args.rows, 12)
    net = (PolicyTF(args.d, args.layers, 4, args.react, args.compact, args.damage) if args.arch == 'tf'
           else net_.PolicyV2()).cuda()
    f = net if args.no_compile else torch.compile(net, dynamic=args.mode is None, mode=args.mode)
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)

    def train_step():
        with amp():
            out = f(obs, legal, opp_action=True) if args.react else f(obs, legal)
        loss = -out[0].logits.logsumexp(-1).mean() + out[1].square().mean()
        if args.react:
            loss = loss + out[2]['opp_action'].logsumexp(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def rollout_step():
        with amp():
            f(obs, legal)[0].sample()

    torch.cuda.reset_peak_memory_stats()
    t_train = timed(train_step, 10)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    t_roll = timed(rollout_step, 20)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'{args.arch} {params:.2f}M params: train {args.rows / t_train / 1e3:,.0f}K rows/s ({t_train * 1e3:.1f} ms), '
          f'rollout {args.rows / t_roll / 1e3:,.0f}K rows/s ({t_roll * 1e3:.1f} ms), peak {mem:.2f} GiB')
    if args.kernels:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            train_step()
            torch.cuda.synchronize()
        agg = {}
        for e in prof.events():
            if e.device_type.name == 'CUDA':
                a = agg.setdefault(e.name, [0.0, 0])
                a[0] += e.device_time_total
                a[1] += 1
        total = sum(v[0] for v in agg.values())
        print(f'{total / 1e3:.1f} ms of kernels, {sum(v[1] for v in agg.values())} launches')
        for name, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:args.kernels]:
            print(f'{t / 1e3:7.2f} ms {c:4d}x  {name[:100]}')


if __name__ == '__main__':
    main()
