"""PPO smoke run: the engine end to end on the GPU, M5's exit check.

p1 learns with PPO; p2 plays uniformly random legal actions. Only decisions
where p1 has a real choice become transitions, rewards come at game ends
(+1/-1/0), and values bootstrap only on truncation. Prints the win rate over
finished battles per iteration; it should climb well above one half.

    uv run --group rl python examples/ppo_smoke.py --iters 40
    uv run --group rl python examples/ppo_smoke.py --iters 300 --save artifacts/ppo_baseline.npz
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from advsim import net as net_
from advsim.env import Gen3Env
from advsim.net import Policy, legal_bits

def random_legal(legal: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(legal.float(), 1).squeeze(-1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=int, default=4096, help='battles played at once')
    ap.add_argument('--steps', type=int, default=64, help='decisions per rollout')
    ap.add_argument('--iters', type=int, default=40, help='PPO iterations')
    ap.add_argument('--epochs', type=int, default=4, help='PPO epochs per rollout')
    ap.add_argument('--minibatch', type=int, default=16384, help='rows per PPO step')
    ap.add_argument('--lr', type=float, default=3e-4, help='Adam')
    ap.add_argument('--gamma', type=float, default=0.995, help='discount')
    ap.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    ap.add_argument('--max-steps', type=int, default=400, help='decisions before a battle is truncated')
    ap.add_argument('--seed', type=int, default=0, help='the network and the deals')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save', default=None, help='write the trained network here (npz)')
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    dev = args.device
    env = Gen3Env(args.batch, device=dev, max_steps=args.max_steps)
    env.reset(seed=args.seed)
    net = Policy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    B, T = args.batch, args.steps
    first_rate = None

    for it in range(args.iters):
        t0 = time.time()
        buf = {k: [] for k in ('obs', 'legal', 'act', 'logp', 'val', 'rew', 'done', 'trunc', 'mine')}
        wins = games = 0
        with torch.no_grad():
            for _ in range(T):
                obs, mask = env.observe()
                legal = legal_bits(mask)
                obs1, legal1 = obs[:, 0].clone(), legal[:, 0]
                dist, val = net(obs1, legal1)
                a1 = dist.sample()
                a2 = random_legal(legal[:, 1])
                mine = legal1.sum(-1) > 1  # a real choice: more than the one legal action
                env.step(torch.stack([a1, a2], dim=1))
                done = env.done.bool().clone()
                reward = env.reward.clone()
                wins += int((reward[done] > 0).sum())
                games += int(done.sum())
                for k, x in (('obs', obs1), ('legal', legal1), ('act', a1), ('logp', dist.log_prob(a1)),
                             ('val', val), ('rew', reward), ('done', done), ('trunc', env.truncated.bool().clone()),
                             ('mine', mine)):
                    buf[k].append(x)
            _, last_val = net(*[x[:, 0] for x in (env.observe()[0], legal_bits(env.observe()[1]))])
        buf = {k: torch.stack(v) for k, v in buf.items()}

        # GAE over every step of the stream; forced steps pass values through.
        adv = torch.zeros(T, B, device=dev)
        gae = torch.zeros(B, device=dev)
        next_val = last_val
        for t in reversed(range(T)):
            end = buf['done'][t] | buf['trunc'][t]
            nv = torch.where(end, torch.zeros_like(next_val), next_val)
            nv = torch.where(buf['trunc'][t], buf['val'][t], nv)  # bootstrap on truncation only
            delta = buf['rew'][t] + args.gamma * nv - buf['val'][t]
            gae = delta + args.gamma * args.lam * torch.where(end, torch.zeros_like(gae), gae)
            adv[t] = gae
            next_val = buf['val'][t]
        ret = adv + buf['val']

        sel = buf['mine'].reshape(-1)
        flat = {k: v.reshape(T * B, *v.shape[2:])[sel] for k, v in buf.items()}
        a, r = adv.reshape(-1)[sel], ret.reshape(-1)[sel]
        a = (a - a.mean()) / (a.std() + 1e-8)
        n = a.shape[0]
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, args.minibatch):
                i = perm[s:s + args.minibatch]
                dist, v = net(flat['obs'][i], flat['legal'][i])
                ratio = (dist.log_prob(flat['act'][i]) - flat['logp'][i]).exp()
                pg = -torch.min(ratio * a[i], ratio.clamp(0.8, 1.2) * a[i]).mean()
                vf = ((v - r[i]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + 0.5 * vf - 0.01 * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        assert torch.isfinite(loss), 'a non-finite loss'
        assert not env.err.any(), 'an illegal action reached the engine'
        rate = wins / max(games, 1)
        first_rate = rate if first_rate is None else first_rate
        sps = T * B / (time.time() - t0)
        print(f'iter {it:3d}  games {games:6d}  p1 win rate {rate:.3f}  loss {loss.item():+.3f}  '
              f'entropy {ent.item():.3f}  transitions {n}  {sps:,.0f} decisions/s', flush=True)
    print(f'win rate {first_rate:.3f} -> {rate:.3f}')
    if args.save:
        net_.save(net, args.save)
        print(f'saved {args.save}')


if __name__ == '__main__':
    main()
