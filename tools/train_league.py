"""PPO in a league: self-play, past snapshots and the five baselines.

    uv run --group rl python tools/train_league.py --anneal --minutes 120 --save artifacts/ppo_x.npz
    uv run --group rl python tools/train_league.py --init dmg46 --lr 1e-4 --anneal --minutes 1380 --save ...

The defaults are the recipe that works: PolicyTF, compact, width 96, 3 layers,
reacting to its prediction of the foe's action (--opp-action 0.1), with damage
ranges as inputs, compiled, the backward in 8,192-row chunks. --arch v2 or mlp,
--no-damage, --no-tf-compact and --no-compile turn parts off. A warm start
(--init) keeps the checkpoint's own architecture; restart it at --lr 1e-4,
since the default 3e-4 first knocks a trained network well below where it began.

Each battle slot draws its opponent when it is dealt, again after every game:
the current policy (self-play: the learner plays both sides, and both sides
train), a frozen snapshot of the learner, or a baseline. Snapshots are taken
every `--snapshot-every` iterations, the last `--pool` kept. Within snapshots
and within baselines the draw is prioritized by how often the learner loses
to each (AlphaStar's PFSP, weight (1 - win rate)^2), so it keeps playing what
it has not yet beaten. Against anything but itself the learner takes p1 or
p2 at random; every reward and value is from the side that acted.

Prints the learner's win rate against each baseline and
the snapshots as it goes, and every `--eval-every` iterations plays a fixed
set of mirrored couples against maxdamage and a reference network, which is
the learning curve to trust: the in-training rates are running averages
against opponents chosen for being hard. Matrix math runs in TF32 and the
networks in bf16 autocast; logits and values come back fp32.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import json

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from advsim import fileio, net as net_, players  # noqa: E402
from advsim.env import Gen3Env  # noqa: E402
from advsim.net import Policy, legal_bits  # noqa: E402
from advsim.net_tf import PolicyTF, with_damage  # noqa: E402
from advsim.players import Baselines  # noqa: E402
from league import BASE, NAMES, SELF, Evaluator, League, Runner, learner_mask  # noqa: E402,F401
from league_targets import foe_action, foe_action_loss  # noqa: E402

def gae(rew, val, done, trunc, last_val, gamma: float, lam: float):
    """Advantages and returns over [T, B, 2] streams, one per side of a slot:
    a game end cuts both, and a truncation bootstraps from the value there."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    running = torch.zeros_like(last_val)
    next_val = last_val
    for t in reversed(range(T)):
        end = (done[t] | trunc[t]).unsqueeze(-1)
        nv = torch.where(end, torch.zeros_like(next_val), next_val)
        nv = torch.where(trunc[t].unsqueeze(-1), val[t], nv)
        delta = rew[t] + gamma * nv - val[t]
        running = delta + gamma * lam * torch.where(end, torch.zeros_like(running), running)
        adv[t] = running
        next_val = val[t]
    return adv, adv + val


class Stopwatch:
    """--profile: seconds per stage, synchronising the GPU at each boundary
    (which itself slows the loop a little). Off, every call is a no-op."""

    def __init__(self, on: bool, cuda: bool) -> None:
        self.on, self.cuda, self.totals, self.last = on, cuda, {}, None

    def lap(self, stage: str | None = None) -> None:
        if not self.on:
            return
        if self.cuda:
            torch.cuda.synchronize()
        now = time.time()
        if stage is not None and self.last is not None:
            self.totals[stage] = self.totals.get(stage, 0.0) + now - self.last
        self.last = now

    def report(self) -> str:
        total = sum(self.totals.values())
        parts = sorted(self.totals.items(), key=lambda kv: -kv[1])
        self.totals = {}
        return '  '.join(f'{k} {v:.2f}s ({100 * v / total:.0f}%)' for k, v in parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=int, default=4096, help='battles played at once')
    ap.add_argument('--steps', type=int, default=64, help='decisions per rollout')
    ap.add_argument('--iters', type=int, default=None, help='iterations (default 1000, unbounded with --minutes)')
    ap.add_argument('--epochs', type=int, default=4, help='PPO epochs per rollout')
    ap.add_argument('--minibatch', type=int, default=32768, help='rows per PPO step')
    ap.add_argument('--micro', type=int, default=8192, help='backward in chunks of this many rows (0: whole minibatch)')
    ap.add_argument('--lr', type=float, default=3e-4, help='Adam; use 1e-4 with --init')
    ap.add_argument('--gamma', type=float, default=0.995, help='discount')
    ap.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    ap.add_argument('--ent', type=float, default=0.01, help='entropy bonus')
    ap.add_argument('--max-steps', type=int, default=400, help='decisions before a battle is truncated')
    ap.add_argument('--mix', default='0.5,0.25,0.25', help='self, snapshots, baselines')
    ap.add_argument('--snapshot-every', type=int, default=25, help='iterations between league snapshots')
    ap.add_argument('--pool', type=int, default=10, help='snapshots kept in the league')
    ap.add_argument('--seed', type=int, default=0, help='the network, the deals and the league')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save', default=None, help='the final network; with --checkpoint-every, also <stem>-<k>.npz')
    ap.add_argument('--checkpoint-every', type=int, default=0, help='save a numbered checkpoint every N iterations')
    ap.add_argument('--arch', default='tf', choices=['mlp', 'v2', 'tf'])
    ap.add_argument('--tf-layers', type=int, default=3, help='tf: transformer layers (--token-dim is the width)')
    ap.add_argument('--tf-heads', type=int, default=4, help='tf: attention heads')
    ap.add_argument('--tf-compact', action=argparse.BooleanOptionalAction, default=True,
                    help='tf: 17 tokens, actives and foe moves folded into Pokemon')
    ap.add_argument('--damage', action=argparse.BooleanOptionalAction, default=True,
                    help='tf: damage ranges as inputs (advsim/damage.py, obs v3)')
    ap.add_argument('--minutes', type=float, default=0, help='stop after this much training time (0: --iters)')
    ap.add_argument('--eval-every', type=int, default=100, help='iterations between evaluations, written to <save stem>_curve.jsonl (0: none)')
    ap.add_argument('--eval-games', type=int, default=2048, help='games per evaluation opponent')
    ap.add_argument('--eval-ref', default='long', help='a reference network (tag or .npz path) evaluated against, besides maxdamage; skipped if missing')
    ap.add_argument('--eval-seed', type=int, default=7, help='the evaluation teams: fixed, so runs of any --seed compare')
    ap.add_argument('--no-amp', action='store_true', help='fp32 throughout: no TF32, no bf16')
    ap.add_argument('--profile', action='store_true', help='print where each logged iteration\'s time went')
    ap.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True, help='torch.compile the learner and the snapshots')
    ap.add_argument('--hidden', type=int, default=512, help='mlp: layer width; v2: trunk width')
    ap.add_argument('--token-dim', type=int, default=96, help='tf: the width; v2: the Pokemon encoding size')
    ap.add_argument('--opp-action', type=float, default=0.1,
                    help="tf: weight of the prediction of the foe's action this turn, which the network reacts to (0: off)")
    ap.add_argument('--anneal', action='store_true', help='decay the learning rate linearly to 0 over the run')
    ap.add_argument('--init', default=None, help='start from this checkpoint (tag or .npz path) instead of from scratch')
    ap.add_argument('--layers', type=int, default=2, help='mlp: hidden layers')
    ap.add_argument('--log-every', type=int, default=10, help='iterations between log lines')
    args = ap.parse_args(argv)
    if args.arch != 'tf':  # the transformer's options
        args.opp_action, args.damage = 0.0, False
    if args.iters is None:
        args.iters = 1 << 62 if args.minutes else 1000

    torch.manual_seed(args.seed)
    torch.distributions.Distribution.set_default_validate_args(False)  # validation waits on the GPU
    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    B, T = args.batch, args.steps
    env = Gen3Env(B, device=dev, max_steps=args.max_steps)
    env.reset(seed=args.seed)
    baselines = Baselines(env)
    cuda = torch.device(dev).type == 'cuda'
    if not args.no_amp:
        torch.set_float32_matmul_precision('high')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda and not args.no_amp)
    if args.init:
        net = net_.load(players.checkpoint(args.init), dev).train()
        if args.damage and isinstance(net, PolicyTF) and not net.damage:  # widen: the same function
            net = with_damage(net)
        if not getattr(net, 'react', False):  # the checkpoint's architecture, not --arch, decides
            args.opp_action = 0.0
    else:
        if args.arch == 'tf':  # --opp-action makes it react to its own prediction
            net = PolicyTF(d=args.token_dim, layers=args.tf_layers, heads=args.tf_heads, react=bool(args.opp_action),
                           compact=args.tf_compact, damage=args.damage)
        elif args.arch == 'v2':
            net = net_.PolicyV2(width=args.hidden, d=args.token_dim)
        else:
            net = Policy(args.hidden, args.layers)
        net = net.to(dev)
    model = net  # the module to save, snapshot and evaluate; `net` may be its compiled form
    if args.compile:
        net = torch.compile(model, dynamic=True)
    runner = Runner(model, args.compile)
    evaluator = Evaluator(args.eval_games, dev, args.eval_seed, amp) if args.eval_every else None
    ref_path = players.checkpoint(args.eval_ref)
    ref = net_.load(ref_path, dev) if evaluator and ref_path.exists() else None
    if evaluator and not ref:
        print(f'{ref_path} missing: evaluating against maxdamage only', flush=True)
    curve = []
    started = time.time()
    watch = Stopwatch(args.profile, cuda)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    league = League(B, [float(x) for x in args.mix.split(',')], args.pool, dev, gen)
    slots = torch.arange(B, device=dev)
    tick = 0

    for it in range(args.iters):
        t0 = time.time()
        out_of_time = args.minutes and time.time() - started > args.minutes * 60
        if args.anneal:
            done_frac = (time.time() - started) / (args.minutes * 60) if args.minutes else it / args.iters
            for group in opt.param_groups:
                group['lr'] = args.lr * max(1 - done_frac, 0.05)
        if it and it % args.snapshot_every == 0:
            league.snapshot(model)
        buf = {k: [] for k in ('obs', 'legal', 'act', 'logp', 'val', 'rew', 'done', 'trunc', 'mine', 'foe_act')}
        watch.lap()
        with torch.no_grad():
            for _ in range(T):
                obs, mask = env.observe()
                obs = obs.clone()
                legal = legal_bits(mask)
                watch.lap('observe')
                with amp():
                    dist, val = net(obs.view(B * 2, -1), legal.view(B * 2, -1))
                act = dist.sample().view(B, 2)
                logp = dist.log_prob(act.view(-1)).view(B, 2)
                val = val.view(B, 2)
                acts = act.clone()
                foe = 1 - league.side
                watch.lap('learner forward')
                for k, s in league.groups():
                    f = foe[s]
                    if k < args.pool:
                        with amp():
                            acts[s, f] = runner(league.pool[k], obs[s, f], legal[s, f])[0].sample()
                    else:
                        acts[s, f] = baselines.act(s, f, league.opp[s] - BASE, args.seed, tick)
                watch.lap('opponents')
                tick += 1
                mine = learner_mask(league.opp, league.side) & (legal.sum(-1) > 1)
                if args.opp_action:
                    buf['foe_act'].append(foe_action(env, acts, legal))
                env.step(acts)
                watch.lap('env step')
                r = env.reward.clone()
                done, trunc = env.done.bool().clone(), env.truncated.bool().clone()
                rew = torch.stack([r, -r], dim=-1)
                league.record(done & (league.opp != SELF), rew[slots, league.side])
                for k, x in (('obs', obs), ('legal', legal), ('act', act), ('logp', logp), ('val', val),
                             ('rew', rew), ('done', done), ('trunc', trunc), ('mine', mine)):
                    buf[k].append(x)
                league.draw(done | trunc)
                watch.lap('bookkeeping')
            obs, mask = env.observe()
            with amp():
                _, last_val = net(obs.reshape(B * 2, -1), legal_bits(mask).view(B * 2, -1))
        buf = {k: torch.stack(v) for k, v in buf.items() if v}
        adv, ret = gae(buf['rew'], buf['val'], buf['done'], buf['trunc'], last_val.view(B, 2), args.gamma, args.lam)

        sel = buf['mine'].reshape(-1)
        flat = {k: v.reshape(T * B * 2, *v.shape[3:])[sel] for k, v in buf.items()
                if k in ('obs', 'legal', 'act', 'logp', 'foe_act')}
        a, r_ = adv.reshape(-1)[sel], ret.reshape(-1)[sel]
        a = (a - a.mean()) / (a.std() + 1e-8)
        n = a.shape[0]
        watch.lap('gae')
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for s0 in range(0, n, args.minibatch):
                mb = perm[s0:s0 + args.minibatch]
                opt.zero_grad()
                for c0 in range(0, len(mb), args.micro or len(mb)):  # gradient accumulation: the same step
                    i = mb[c0:c0 + (args.micro or len(mb))]
                    with amp():
                        if args.opp_action:
                            dist, v, extras = net(flat['obs'][i], flat['legal'][i], opp_action=True)
                        else:
                            dist, v = net(flat['obs'][i], flat['legal'][i])
                    ratio = (dist.log_prob(flat['act'][i]) - flat['logp'][i]).exp()
                    pg = -torch.min(ratio * a[i], ratio.clamp(0.8, 1.2) * a[i]).mean()
                    vf = ((v - r_[i]) ** 2).mean()
                    if args.opp_action:
                        vf = vf + args.opp_action * foe_action_loss(extras['opp_action'], flat['foe_act'][i])
                    ent = dist.entropy().mean()
                    loss = pg + 0.5 * vf - args.ent * ent
                    (loss * (len(i) / len(mb))).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        watch.lap('update')
        assert torch.isfinite(loss), 'a non-finite loss'
        assert not env.err.any(), 'an illegal action reached the engine'
        if it % args.log_every == 0 or it == args.iters - 1:
            wr = ' '.join(f'{name[:6]} {float(w):.2f}' for name, w in zip(NAMES, league.base_wr))
            snaps = f'snaps {float(league.snap_wr[:len(league.pool)].mean()):.2f}' if league.pool else 'snaps -'
            sps = T * B / (time.time() - t0)
            print(f'iter {it:4d}  transitions {n:7d}  loss {loss.item():+.3f}  entropy {ent.item():.3f}  '
                  f'win vs {wr}  {snaps}  {sps:,.0f} decisions/s', flush=True)
            if args.profile:
                print('  time: ' + watch.report(), flush=True)
        last = it == args.iters - 1 or out_of_time
        if evaluator and (it % args.eval_every == args.eval_every - 1 or last):
            model.eval()
            row = {'iter': it + 1, 'minutes': round((time.time() - started) / 60, 2),
                   'maxdamage': evaluator.score(model, 'maxdamage')}
            if ref is not None:
                row[args.eval_ref] = evaluator.score(model, ref)
            model.train()
            curve.append(row)
            print('eval ' + '  '.join(f'{k} {v}' for k, v in row.items()), flush=True)
            if args.save:
                stem = pathlib.Path(args.save)
                fileio.write_text(stem.with_name(stem.stem + '_curve.jsonl'), ''.join(json.dumps(r) + '\n' for r in curve))
                net_.save(model, stem.with_name(f'{stem.stem}-e{it + 1}.npz'))
        if args.save and (it % 100 == 99 or last):
            net_.save(model, args.save)
        if args.save and args.checkpoint_every and (it + 1) % args.checkpoint_every == 0:
            k = (it + 1) // 1000 if (it + 1) % 1000 == 0 else it + 1
            stem = pathlib.Path(args.save)
            net_.save(model, stem.with_name(f'{stem.stem}-{k}{"k" if (it + 1) % 1000 == 0 else ""}.npz'))
        if out_of_time:
            break
    if args.save:
        print(f'saved {args.save}')


if __name__ == '__main__':
    main()
