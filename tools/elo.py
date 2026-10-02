"""The Elo bracket: every player against every other, all games at once on the GPU.

    uv run --group rl python tools/elo.py --games 2048
    uv run --group rl python tools/elo.py --games 512 --selfplay search-32  # p1 should score ~0.5
    uv run --group rl python tools/elo.py --add search-64-net             # one new player
    uv run --group rl python tools/elo.py --ratings artifacts/elo/games.jsonl  # re-rate

Every pair plays `--games` games in one arena, all pairs together, in mirrored
couples: both games of a couple deal the same two teams and the same RNG key
(`paired` mode, so both see the same draws turn by turn), with the players
swapped between sides. Each decision, every player acts for all its games at
once: one kernel for the five baselines, one forward pass for PPO, and search
in waves of up to `--capacity` trees. A game still going after
`--max-decisions` counts as a tie.

Ratings are a Bradley-Terry fit over every game (a tie is half a win each),
anchored at random = 1000, with 95% intervals from resampling couples. So a
new player need not rerun the bracket: `--add NAME` plays it against every
player already in the games file, appends those games and refits everything.

`search-N` searches N iterations per decision with random rollouts and an
HP-fraction leaf; `search-N-net` scores leaves with the PPO value head instead,
and `search-N-pnet` also takes the network's policy as every node's prior.
A `-tNN` suffix plays a search's root strategy at temperature NN/100, `-mix`
the geometric mean of the equilibrium and the network's root policy; `-gNN`
sets the prior's weight and `-pruneNN` the pruning ratio, in hundredths.
A network player takes a checkpoint tag after a colon: `ppo:league` and
`search-64-net:league` use artifacts/ppo_league.npz; no tag is the first
baseline, artifacts/ppo_baseline.npz. `ppo-t50:v2` plays that network at
temperature 0.5, `ppo-greedy:v2` takes its top action.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advsim import fileio  # noqa: E402
from advsim.players import BASELINES  # noqa: E402

PLAYERS = ['random', 'emerald', 'maxdamage', 'status', 'switchaverse', 'ppo', 'search-32', 'search-64', 'search-128']
OUT = fileio.ARTIFACTS / 'elo'


def search_spec(name: str) -> tuple[int, bool, bool]:
    """'search-64-pnet-t25:league' -> (64 iterations, value network, policy prior)."""
    flags = set(name.split(':')[0].split('-')[2:])
    return int(name.split('-')[1]), bool(flags & {'net', 'pnet'}), 'pnet' in flags


def temperature_of(name: str) -> float:
    """'ppo-t50:v2' -> 0.5, 'search-64-pnet-t25:v2' -> 0.25, 'ppo-greedy:v2' -> ~0, 'ppo:v2' -> 1."""
    base = name.split(':')[0]
    if base.endswith('-greedy'):
        return 1e-3
    for part in base.split('-'):
        if part.startswith('t') and part[1:].isdigit():
            return int(part[1:]) / 100
    return 1.0


def final_of(name: str) -> str:
    flags = set(name.split(':')[0].split('-'))
    return 'mix' if 'mix' in flags else 'eq'


def knobs_of(name: str) -> tuple[float, float]:
    """'search-64-pnet-t25-mix-g25-prune02:night' -> (0.25, 0.02): the prior's
    weight and the pruning ratio (engine/tree.py), in hundredths;
    mcts.DEFAULT_KNOBS where a name has none."""
    from advsim.search.mcts import DEFAULT_KNOBS
    knobs = list(DEFAULT_KNOBS)
    for part in name.split(':')[0].split('-')[2:]:
        for i, key in enumerate(('g', 'prune')):
            if part.startswith(key) and part[len(key):].isdigit():
                knobs[i] = int(part[len(key):]) / 100
    return tuple(knobs)


def tag_of(name: str) -> str | None:
    return name.split(':')[1] if ':' in name else None


def order(p: str) -> tuple:
    base = p.split(':')[0]
    return (PLAYERS.index(base) if base in PLAYERS else 99, p)


def play(args, pairs: list[tuple[str, str]]) -> list[dict]:
    import torch
    import warp as wp

    from advsim import net as net_, players as players_
    from advsim.engine import kernels, obs_layout
    from advsim.env import Gen3Env
    from advsim.search.mcts import Search

    names = sorted({p for pair in pairs for p in pair}, key=order)
    idx = {n: i for i, n in enumerate(names)}
    per = args.games
    G = len(pairs) * per
    searchers = [n for n in names if n.startswith('search-')]
    netted = [n for n in names if n.startswith('ppo') or (n in searchers and search_spec(n)[1])]
    cap = args.capacity if searchers else 0
    dev = args.device
    env = Gen3Env(G + cap * args.lanes, device=dev, rng_mode='paired')
    env.reset(seed=args.seed)
    players_.mirror_couples(env, G)  # game 2k+1 is game 2k with the players swapped

    owner = np.zeros((G, 2), np.int32)
    for p, (a, b) in enumerate(pairs):
        g = np.arange(p * per, (p + 1) * per)
        owner[g] = np.where((g % 2 == 0)[:, None], [idx[a], idx[b]], [idx[b], idx[a]])
    kind = np.array([BASELINES.get(n, -1) for n in names])
    search = Search(env, cap, lane_base=G, lanes=args.lanes, nodes=args.lanes * max(
        search_spec(n)[0] for n in searchers), depth=args.depth, rollout=args.rollout,
        key=args.seed) if searchers else None
    tags = {tag_of(n) for n in netted}
    nets = {tag: net_.load(players_.checkpoint(tag), dev) for tag in tags}
    baselines = players_.Baselines(env)

    rng = np.random.default_rng(args.seed)
    legal_buf = wp.zeros((G, 2), dtype=wp.int32, device=dev)
    obs_buf = wp.zeros((G, 2, obs_layout.OBS_DIM), dtype=wp.int16, device=dev)
    acts = wp.zeros((env.batch, 2), dtype=wp.int32, device=dev)
    log = wp.zeros((env.batch, 1), dtype=wp.uint32, device=dev)
    decisions = np.zeros(G, np.int32)
    t0 = time.time()
    for tick in range(args.max_decisions):
        result = env.numpy('result')[:G]
        live = np.nonzero(result == 0)[0]
        if not len(live):
            break
        wp.launch(kernels.legal_mask, dim=(G, 2), device=dev, inputs=[env.state, env.dex, legal_buf])
        legal = legal_buf.numpy()
        choice = np.full((G, 2), 10, np.int32)
        g_all, s_all = np.repeat(live, 2), np.tile([0, 1], len(live))
        who = owner[g_all, s_all]
        # A side with one legal action takes it; only real choices go to a player.
        single = (legal[g_all, s_all] & (legal[g_all, s_all] - 1)) == 0
        choice[g_all[single], s_all[single]] = np.log2(np.maximum(legal[g_all[single], s_all[single]], 1)).astype(np.int32)
        todo = ~single
        heur = todo & (kind[who] >= 0)
        if heur.any():
            t_ = lambda x: torch.as_tensor(x, device=dev)
            choice[g_all[heur], s_all[heur]] = baselines.act(
                t_(g_all[heur]), t_(s_all[heur]), t_(kind[who[heur]]), args.seed, tick).cpu().numpy()
        observed = False
        for n in netted:
            mine = todo & (who == idx[n])
            if n in searchers or not mine.any():
                continue
            if not observed:
                wp.launch(kernels.obs, dim=(G, 2), device=dev, inputs=[env.state, env.dex, obs_buf])
                observed = True
            gi = torch.as_tensor(g_all[mine], device=dev)
            si = torch.as_tensor(s_all[mine], device=dev)
            player = nets[tag_of(n)]
            if temperature_of(n) != 1.0:
                player = net_.Tempered(player, temperature_of(n))
            choice[g_all[mine], s_all[mine]] = players_.net_act(
                player, wp.to_torch(obs_buf)[gi, si], wp.to_torch(legal_buf)[gi, si]).cpu().numpy()
        for n in searchers:
            mine = todo & (who == idx[n])
            if mine.any():
                search.key = args.seed * 1000003 + tick
                iters, with_net, prior = search_spec(n)
                choice[g_all[mine], s_all[mine]] = search.decide(
                    g_all[mine], s_all[mine], legal[g_all[mine], s_all[mine]], iters, rng,
                    net=nets[tag_of(n)] if with_net else None, prior=prior, temperature=temperature_of(n),
                    final=final_of(n), knobs=knobs_of(n))
        host = np.zeros((env.batch, 2), np.int32)
        host[:G] = choice
        acts.assign(host)
        wp.launch(kernels.step_idx, dim=len(live), device=dev,
                  inputs=[env.state, env.dex, acts, log, wp.array(live.astype(np.int32), dtype=wp.int32, device=dev)])
        decisions[live] += 1
        if tick % 25 == 0:
            print(f'decision {tick:4d}: {len(live):6d} games live, {time.time() - t0:6.0f}s', flush=True)
    err = env.numpy('err')[:G]
    if err.any():
        raise RuntimeError(f'{int((err != 0).sum())} games set an error bit')
    result = env.numpy('result')[:G]
    print(f'{G} games in {time.time() - t0:.0f}s; {int((result == 0).sum())} hit the decision cap', flush=True)
    rows = []
    for p, (a, b) in enumerate(pairs):
        for g in range(p * per, (p + 1) * per):
            rows.append({'pair': p, 'couple': g // 2, 'a': a, 'b': b, 'a_side': g % 2,
                         'result': int(result[g]), 'decisions': int(decisions[g])})
    return rows


def score_a(row: dict) -> float:
    """A's score: 1 a win, 0.5 a tie or an unfinished game, 0 a loss."""
    r = row['result']
    if r in (0, 3):
        return 0.5
    return 1.0 if (r == 1) == (row['a_side'] == 0) else 0.0


def bradley_terry(names: list[str], rows: list[dict], iters: int = 2000) -> np.ndarray:
    """Strengths by Hunter's MM algorithm; ties are half a win each."""
    k = {n: i for i, n in enumerate(names)}
    n = len(names)
    wins = np.zeros((n, n))
    for r in rows:
        s = score_a(r)
        wins[k[r['a']], k[r['b']]] += s
        wins[k[r['b']], k[r['a']]] += 1 - s
    games = wins + wins.T
    gamma = np.ones(n)
    for _ in range(iters):
        denom = (games / (gamma[:, None] + gamma[None, :])).sum(1)
        gamma = np.maximum(wins.sum(1), 1e-3) / np.maximum(denom, 1e-12)
        gamma /= np.exp(np.log(gamma).mean())
    return gamma


def ratings(rows: list[dict], anchor: str = 'random', boot: int = 200, seed: int = 0) -> dict:
    names = sorted({r['a'] for r in rows} | {r['b'] for r in rows}, key=order)
    elo = lambda g: 400 * np.log10(g) - 400 * np.log10(g[names.index(anchor)]) + 1000
    point = elo(bradley_terry(names, rows))
    couples = {}
    for r in rows:
        couples.setdefault((r['pair'], r['couple']), []).append(r)
    keys = list(couples)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(boot):
        pick = rng.integers(0, len(keys), len(keys))
        samples.append(elo(bradley_terry(names, [r for i in pick for r in couples[keys[i]]], iters=500)))
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    matrix = {a: {b: None for b in names} for a in names}
    for a in names:
        for b in names:
            s = [score_a(r) if r['a'] == a else 1 - score_a(r) for r in rows
                 if (r['a'], r['b']) in ((a, b), (b, a)) and a != b]
            matrix[a][b] = float(np.mean(s)) if s else None
    return {'players': names, 'elo': point.tolist(), 'lo': lo.tolist(), 'hi': hi.tolist(), 'matrix': matrix,
            'games': len(rows)}


def report(summary: dict) -> None:
    order = np.argsort(summary['elo'])[::-1]
    print(f"\n{'player':14s} {'elo':>6s}   95% interval")
    for i in order:
        p = summary['players'][i]
        print(f"{p:14s} {summary['elo'][i]:6.0f}   [{summary['lo'][i]:5.0f}, {summary['hi'][i]:5.0f}]")
    names = [summary['players'][i] for i in order]
    print('\nrow player\'s score against column player')
    print(' ' * 14 + ''.join(f'{n[:8]:>9s}' for n in names))
    for a in names:
        row = summary['matrix'][a]
        print(f'{a:14s}' + ''.join('        -' if row[b] is None else f'{row[b]:9.3f}' for b in names))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--games', type=int, default=2048, help='per pair; even')
    ap.add_argument('--players', default=','.join(PLAYERS), help='comma-separated; names as in the module docstring')
    ap.add_argument('--selfplay', default=None, help='one player against itself only: a fairness check')
    ap.add_argument('--seed', type=int, default=1, help='the deals and the RNG keys')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--capacity', type=int, default=4096, help='search trees per wave (8192 needs more than 12 GB with a transformer)')
    ap.add_argument('--lanes', type=int, default=8, help='search: descents per tree per iteration')
    ap.add_argument('--depth', type=int, default=6, help='search: the longest descent')
    ap.add_argument('--rollout', type=int, default=16, help='search without a network: random decisions past the tree')
    ap.add_argument('--max-decisions', type=int, default=1000, help='a game still going after this many is a draw')
    ap.add_argument('--out', default=str(OUT / 'games.jsonl'), help='the games file; summary.json goes next to it')
    ap.add_argument('--ratings', default=None, help='rate an existing games file instead of playing')
    ap.add_argument('--add', default=None, help='play one new player against everyone in --out, append, refit')
    ap.add_argument('--against', default=None, help='with --add: only these opponents (comma-separated), for sweeps')
    ap.add_argument('--anchor', default='random', help='the player rated 1000')
    args = ap.parse_args(argv)
    if args.games % 2:
        ap.error('--games must be even: games come in mirrored couples')

    out = pathlib.Path(args.out)
    if args.ratings:
        rows = list(fileio.iter_jsonl(pathlib.Path(args.ratings)))
    elif args.add:
        if not out.exists():
            ap.error(f'--add plays against the games in --out ({out}), which does not exist: run the bracket first')
        old = list(fileio.iter_jsonl(out))
        field = (args.against.split(',') if args.against
                 else sorted({r['a'] for r in old} | {r['b'] for r in old} - {args.add}))
        new = play(args, [(args.add, p) for p in field])
        pair0, couple0 = max(r['pair'] for r in old) + 1, max(r['couple'] for r in old) + 1
        for r in new:
            r['pair'] += pair0
            r['couple'] += couple0
        rows = [r for r in old if args.add not in (r['a'], r['b'])] + new
        fileio.write_text(out, ''.join(json.dumps(r) + '\n' for r in rows))
    else:
        players = args.players.split(',')
        pairs = [(args.selfplay, args.selfplay)] if args.selfplay else list(itertools.combinations(players, 2))
        rows = play(args, pairs)
        fileio.write_text(out, ''.join(json.dumps(r) + '\n' for r in rows))
    if args.selfplay:
        # By symmetry either copy scores one half; what can go wrong is a side:
        # a player broken as p2 hands p1 the games.
        p1 = np.mean([1.0 if r['result'] == 1 else 0.5 if r['result'] in (0, 3) else 0.0 for r in rows])
        print(f'{args.selfplay} against itself: p1 scores {p1:.3f} over {len(rows)} games')
        return 0
    summary = ratings(rows, anchor=args.anchor)
    fileio.write_json(pathlib.Path(args.ratings or out).with_name('summary.json'), summary)
    report(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
