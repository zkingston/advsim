"""The league the trainer plays in: who each slot faces (self-play, past
snapshots, the five baselines), the runner that plays every snapshot through
one compiled network, and the evaluator that scores a network on fixed
mirrored couples. Used by tools/train_league.py and tools/policy_eval.py."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import warp as wp

from advsim import players
from advsim.engine import kernels, obs_layout
from advsim.env import Gen3Env
from advsim.players import BASELINES, Baselines

SELF = -1
BASE = 100  # opponent codes: -1 self, 0.. a snapshot's pool index, 100 + k baseline k
NAMES = list(BASELINES)



def learner_mask(opp: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """[B, 2]: which sides the learner plays. Both in self-play, else its own."""
    one = torch.nn.functional.one_hot(side, 2).bool()
    return torch.where((opp == SELF).unsqueeze(-1), torch.ones_like(one), one)


class League:
    """Who each slot plays, and how the learner is doing against each."""

    def __init__(self, B: int, mix: list[float], pool: int, device, gen: torch.Generator) -> None:
        self.mix = torch.tensor(mix, device=device)
        self.gen, self.device, self.pool_size = gen, device, pool
        self.pool: list[dict] = []
        self.base_wr = torch.full((len(NAMES),), 0.5, device=device)
        self.snap_wr = torch.full((pool,), 0.5, device=device)
        self.opp = torch.full((B,), SELF, dtype=torch.long, device=device)
        self.side = torch.zeros(B, dtype=torch.long, device=device)
        self.draw(torch.ones(B, dtype=torch.bool, device=device))

    def draw(self, which: torch.Tensor) -> None:
        """Deal new opponents to the `which` slots. Draws for every slot and
        keeps the chosen ones, so the step never waits on the GPU for a count."""
        B = len(which)
        kind = torch.multinomial(self.mix, B, replacement=True, generator=self.gen)
        if not self.pool:
            kind = torch.where(kind == 1, torch.zeros_like(kind), kind)  # no snapshots yet: self-play
        opp = torch.full((B,), SELF, dtype=torch.long, device=self.device)
        pfsp = lambda wr: (1 - wr).clamp(min=0.02) ** 2
        if self.pool:
            k = torch.multinomial(pfsp(self.snap_wr[:len(self.pool)]), B, replacement=True, generator=self.gen)
            opp = torch.where(kind == 1, k, opp)
        b = torch.multinomial(pfsp(self.base_wr), B, replacement=True, generator=self.gen)
        opp = torch.where(kind == 2, BASE + b, opp)
        self.opp = torch.where(which, opp, self.opp)
        side = torch.randint(0, 2, (B,), device=self.device, generator=self.gen)
        self.side = torch.where(which, side, self.side)

    def snapshot(self, net: nn.Module) -> None:
        """Keep the learner's weights; a snapshot plays through `Runner`."""
        frozen = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if len(self.pool) < self.pool_size:
            self.pool.append(frozen)
            self.snap_wr[len(self.pool) - 1] = 0.5
        else:  # the oldest goes
            self.pool = self.pool[1:] + [frozen]
            self.snap_wr = torch.cat([self.snap_wr[1:], torch.full((1,), 0.5, device=self.device)])
            self.opp = torch.where((self.opp >= 0) & (self.opp < BASE), (self.opp - 1).clamp(min=0), self.opp)

    def record(self, finished: torch.Tensor, learner_reward: torch.Tensor, rate: float = 0.05) -> None:
        """Fold this step's finished games into the running win rates: each
        opponent's rate moves toward its mean score this step, if it had games."""
        score = (learner_reward + 1) / 2  # win 1, tie 0.5, loss 0
        for wr, lo, n in ((self.base_wr, BASE, len(NAMES)), (self.snap_wr, 0, self.pool_size)):
            mine = finished & (self.opp >= lo) & (self.opp < lo + n)
            bin_ = torch.where(mine, self.opp - lo, n)  # everything else lands in bin n
            games = torch.zeros(n + 1, device=self.device).index_add_(0, bin_, torch.ones_like(score))[:n]
            total = torch.zeros(n + 1, device=self.device).index_add_(0, bin_, score)[:n]
            wr += rate * (total / games.clamp(min=1) - wr) * (games > 0)

    def groups(self) -> list[tuple[int, torch.Tensor]]:
        """(group, slots) for each snapshot index and then the baselines (group
        `pool_size`) that some slot plays, with one wait on the GPU for all of them."""
        g = torch.where(self.opp >= BASE, self.pool_size, torch.where(self.opp >= 0, self.opp, self.pool_size + 1))
        order = torch.argsort(g, stable=True)
        counts = torch.bincount(g, minlength=self.pool_size + 2).tolist()
        out, start = [], 0
        for k, c in enumerate(counts[:-1]):
            if c:
                out.append((k, order[start:start + c]))
            start += c
        return out


class Runner:
    """Plays every snapshot through one (compiled) copy of the network, loading
    each snapshot's weights in place before its forward pass: separate copies
    ran uncompiled and cost a fifth of the loop once the pool filled."""

    def __init__(self, model: nn.Module, compile_: bool) -> None:
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.weights = self.model.state_dict()  # views of the live parameters
        self.names, self.live = list(self.weights), list(self.weights.values())
        self.net = torch.compile(self.model, dynamic=True) if compile_ else self.model

    def __call__(self, snapshot: dict, obs: torch.Tensor, legal: torch.Tensor):
        torch._foreach_copy_(self.live, [snapshot[k] for k in self.names])  # one launch, not one per tensor
        return self.net(obs, legal)


class Evaluator:
    """Mirrored couples on fixed teams against one opponent: the learner's score."""

    def __init__(self, games: int, device: str, seed: int, amp) -> None:
        self.env = Gen3Env(games, device=device, rng_mode='paired')
        self.games, self.seed, self.device, self.amp = games, seed, device, amp
        self.baselines = Baselines(self.env)
        self.legal = wp.zeros((games, 2), dtype=wp.int32, device=device)
        self.obs = wp.zeros((games, 2, obs_layout.OBS_DIM), dtype=wp.int16, device=device)
        self.acts = wp.zeros((games, 2), dtype=wp.int32, device=device)
        self.log = wp.zeros((games, 1), dtype=wp.uint32, device=device)

    @torch.no_grad()
    def score(self, net, opponent, max_decisions: int = 1000) -> float:
        """`opponent` is a baseline name or a network. Games still going at
        the cap count as ties."""
        env, G, dev = self.env, self.games, self.device
        env.reset(seed=self.seed)
        players.mirror_couples(env, G)
        mine = torch.arange(G, device=dev) % 2  # the learner is p1 in even games, p2 in odd
        result = wp.to_torch(env.arrays['result'])
        for tick in range(max_decisions):
            live = torch.nonzero(result == 0).squeeze(-1)
            if not len(live):
                break
            wp.launch(kernels.legal_mask, dim=(G, 2), device=dev, inputs=[env.state, env.dex, self.legal])
            wp.launch(kernels.obs, dim=(G, 2), device=dev, inputs=[env.state, env.dex, self.obs])
            legal, obs = wp.to_torch(self.legal), wp.to_torch(self.obs)
            g, s = live.repeat_interleave(2), torch.tensor([0, 1], device=dev).repeat(len(live))
            m = legal[g, s]
            acts = wp.to_torch(self.acts)
            acts[g, s] = torch.log2(m.clamp(min=1).float()).long().int()
            real = (m & (m - 1)) != 0
            for who, player in ((s == mine[g], net), (s != mine[g], opponent)):
                sel = real & who
                if not sel.any():
                    continue
                gs, ss = g[sel], s[sel]
                if isinstance(player, str):
                    a = self.baselines.act(gs, ss, torch.full_like(gs, BASELINES[player]), self.seed, tick)
                else:
                    with self.amp():
                        a = players.net_act(player, obs[gs, ss], legal[gs, ss])
                acts[gs, ss] = a.int()
            wp.launch(kernels.step_idx, dim=len(live), device=dev,
                      inputs=[env.state, env.dex, self.acts, self.log, wp.from_torch(live.int().contiguous())])
        res = result.long()
        won = torch.where(mine == 0, res == 1, res == 2).float()
        lost = torch.where(mine == 0, res == 2, res == 1).float()
        return float((won + (1 - won - lost) * 0.5).mean())
