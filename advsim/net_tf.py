"""PolicyTF: a transformer over the observation's tokens.

24 tokens: the 12 Pokemon, the own active's 4 moves (in slot order), the foe
active's 4 revealed moves (in id order), the 2 actives' volatile state, the
field (with each side's last move), and a summary token. Pre-RMSNorm layers;
each token type (Pokemon, move, active, global) has its own feed-forward
weights. Absent Pokemon and empty move slots are masked out of attention.
Heads point as in PolicyV2: a move's logit comes from its move token, a
switch's from the Pokemon token it sends in, each read with the summary.

With `compact`, 17 tokens: each active's volatile state is added into its
Pokemon's token, the foe active's revealed moves into the foe active's, and
the field into the summary; about 30% less work per row.

With `damage`, each move token also reads its damage range (the least and
most one use takes off its target, advsim/damage.py) and each own Pokemon
token the most it could deal to the foe active and take from its revealed
moves.

With `react`, the first half of the layers ends in a prediction of the
opponent's action this turn (its move id, or the species it switches to:
the `--opp-action` labels), and the prediction enters the second half as one
more token, so the policy can answer what it expects the opponent to do.

`config` is (d, layers, heads, react, compact, damage); a checkpoint saved
before `compact` or `damage` existed has a shorter one, and reads as off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from advsim import damage as damage_
from advsim.engine import obs_layout as L
from advsim.engine._generated import ids
from advsim.net import ACTIVE_BOOSTS, ACTIVE_FLAGS, FIELD_NUMERIC, MON_NUMERIC, M, col

MONS, MOVES, ACTIVES = slice(0, 12), slice(12, 20), slice(20, 22)
MONS_OWN, MONS_FOE = slice(0, 6), slice(6, 12)
GROUPS = (MONS, MOVES, ACTIVES, slice(22, None))  # the last: field, summary and the prediction
FF_KIND = torch.tensor([0] * 12 + [1] * 8 + [2] * 2 + [3] * 3)  # each token's feed-forward, by GROUPS
N_PRED = ids.N_MOVES + ids.N_SPECIES
# One learned embedding per token type, never per position: party order and move slot order carry no meaning.
KINDS = [0] * 6 + [1] * 6 + [2] * 4 + [3] * 4 + [4, 5, 6, 7, 8]  # own/foe Pokemon, own/foe move, actives, field,
# summary, prediction
# compact: 17 tokens, the actives' state added into their Pokemon, the foe's moves into its active, the field
# into the summary; the own moves stay tokens, since the move heads point at them.
COMPACT_FF_KIND = [0] * 12 + [1] * 4 + [3] * 2
COMPACT_KINDS = [0] * 6 + [1] * 6 + [2] * 4 + [7, 8]


class RMSNorm(nn.Module):
    """nn.RMSNorm's arithmetic, in fp32, returning the input's dtype: under autocast
    nn.RMSNorm returns fp32, which kept the whole residual stream in fp32 and
    doubled the bytes of every memory-bound op around it."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + torch.finfo(torch.float32).eps)
        return (y * self.weight).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in GROUPS)

    def forward(self, x: torch.Tensor, keep: torch.Tensor, kind: torch.Tensor | None = None) -> torch.Tensor:
        """`kind` [T]: each token's feed-forward (GROUPS index); the full layout's by default."""
        n, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(n, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        x = x + self.out(a.transpose(1, 2).reshape(n, T, d))
        # Each token's feed-forward is its type's, as one batched matmul over the tokens
        # (token-major): per-type slices of the activation sent cuBLAS to slow strided kernels.
        kind = FF_KIND[:T].to(x.device) if kind is None else kind
        w1, b1 = (torch.stack([ff[0].weight for ff in self.ff])[kind],
                  torch.stack([ff[0].bias for ff in self.ff])[kind])
        w2, b2 = (torch.stack([ff[2].weight for ff in self.ff])[kind],
                  torch.stack([ff[2].bias for ff in self.ff])[kind])
        h = self.n2(x).transpose(0, 1)  # [T, n, d]
        z = F.gelu(torch.baddbmm(b1[:, None].to(h.dtype), h, w1.transpose(1, 2).to(h.dtype)))
        return x + torch.baddbmm(b2[:, None].to(z.dtype), z, w2.transpose(1, 2).to(z.dtype)).transpose(0, 1)


class PolicyTF(nn.Module):
    ARCH = 3

    def __init__(self, d: int = 96, layers: int = 3, heads: int = 4, react: bool = False,
                 compact: bool = False, damage: bool = False) -> None:
        super().__init__()
        self.config = (d, layers, heads, int(react), int(compact), int(damage))
        self.damage = damage
        self.features = damage_.Features() if damage else None
        self.compact, self.summary_at = compact, 16 if compact else 23
        self.register_buffer('ff_kind', torch.tensor(COMPACT_FF_KIND) if compact else FF_KIND.clone(), persistent=False)
        self.species = nn.Embedding(ids.N_SPECIES, 32)
        self.move = nn.Embedding(ids.N_MOVES, 32)
        self.ability = nn.Embedding(ids.N_ABILITIES, 16)
        self.item = nn.Embedding(ids.N_ITEMS, 8)
        self.status = nn.Embedding(ids.N_CONDITIONS, 8)
        self.types = nn.Embedding(18, 8)
        self.weather = nn.Embedding(4, 4)
        self.mon_in = nn.Linear(32 + 16 + 8 + 8 + 32 + len(MON_NUMERIC) + 1 + (2 if damage else 0), d)
        self.move_in = nn.Linear(32 + 1 + 2 + (2 if damage else 0), d)
        self.act_in = nn.Linear(len(ACTIVE_BOOSTS) + 16 + len(ACTIVE_FLAGS) + 32, d)
        self.field_in = nn.Linear(4 + len(FIELD_NUMERIC) + 64, d)
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        self.kind = nn.Embedding(9, d)
        self.register_buffer('kinds', torch.tensor(COMPACT_KINDS if compact else KINDS), persistent=False)
        first = layers // 2 if react else layers
        self.lower = nn.ModuleList(Block(d, heads) for _ in range(first))
        self.upper = nn.ModuleList(Block(d, heads) for _ in range(layers - first))
        self.react = react
        self.opp_action = nn.Sequential(RMSNorm(d), nn.Linear(d, N_PRED)) if react else None
        self.pred_in = nn.Linear(32 + 32 + 1, d) if react else None
        self.norm = RMSNorm(d)
        self.move_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.switch_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.other = nn.Linear(d, 2)  # pass, forced
        self.v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def tokens(self, obs: torch.Tensor):
        """([n, T, d] input tokens, [n, T] which take part in attention): T is 24, or 17 compact."""
        n, x = obs.shape[0], obs.long()
        sp_w, mv_w = self.species.weight, self.move.weight
        emb = F.embedding
        mon = x[:, L.MON_BASE:L.MON_BASE + 12 * M].view(n, 12, M)
        moves = mon[..., col('move0'):col('move0') + 4]
        pp = mon[..., col('pp0'):col('pp0') + 4].float() / 64
        numeric = torch.stack([mon[..., c].float() / s for c, s in MON_NUMERIC], -1)
        side = torch.cat([torch.zeros(n, 6, 1, device=obs.device), torch.ones(n, 6, 1, device=obs.device)], 1)
        mon_parts = [emb(mon[..., col('species')], sp_w), self.ability(mon[..., col('ability')]),
                     self.item(mon[..., col('item')]), self.status(mon[..., col('status')]), emb(moves, mv_w).mean(2),
                     numeric, side]
        if self.damage:
            dmg = self.features(obs)
            mon_parts.append(torch.cat([dmg['party'], torch.zeros_like(dmg['party'])], 1))  # own tokens only
        mons = self.mon_in(torch.cat(mon_parts, -1))
        present = mon[..., col('present')].bool()
        active = mon[..., col('active')] * mon[..., col('present')]  # [n, 12]
        act_moves = torch.stack([(moves[:, s] * active[:, s, None]).sum(1) for s in (MONS_OWN, MONS_FOE)], 1)
        act_pp = torch.stack([(pp[:, s] * active[:, s, None]).sum(1) for s in (MONS_OWN, MONS_FOE)], 1)
        mu = x[:, L.MATCHUP_BASE:L.MATCHUP_BASE + len(L.MATCHUP)].view(n, 2, 4, 2).float()
        mv_parts = [emb(act_moves, mv_w), act_pp.unsqueeze(-1), mu[..., :1] / 4, mu[..., 1:]]
        if self.damage:
            mv_parts.append(torch.stack([dmg['own_moves'], dmg['foe_moves']], 1))  # [n, 2, 4, 2], the matchup's order
        mv = self.move_in(torch.cat(mv_parts, -1))
        act = x[:, L.ACTIVE_BASE:L.ACTIVE_BASE + 2 * len(L.ACTIVE)].view(n, 2, len(L.ACTIVE))
        types = self.types(act[..., [L.ACTIVE.index('type0'), L.ACTIVE.index('type1')]]).flatten(-2)
        acts = self.act_in(torch.cat([act[..., ACTIVE_BOOSTS].float() / 6, types, act[..., ACTIVE_FLAGS].float(),
                                      emb(act[..., L.ACTIVE.index('choice_move')], mv_w)], -1))
        f = x[:, L.FIELD_BASE:L.FIELD_BASE + len(L.FIELD)]
        last = emb(x[:, L.HISTORY_BASE:L.HISTORY_BASE + 2], mv_w).flatten(1)
        field = self.field_in(torch.cat([self.weather(f[:, L.FIELD.index('weather')])]
                                        + [f[:, [c]].float() / s for c, s in FIELD_NUMERIC] + [last], -1))
        if self.compact:
            own, foe = active[:, MONS_OWN, None].float(), active[:, MONS_FOE, None].float()
            foe_moves = (mv[:, 1] * (act_moves[:, 1] > 0).unsqueeze(-1)).sum(1)
            mons = mons + torch.cat([own * acts[:, :1], foe * (acts[:, 1:] + foe_moves[:, None])], 1)
            toks = torch.cat([mons, mv[:, 0], self.summary + field[:, None]], 1)
            keep = torch.cat([present, act_moves[:, 0] > 0, torch.ones(n, 1, dtype=torch.bool, device=obs.device)], 1)
        else:
            toks = torch.cat([mons, mv.flatten(1, 2), acts, field[:, None], self.summary.expand(n, -1, -1)], 1)
            keep = torch.cat([present, act_moves.flatten(1) > 0, torch.ones(n, 4, dtype=torch.bool, device=obs.device)],
                             1)
        return toks + self.kind(self.kinds[:toks.shape[1]]), keep

    def encode(self, obs: torch.Tensor):
        """(final tokens [n, T, d], opponent-action logits or None)."""
        x, keep = self.tokens(obs)
        if torch.is_autocast_enabled(x.device.type):  # the residual stream in bf16, not fp32
            x = x.to(torch.get_autocast_dtype(x.device.type))
        for block in self.lower:
            x = block(x, keep, self.ff_kind[:x.shape[1]])
        pred = None
        if self.react:
            pred = self.opp_action(x[:, self.summary_at]).float()
            q = pred.softmax(-1).detach()  # the prediction as information: only its own loss trains it
            expect = torch.cat([q[:, :ids.N_MOVES] @ self.move.weight, q[:, ids.N_MOVES:] @ self.species.weight,
                                q[:, ids.N_MOVES:].sum(-1, keepdim=True)], -1)
            x = torch.cat([x, (self.pred_in(expect) + self.kind.weight[8]).to(x.dtype)[:, None]], 1)
            keep = torch.cat([keep, torch.ones_like(keep[:, :1])], 1)
        for block in self.upper:
            x = block(x, keep, self.ff_kind[:x.shape[1]])
        return self.norm(x), pred

    def forward(self, obs: torch.Tensor, legal: torch.Tensor, opp_action: bool = False):
        """(policy, value), or with `opp_action` (policy, value, extras) where
        extras['opp_action'] is the prediction's logits (`react` only)."""
        x, pred = self.encode(obs)
        s = x[:, self.summary_at:self.summary_at + 1]
        move_logits = self.move_head(torch.cat([x[:, 12:16], s.expand(-1, 4, -1)], -1)).squeeze(-1)
        switch_logits = self.switch_head(torch.cat([x[:, :6], s.expand(-1, 6, -1)], -1)).squeeze(-1)
        logits = torch.cat([move_logits, switch_logits, self.other(s[:, 0])], -1).float().masked_fill(~legal, -1e9)
        dist, v = torch.distributions.Categorical(logits=logits), self.v(s[:, 0]).float().squeeze(-1)
        if not opp_action:
            return dist, v
        return dist, v, {'opp_action': pred}

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self.encode(obs)[0][:, self.summary_at]).float().squeeze(-1)


def with_damage(net: PolicyTF) -> PolicyTF:
    """`net` reading the damage inputs too: their columns of the two input
    layers they feed start at zero, so it computes what it did until trained."""
    config = list(net.config)
    config[5] = 1  # damage
    wide = PolicyTF(*config).to(next(net.parameters()).device)
    state = net.state_dict()
    for name in ('mon_in.weight', 'move_in.weight'):  # the damage inputs are these layers' last columns
        w = torch.zeros_like(wide.state_dict()[name])
        w[:, :state[name].shape[1]] = state[name]
        state[name] = w
    wide.load_state_dict(state)
    return wide.train(net.training)
