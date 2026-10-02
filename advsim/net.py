"""The policy networks PPO trains and the bracket plays. A checkpoint is an
npz of the state dict, plus an `arch` array for anything but the first MLP.

- `Policy`: a plain MLP over the scaled int16 observation.
- `PolicyTF` (`net_tf.py`): a transformer over the observation's tokens.
- `PolicyV2`: built from obs_layout. Species, moves, abilities, items,
  statuses, types and weather are learned embeddings, not numbers; one shared
  encoder reads each of the 12 Pokemon tokens, so what it learns about a
  species or a move holds in any slot; and the heads point: a move's logit
  comes from the active Pokemon's encoding of that move, a switch's from the
  encoding of the Pokemon it sends in.

All three return fp32 logits and values under autocast, so PPO's ratios keep
their precision.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from advsim import fileio
from advsim.engine import obs_layout as L
from advsim.engine._generated import ids

ACTIONS = 12


def legal_bits(mask: torch.Tensor) -> torch.Tensor:
    """int32 [..] bitmasks to bool [.., 12]."""
    bits = torch.arange(ACTIONS, device=mask.device, dtype=torch.int32)
    return ((mask.unsqueeze(-1) >> bits) & 1).bool()


def mlp(*sizes: int) -> nn.Sequential:
    return nn.Sequential(*[m for a, b in zip(sizes, sizes[1:]) for m in (nn.Linear(a, b), nn.ReLU())])


class Policy(nn.Module):
    """A plain MLP over the int16 observation, scaled: `layers` hidden layers
    of `hidden` units, then a policy head and a value head."""

    def __init__(self, hidden: int = 512, layers: int = 2, inputs: int = L.OBS_DIM) -> None:
        super().__init__()
        self.body = mlp(inputs, *[hidden] * layers)
        self.pi = nn.Linear(hidden, ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, legal: torch.Tensor):
        h = self.body(self.inputs(obs))
        logits = self.pi(h).float().masked_fill(~legal, -1e9)
        return torch.distributions.Categorical(logits=logits), self.v(h).float().squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """The value head alone: the observing player's expected result."""
        return self.v(self.body(self.inputs(obs))).float().squeeze(-1)

    def inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """The columns it was trained on: later observation versions only append."""
        return obs[..., :self.body[0].in_features].float() / 64.0


M = len(L.MON)
col = lambda name: L.MON.index(name)
MON_NUMERIC = [(col(n), s) for n, s in (('present', 1), ('level', 100), ('hp_pct', 100), ('hp', 512),
                                        ('maxhp', 512), ('toxic_stage', 16), ('fainted', 1), ('active', 1),
                                        ('ability_known', 1), ('item_known', 1))]
ACTIVE_FLAGS = [L.ACTIVE.index(n) for n in ('substitute', 'confusion', 'leech_seed', 'encore', 'partial_trap',
                                            'yawn', 'perish', 'attract', 'transformed', 'flash_fire',
                                            'destiny_bond', 'charging', 'recharge', 'trapped')]
ACTIVE_BOOSTS = [L.ACTIVE.index(n) for n in L.ACTIVE if n.startswith('boost_')]
FIELD_NUMERIC = [(L.FIELD.index(n), s) for n, s in (('weather_turns', 8), ('turn', 100), ('spikes_own', 3),
                                                     ('spikes_foe', 3), ('wish_own', 1), ('wish_foe', 1))]


class PolicyV2(nn.Module):
    ARCH = 2

    def __init__(self, width: int = 512, d: int = 128) -> None:
        super().__init__()
        self.species = nn.Embedding(ids.N_SPECIES, 32)
        self.move = nn.Embedding(ids.N_MOVES, 32)
        self.ability = nn.Embedding(ids.N_ABILITIES, 16)
        self.item = nn.Embedding(ids.N_ITEMS, 8)
        self.status = nn.Embedding(ids.N_CONDITIONS, 8)
        self.types = nn.Embedding(18, 8)
        self.weather = nn.Embedding(4, 4)
        self.move_enc = mlp(32 + 1, 32)
        self.mon_enc = mlp(32 + 16 + 8 + 8 + 32 + len(MON_NUMERIC) + 1, d, d)
        self.act_enc = mlp(len(ACTIVE_BOOSTS) + 16 + len(ACTIVE_FLAGS) + 32, 64)
        self.trunk = mlp(6 * d + 2 * 64 + 4 + len(FIELD_NUMERIC) + len(L.MATCHUP), width, width)
        self.move_head = nn.Sequential(mlp(width + 32, 128), nn.Linear(128, 1))
        self.switch_head = nn.Sequential(mlp(width + d, 128), nn.Linear(128, 1))
        self.other = nn.Linear(width, 2)  # pass, forced
        self.v = nn.Linear(width, 1)

    def encode(self, obs: torch.Tensor):
        """(trunk features [n, width], own tokens [n, 6, d], own active's moves [n, 4, 32])."""
        n = obs.shape[0]
        x = obs.long()
        mon = x[:, L.MON_BASE:L.MON_BASE + 12 * M].view(n, 12, M)
        moves = mon[..., col('move0'):col('move0') + 4]
        pp = mon[..., col('pp0'):col('pp0') + 4].float().unsqueeze(-1) / 64
        mv = self.move_enc(torch.cat([self.move(moves), pp], -1))  # [n, 12, 4, 32]
        numeric = torch.stack([mon[..., c].float() / s for c, s in MON_NUMERIC], -1)
        side = torch.cat([torch.zeros(n, 6, 1, device=obs.device), torch.ones(n, 6, 1, device=obs.device)], 1)
        tok = self.mon_enc(torch.cat([self.species(mon[..., col('species')]), self.ability(mon[..., col('ability')]),
                                      self.item(mon[..., col('item')]), self.status(mon[..., col('status')]),
                                      mv.mean(2), numeric, side], -1))  # [n, 12, d]
        present = mon[..., col('present')].bool().unsqueeze(-1)
        active = (mon[..., col('active')] * mon[..., col('present')]).float().unsqueeze(-1)

        def pool(t, keep):
            mean = (t * keep).sum(1) / keep.sum(1).clamp(min=1)
            top = t.masked_fill(~keep, -1e4).amax(1).clamp(min=-1e3)
            return mean, top

        own_mean, own_max = pool(tok[:, :6], present[:, :6])
        foe_mean, foe_max = pool(tok[:, 6:], present[:, 6:])
        own_active = (tok[:, :6] * active[:, :6]).sum(1)
        foe_active = (tok[:, 6:] * active[:, 6:]).sum(1)
        act = x[:, L.ACTIVE_BASE:L.ACTIVE_BASE + 2 * len(L.ACTIVE)].view(n, 2, len(L.ACTIVE))
        types = self.types(act[..., [L.ACTIVE.index('type0'), L.ACTIVE.index('type1')]]).flatten(-2)
        a = self.act_enc(torch.cat([act[..., ACTIVE_BOOSTS].float() / 6, types, act[..., ACTIVE_FLAGS].float(),
                                    self.move(act[..., L.ACTIVE.index('choice_move')])], -1)).flatten(1)
        f = x[:, L.FIELD_BASE:L.FIELD_BASE + len(L.FIELD)]
        field = torch.cat([self.weather(f[:, L.FIELD.index('weather')])]
                          + [f[:, [c]].float() / s for c, s in FIELD_NUMERIC], -1)
        matchup = x[:, L.MATCHUP_BASE:L.MATCHUP_BASE + len(L.MATCHUP)].float() / 4
        h = self.trunk(torch.cat([own_active, foe_active, own_mean, own_max, foe_mean, foe_max, a, field, matchup], -1))
        own_moves = (mv[:, :6] * active[:, :6].unsqueeze(-1)).sum(1)  # [n, 4, 32]
        return h, tok[:, :6], own_moves

    @staticmethod
    def point(head: nn.Sequential, h: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """head(concat(h, item)) for each item, with the first layer split so the
        trunk's half is computed once rather than once per move or switch: the
        same function and parameters, a third of the work."""
        first, out = head[0][0], head[1]
        w = h.shape[-1]
        z = F.linear(h, first.weight[:, :w], first.bias).unsqueeze(1) + F.linear(items, first.weight[:, w:])
        return out(torch.relu(z)).squeeze(-1)

    def forward(self, obs: torch.Tensor, legal: torch.Tensor):
        h, own, moves = self.encode(obs)
        move_logits = self.point(self.move_head, h, moves)
        switch_logits = self.point(self.switch_head, h, own)
        logits = torch.cat([move_logits, switch_logits, self.other(h)], -1).float().masked_fill(~legal, -1e9)
        return torch.distributions.Categorical(logits=logits), self.v(h).float().squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self.encode(obs)[0]).float().squeeze(-1)


class Tempered(nn.Module):
    """A network playing at a temperature: logits / t, so t < 1 sharpens its
    choices and t near 0 takes its top action."""

    def __init__(self, net: nn.Module, t: float) -> None:
        super().__init__()
        self.net, self.t = net, t

    def forward(self, obs: torch.Tensor, legal: torch.Tensor):
        dist, v = self.net(obs, legal)
        return torch.distributions.Categorical(logits=dist.logits / self.t), v

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net.value(obs)


def save(net: nn.Module, path) -> None:
    arrays = {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()}
    if hasattr(net, 'ARCH'):
        arrays['arch'] = np.array([net.ARCH])
    if hasattr(net, 'config'):
        arrays['config'] = np.array(net.config)
    fileio.write_npz(pathlib.Path(path), arrays)


def load(path, device) -> nn.Module:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: download the release checkpoints (README), or train one '
                                '(examples/ppo_smoke.py --save, tools/train_league.py --save)')
    arrays = fileio.read_npz(path)
    arch = int(arrays['arch'][0]) if 'arch' in arrays else 1
    if arch == 3:
        from advsim.net_tf import PolicyTF  # it imports this module's constants
        config = [int(c) for c in arrays['config']]
        if len(config) > 6:
            raise ValueError(f'{path}: saved with PolicyTF options this version no longer has')
        net = PolicyTF(*config)
    elif arch == PolicyV2.ARCH:
        net = PolicyV2(width=arrays['trunk.0.weight'].shape[0], d=arrays['mon_enc.0.weight'].shape[0])
    else:
        layers = sum(1 for k in arrays if k.startswith('body.') and k.endswith('.weight'))
        net = Policy(hidden=arrays['pi.weight'].shape[1], layers=layers, inputs=arrays['body.0.weight'].shape[1])
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items() if k not in ('arch', 'config')})
    return net.to(device).eval()
