"""Acting for any list of (battle, side): the baselines through the engine's
kernel, and a policy network. The Elo bracket and the league trainer both
act through here, on torch tensors on the env's device."""
from __future__ import annotations

import pathlib

import numpy as np
import torch
import warp as wp

from advsim import fileio, net as net_
from advsim.engine import kernels_search, policy_heuristic as ph
from advsim.env import Gen3Env

BASELINES = {'random': ph.RANDOM, 'emerald': ph.EMERALD, 'maxdamage': ph.MAXDAMAGE, 'status': ph.STATUS,
             'switchaverse': ph.SWITCHAVERSE}
# Rows per forward pass in `net_act`: sized for PolicyTF (d=128, 4 layers) beside a full bracket on a
# 12 GB RTX 4070. Memory per row grows with the width and the token count, so raise it on a larger
# card (fewer, fuller launches); PolicyV2 needs a small fraction of the memory per row.
NET_ACT_CHUNK = 8192


def mirror_couples(env: Gen3Env, games: int) -> None:
    """Games 2k and 2k+1 become the same battle, RNG key included, for
    mirrored couples: the caller swaps the players between them."""
    even = np.arange(0, games, 2)
    env.fork(even, even + 1)
    keys = env.arrays['rng_key'].numpy().copy()
    keys[even + 1] = keys[even]
    env.arrays['rng_key'].assign(keys)


def checkpoint(tag: str | None):
    """'league' -> artifacts/ppo_league.npz; none is the first PPO baseline; a path ending in .npz is itself."""
    if tag and tag.endswith('.npz'):
        return pathlib.Path(tag)
    return fileio.ARTIFACTS / f'ppo_{tag or "baseline"}.npz'


class Baselines:
    """play.js's five policies for many (battle, side) at once, in one launch."""

    def __init__(self, env: Gen3Env) -> None:
        self.env = env
        self.out = wp.zeros((env.batch, 2), dtype=wp.int32, device=env.device)

    def act(self, slots: torch.Tensor, sides: torch.Tensor, kinds: torch.Tensor, key: int, tick: int) -> torch.Tensor:
        if not len(slots):
            return torch.zeros(0, dtype=torch.long, device=slots.device)
        i32 = lambda x: wp.from_torch(x.to(torch.int32).contiguous())
        wp.launch(kernels_search.heuristic_actions, dim=len(slots), device=self.env.device, inputs=[
            self.env.state, self.env.dex, i32(slots), i32(sides), i32(kinds), wp.uint32(key % 2 ** 32), tick,
            self.out])
        return wp.to_torch(self.out)[slots.long(), sides.long()].long()


@torch.no_grad()
def net_act(net: net_.Policy, obs: torch.Tensor, mask: torch.Tensor, chunk: int = NET_ACT_CHUNK) -> torch.Tensor:
    """Sample the network's policy over the legal actions: obs [n, OBS_DIM], mask [n] bitmasks.
    In chunks (NET_ACT_CHUNK): a transformer's activations over every live game of the bracket
    overrun a 12 GB card."""
    if not len(obs):
        return torch.empty(0, dtype=torch.long, device=obs.device)
    return torch.cat([net(obs[i:i + chunk], net_.legal_bits(mask[i:i + chunk]))[0].sample()
                      for i in range(0, len(obs), chunk)])
