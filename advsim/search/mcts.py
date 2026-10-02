"""Search over a Gen3Env's arena: the host half of engine/tree.py.

Trees search arbitrary arena slots (`roots`), each for its own side. Lanes
occupy `capacity * lanes` slots from `lane_base`, which the caller keeps free.
`decide` takes any number of roots and searches them in waves of up to
`capacity` trees; on CUDA each iteration of a wave is one captured graph
launch (two around a value network's forward pass), the same for every wave.
"""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from advsim.engine import kernels, kernels_search, obs_layout
from advsim.engine.tree import Tree
from advsim.env import Gen3Env, load_setdist

if TYPE_CHECKING:
    import torch


DEFAULT_KNOBS = (0.1, 0.0)  # engine/tree.py: the prior's weight and the pruning ratio (0 off)


class Search:
    def __init__(self, env: Gen3Env, capacity: int, lane_base: int, lanes: int = 8, nodes: int = 1024,
                 depth: int = 6, rollout: int = 16, hidden: bool = True, key: int = 0) -> None:
        L = capacity * lanes
        if lane_base + L > env.batch:
            raise ValueError(f'lanes need slots {lane_base}..{lane_base + L}; the env has {env.batch}')
        self.env, self.capacity, self.lanes, self.depth, self.rollout = env, capacity, lanes, depth, rollout
        self.hidden, self.key, self.waves = int(hidden), key, 0
        dev = env.device
        t = Tree()
        t.child = wp.full((capacity * nodes, 144), -1, dtype=wp.int32, device=dev)
        t.regret = wp.zeros((capacity * nodes, 2, 12), dtype=wp.float32, device=dev)
        t.prior = wp.zeros((capacity * nodes, 2, 12), dtype=wp.float32, device=dev)
        t.cell_n = wp.zeros((capacity, 144), dtype=wp.int32, device=dev)
        t.cell_w = wp.zeros((capacity, 144), dtype=wp.float32, device=dev)
        t.count = wp.zeros(capacity, dtype=wp.int32, device=dev)
        t.knobs = wp.array(np.array(DEFAULT_KNOBS, np.float32), dtype=wp.float32, device=dev)
        t.roots = wp.zeros(capacity, dtype=wp.int32, device=dev)
        t.side = wp.zeros(capacity, dtype=wp.int32, device=dev)
        t.n_trees = wp.zeros(1, dtype=wp.int32, device=dev)
        t.nodes, t.lanes, t.depth, t.lane_base = nodes, lanes, depth, lane_base
        t.lane_node = wp.zeros(L, dtype=wp.int32, device=dev)
        t.lane_depth = wp.zeros(L, dtype=wp.int32, device=dev)
        t.lane_live = wp.zeros(L, dtype=wp.int32, device=dev)
        t.lane_leaf = wp.full(L, -1, dtype=wp.int32, device=dev)
        t.path_node = wp.zeros((L, depth), dtype=wp.int32, device=dev)
        t.path_act = wp.zeros((L, depth, 2), dtype=wp.int32, device=dev)
        t.path_prob = wp.zeros((L, depth, 2), dtype=wp.float32, device=dev)
        t.path_mask = wp.zeros((L, depth, 2), dtype=wp.int32, device=dev)
        t.value = wp.zeros(L, dtype=wp.float32, device=dev)
        t.actions = wp.zeros((env.batch, 2), dtype=wp.int32, device=dev)
        t.counter = wp.zeros(1, dtype=wp.int32, device=dev)
        t.key = wp.zeros(1, dtype=wp.uint32, device=dev)
        self.tree = t
        self.setdist = load_setdist(dev)
        self.log = wp.zeros((env.batch, 1), dtype=wp.uint32, device=dev)
        self.lane_slots = wp.array(np.arange(lane_base, lane_base + L, dtype=np.int32), dtype=wp.int32, device=dev)
        self.q = wp.zeros((capacity, 12, 12), dtype=wp.float32, device=dev)
        self.out = wp.zeros((capacity, 6, 12), dtype=wp.float32, device=dev)
        self.masks = wp.zeros((capacity, 2), dtype=wp.int32, device=dev)
        self.graphs = {}
        self.obs = None
        self.width = capacity

    def _launch(self, kernel, dim, *inputs) -> None:
        wp.launch(kernel, dim=dim, device=self.env.device, inputs=list(inputs))

    def _pre(self, rollout: int, net) -> None:
        """begin -> depth x (select, step) -> rollout x (pick, step) [-> lane observations]."""
        s, dex, t, L = self.env.state, self.env.dex, self.tree, self.width * self.lanes
        self._launch(kernels_search.search_begin, L, s, dex, self.setdist, t, self.hidden)
        for _ in range(self.depth):
            self._launch(kernels_search.search_select, L, s, dex, t)
            self._launch(kernels.step_idx, L, s, dex, t.actions, self.log, self.lane_slots)
        for r in range(rollout):
            self._launch(kernels_search.search_rollout, L, s, dex, t, r)
            self._launch(kernels.step_idx, L, s, dex, t.actions, self.log, self.lane_slots)
        if net is not None:
            self._launch(kernels_search.lane_obs, (L, 2), s, dex, t, self.obs)

    def _forward(self, net, rows: torch.Tensor, prior: bool):
        """Leaf values to p1 for the lanes at `rows` (arena slots), and with
        `prior` both players' policies there, [n, 2, 12]. In chunks: PolicyV2's
        per-Pokemon and per-move intermediates for a full wave (131K rows) do
        not fit next to the arena. An observation is the viewer's, so p1's
        value is the mean of p1's and the negated p2's: (V(p1) - V(p2)) / 2."""
        import torch
        obs = wp.to_torch(self.obs)[rows]  # [lanes, 2 sides, OBS_DIM]
        flat = obs.reshape(-1, obs.shape[-1])
        vs, ps = [], []
        for i in range(0, len(flat), 16384):
            chunk = flat[i:i + 16384]
            if prior:
                legal = chunk[:, obs_layout.MASK_BASE:obs_layout.MASK_BASE + 12] > 0
                dist, v = net(chunk, legal)
                ps.append(dist.probs)
            else:
                v = net.value(chunk)
            vs.append(v)
        v = torch.cat(vs).view(-1, 2)
        value = ((v[:, 0] - v[:, 1]) / 2).clamp(-1, 1)
        return value, (torch.cat(ps).view(-1, 2, 12) if prior else None)

    def _cuda_stream(self):
        import torch
        dev = self.env.device
        if wp.get_device(dev).is_cuda:
            return torch.cuda.stream(wp.stream_to_torch(wp.get_stream(dev)))
        return contextlib.nullcontext()

    def _value(self, net, n: int, prior: bool = False) -> None:
        """Each lane's leaf value, and with `prior` the policies at the node
        its descent made, on Warp's stream so no sync is needed."""
        import torch
        with torch.no_grad(), self._cuda_stream():
            rows = wp.to_torch(self.lane_slots)[:n * self.lanes].long()  # the wave's lanes only
            value, pol = self._forward(net, rows, prior)
            if prior:
                leaf = wp.to_torch(self.tree.lane_leaf)[:n * self.lanes].long()
                made = leaf >= 0
                wp.to_torch(self.tree.prior)[leaf[made]] = pol[made]
            wp.to_torch(self.tree.value)[:n * self.lanes].copy_(value)

    def _root_prior(self, net, n: int) -> None:
        """Each root's prior, from the first lane's world: the searcher's view
        is the same in every world, and the opponent's comes from a sampled
        one, so it never sees what the searcher cannot."""
        import torch
        self._launch(kernels_search.search_begin, self.width * self.lanes, self.env.state, self.env.dex,
                     self.setdist, self.tree, self.hidden)
        self._launch(kernels_search.lane_obs, (self.width * self.lanes, 2), self.env.state, self.env.dex,
                     self.tree, self.obs)
        with torch.no_grad(), self._cuda_stream():
            first = torch.arange(n, device=wp.to_torch(self.lane_slots).device) * self.lanes
            rows = wp.to_torch(self.lane_slots)[first].long()
            _, pol = self._forward(net, rows, True)
            roots = torch.arange(n, device=rows.device) * self.tree.nodes
            wp.to_torch(self.tree.prior)[roots] = pol

    def _post(self, net) -> None:
        self._launch(kernels_search.search_backup, self.width * self.lanes, self.env.state, self.tree,
                     int(net is not None))
        self._launch(kernels_search.search_end, 1, self.tree)

    def iteration(self, rollout: int | None = None, net=None, n: int | None = None, prior: bool = False) -> None:
        rollout = self.rollout if rollout is None else rollout
        self._pre(rollout, net)
        if net is not None:
            self._value(net, self.capacity if n is None else n, prior)
        self._post(net)

    def run(self, roots, sides, iters: int, graph: bool | None = None, rollout: int | None = None,
            net=None, prior: bool = False) -> None:
        """Search one wave of at most `capacity` roots afresh for `iters`
        iterations. With `net`, leaves are scored by its value head after
        `rollout` random decisions (default none); without, by the HP-fraction
        score after `self.rollout`. With `prior` as well, the network's policy
        is each node's prior. On CUDA an iteration is one graph launch, or two
        with the network's forward pass between them."""
        roots, sides = np.asarray(roots, np.int32), np.asarray(sides, np.int32)
        n = len(roots)
        if n > self.capacity:
            raise ValueError(f'{n} roots in one wave; capacity is {self.capacity}')
        rollout = (0 if net is not None else self.rollout) if rollout is None else rollout
        if net is not None and self.obs is None:
            self.obs = wp.zeros((self.env.batch, 2, obs_layout.OBS_DIM), dtype=wp.int16, device=self.env.device)
        pad = np.zeros(self.capacity - n, np.int32)
        self.tree.roots.assign(np.concatenate([roots, pad]))
        self.tree.side.assign(np.concatenate([sides, pad]))
        self.tree.n_trees.assign(np.array([n], np.int32))
        # Launches cover the wave rounded up to a power of two, not the whole
        # capacity: a wave of four trees at the end of a bracket otherwise
        # paid for 8,192. One graph per width.
        self.width = min(self.capacity, max(64, 1 << max(n - 1, 0).bit_length()))
        if graph is None:
            graph = wp.get_device(self.env.device).is_cuda
        mode = (rollout, net is not None, self.width)
        if graph and mode not in self.graphs:
            self.reset()
            self.iteration(rollout, net, n)  # load every kernel before capture
            with wp.ScopedCapture(device=self.env.device) as pre:
                self._pre(rollout, net)
            with wp.ScopedCapture(device=self.env.device) as post:
                self._post(net)
            self.graphs[mode] = (pre.graph, post.graph)
        self.reset()
        # A fresh stream per wave; the first wave's key is `self.key` itself.
        self.tree.key.assign(np.array([(self.key ^ self.waves * 0x9E3779B9) & 0xFFFFFFFF], np.uint32))
        self.waves += 1
        if net is not None and prior:
            self._root_prior(net, n)
        for _ in range(iters):
            if graph:
                pre, post = self.graphs[mode]
                wp.capture_launch(pre)
                if net is not None:
                    self._value(net, n, prior)
                wp.capture_launch(post)
            else:
                self.iteration(rollout, net, n, prior)

    def reset(self) -> None:
        self._launch(kernels_search.tree_reset, self.width, self.tree)

    def policy(self, n: int | None = None, iters: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """([n, 2, 12] each player's root strategy, [n] root value to p1)."""
        n = self.capacity if n is None else n
        self._launch(kernels_search.root_policy, n, self.tree, iters, self.q, self.out, self.masks)
        pol = self.out[:n].numpy()[:, :2].copy()
        cn, cw = self.tree.cell_n[:n].numpy(), self.tree.cell_w[:n].numpy()
        return pol, cw.sum(1) / np.maximum(cn.sum(1), 1)

    def visits(self, n: int | None = None) -> np.ndarray:
        """[n, 144] joint-action visits at each root."""
        return self.tree.cell_n.numpy()[:self.capacity if n is None else n].copy()

    def decide(self, roots, sides, legal, iters: int, rng: np.random.Generator, net=None,
               prior: bool = False, temperature: float = 1.0, final: str = 'eq', knobs: tuple = DEFAULT_KNOBS):
        """One action per root for its side: search in waves, then sample the
        side's root strategy over its true legal actions (a world can disagree
        about them under a hidden trap); uniform over legal where none is left.
        `final` chooses what is played: 'eq' the equilibrium of the root
        matrix, 'mix' its geometric mean with the root prior. `temperature`
        then sharpens it (p ** (1 / t)). `knobs`: the prior's weight and the
        pruning ratio (engine/tree.py)."""
        self.tree.knobs.assign(np.array(knobs, np.float32))  # read at run time: the captured graphs stay valid
        roots, sides, legal = (np.asarray(x, np.int32) for x in (roots, sides, legal))
        out = np.zeros(len(roots), np.int32)
        for lo in range(0, len(roots), self.capacity):
            hi = min(lo + self.capacity, len(roots))
            self.run(roots[lo:hi], sides[lo:hi], iters, net=net, prior=prior)
            pol, _ = self.policy(hi - lo)
            play = restrict(pol[np.arange(hi - lo), sides[lo:hi]], legal[lo:hi])
            if final == 'mix':
                root_nodes = np.arange(hi - lo) * self.tree.nodes
                pri = self.tree.prior.numpy()[root_nodes, sides[lo:hi]]
                play = restrict(np.sqrt(play * restrict(pri, legal[lo:hi])), legal[lo:hi])
            if temperature != 1.0:
                play = play ** (1.0 / temperature)
            out[lo:hi] = sample(play, legal[lo:hi], rng)
        return out


def restrict(weights: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """`weights` [n, 12] over the `legal` bitmasks, normalised; uniform where
    no legal action has weight."""
    allowed = ((legal[:, None] >> np.arange(12)) & 1).astype(bool)
    w = np.where(allowed, np.maximum(weights, 0), 0)
    empty = w.sum(1) <= 0
    w[empty] = allowed[empty]
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def sample(weights: np.ndarray, legal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One action per row from `weights` [n, 12] restricted to the `legal` bitmasks."""
    c = np.cumsum(restrict(weights, legal), 1)
    u = rng.random(len(c)) * c[:, -1]
    return np.minimum((c <= u[:, None]).sum(1), 11).astype(np.int32)
