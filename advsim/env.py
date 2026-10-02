"""Host side: allocation, uploads and launches. No battle logic lives here.

`engine/` is pure Warp and never touches NumPy or files, so everything that
reads an artifact or builds an array is here.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from advsim import artifacts, fileio
from advsim.engine import kernels, kernels_search, layout, obs_layout
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex
from advsim.engine.setdist import SetDist

NP_DTYPE = {'u8': np.uint8, 'u16': np.uint16, 'u32': np.uint32}
WP_DTYPE = {'u8': wp.uint8, 'u16': wp.uint16, 'u32': wp.uint32}
RNG_MODES = {'train': 0, 'paired': 1, 'replay': 2}


def load_dex(device) -> Dex:
    """dex.npz into one struct of int32 arrays; the tables are a few tens of KB."""
    arrays = artifacts.load_dex()
    dex = Dex()
    for name in Dex.vars:
        setattr(dex, name, wp.array(arrays[name].astype(np.int32), dtype=wp.int32, device=device))
    return dex


_POOLS: dict = {}


def load_pool(path, device) -> wp.array:
    """The team pool on a device, shared by every env there: it is 240 MB, and
    a process that made an env per test ran the card out of memory."""
    key = (str(path or fileio.ARTIFACTS / 'pool.npz'), str(device))
    if key not in _POOLS:
        teams = fileio.read_npz(key[0])['teams']
        want = artifacts.read_manifest().get('pool', {}).get('sha256')
        if want and want != fileio.hash_arrays({'teams': teams}):
            raise ValueError('pool.npz does not match the manifest; rebuild')
        _POOLS[key] = wp.array(teams.astype(np.int16), dtype=wp.int16, device=device)
    return _POOLS[key]


def load_setdist(device) -> SetDist:
    arrays = fileio.read_npz(fileio.ARTIFACTS / 'setdist.npz')
    sd = SetDist()
    sd.sets = wp.array(arrays['sets'], dtype=wp.int16, device=device)
    for name in ('weight', 'start', 'count', 'species_weight'):
        setattr(sd, name, wp.array(arrays[name].astype(np.int32), dtype=wp.int32, device=device))
    sd.gender = wp.array(arrays['gender'].astype(np.int32), dtype=wp.int32, device=device)
    return sd


class Gen3Env:
    """A batch of independent battles living on one device."""

    def __init__(self, batch: int, device: str = 'cuda:0', rng_mode: str = 'train',
                 pool_path=None, max_steps: int = 0) -> None:
        if rng_mode not in RNG_MODES:
            raise ValueError(f'rng_mode must be one of {sorted(RNG_MODES)}')
        self.batch = batch
        self.device = device
        self.rng_mode = rng_mode
        self.dex = load_dex(device)
        self.pool = load_pool(pool_path, device)
        self.pool_size = self.pool.shape[0]
        self.cursor = wp.zeros(1, dtype=wp.int32, device=device)
        self.state = self._allocate()
        # The training loop's buffers: fixed, so a captured graph can reuse them.
        self.max_steps = max_steps
        self.seed = 0
        self.actions = wp.zeros((batch, 2), dtype=wp.int32, device=device)
        self.log = wp.zeros((batch, 1), dtype=wp.uint32, device=device)  # replay only
        self.steps = wp.zeros(batch, dtype=wp.int32, device=device)
        self.reward_buf = wp.zeros(batch, dtype=wp.float32, device=device)
        self.done_buf = wp.zeros(batch, dtype=wp.uint8, device=device)
        self.truncated_buf = wp.zeros(batch, dtype=wp.uint8, device=device)
        self.obs_buf = wp.zeros((batch, 2, obs_layout.OBS_DIM), dtype=wp.int16, device=device)
        self.mask_buf = wp.zeros((batch, 2), dtype=wp.int32, device=device)
        self.every = wp.array(np.arange(batch, dtype=np.int32), dtype=wp.int32, device=device)
        self.graph = None

    def _allocate(self) -> State:
        state = State()
        self.arrays = {}
        for f in layout.FIELDS:
            arr = wp.zeros(f.shape(self.batch), dtype=WP_DTYPE[f.dtype], device=self.device)
            self.arrays[f.name] = arr
            setattr(state, f.name, arr)
        return state

    def reset(self, seed: int = 0) -> None:
        """Deal every battle two fresh teams from the pool, from a row the seed
        picks; every field starts from zero."""
        self.seed = seed
        self.steps.zero_()
        start = (seed * 2 * self.batch) % self.pool_size
        # Auto-reset deals from here on, after the rows this reset took.
        self.cursor.fill_((start + 2 * self.batch) % self.pool_size)
        wp.launch(kernels.reset, dim=self.batch, device=self.device,
                  inputs=[self.state, self.dex, self.pool, self.pool_size, start,
                          wp.uint32(seed), RNG_MODES[self.rng_mode]])

    # ---- the training API (SPEC §Kernels and runtime API)

    def observe(self):
        """(obs, mask): int16[B, 2, OBS_DIM] and int32[B, 2] bitmasks, as
        zero-copy views: PyTorch tensors when torch is installed."""
        self._launch_observe()
        return _view(self.obs_buf), _view(self.mask_buf)

    @property
    def needs_decision(self):
        """bool[B, 2]: False where a side can only pass; pass it there."""
        wp.launch(kernels.legal_mask, dim=(self.batch, 2), device=self.device,
                  inputs=[self.state, self.dex, self.mask_buf])
        return _view(self.mask_buf) != (1 << 10)

    def step(self, actions) -> None:
        """Advance every battle one decision. `actions` is [B, 2] action codes;
        finished or truncated battles are dealt again in the same launch."""
        _write(self.actions, actions)
        self._launch_step()

    @property
    def reward(self):
        return _view(self.reward_buf)

    @property
    def done(self):
        return _view(self.done_buf)

    @property
    def truncated(self):
        return _view(self.truncated_buf)

    @property
    def err(self):
        return _view(self.arrays['err'])

    def capture(self) -> None:
        """Record step -> obs -> mask as one CUDA graph; `step_graph` replays it.
        Write actions into `self.actions` (or its view) before each replay."""
        self._launch_step()
        self._launch_observe()  # warm up: kernels load before capture
        with wp.ScopedCapture(device=self.device) as cap:
            self._launch_step()
            self._launch_observe()
        self.graph = cap.graph

    def step_graph(self) -> None:
        wp.capture_launch(self.graph)

    def _launch_step(self) -> None:
        wp.launch(kernels.step_idx, dim=self.batch, device=self.device,
                  inputs=[self.state, self.dex, self.actions, self.log, self.every])
        wp.launch(kernels.settle, dim=self.batch, device=self.device,
                  inputs=[self.state, self.dex, self.pool, self.pool_size, self.cursor, wp.uint32(self.seed),
                          RNG_MODES[self.rng_mode], self.max_steps, self.steps, self.reward_buf, self.done_buf,
                          self.truncated_buf])

    def _launch_observe(self) -> None:
        wp.launch(kernels.obs, dim=(self.batch, 2), device=self.device,
                  inputs=[self.state, self.dex, self.obs_buf])
        wp.launch(kernels.legal_mask, dim=(self.batch, 2), device=self.device,
                  inputs=[self.state, self.dex, self.mask_buf])

    def hash(self, position: bool = False) -> np.ndarray:
        """uint64[B]: the canonical hash, full by default. Mirrored in statehash.py."""
        if not hasattr(self, '_hash_out'):
            self._hash_out = wp.zeros(self.batch, dtype=wp.uint64, device=self.device)
        wp.launch(kernels.state_hash, dim=self.batch, device=self.device,
                  inputs=[self.state, int(position), self._hash_out])
        # A copy: on the CPU device `numpy()` is a view of the reused buffer, so
        # two calls would otherwise hand back the same array twice.
        return self._hash_out.numpy().copy()

    def words(self) -> dict[str, np.ndarray]:
        """Every state field as a host copy, the shape `statehash` hashes.

        A copy: on the CPU device `numpy()` is a view of the live array, and a
        caller that overwrites a field to compare against would be writing into
        the engine and then finding that the two agree.
        """
        return {f.name: self.arrays[f.name].numpy().copy() for f in layout.FIELDS}

    def fork(self, src, dst, salt: int = 0) -> None:
        """Slot dst[i] becomes a copy of slot src[i] with a re-derived RNG key.
        Sources and destinations must not overlap."""
        src = _index(src, self.device)
        dst = _index(dst, self.device)
        wp.launch(kernels_search.fork, dim=len(dst), device=self.device,
                  inputs=[self.state, src, dst, wp.uint32(salt)])

    def determinize(self, src, dst, side: int, key: int) -> None:
        """World slot dst[i]: battle src[i] with everything hidden from `side`
        drawn again with `key` (SPEC §Determinization). Sources and
        destinations must not overlap."""
        if not hasattr(self, 'setdist'):
            self.setdist = load_setdist(self.device)
        dst = _index(dst, self.device)
        wp.launch(kernels_search.determinize, dim=len(dst), device=self.device,
                  inputs=[self.state, self.dex, self.setdist, _index(src, self.device), dst, side,
                          wp.uint32(key)])

    def load_words(self, rows: list[dict], at: int = 0) -> None:
        """Battle at + i takes the state `export_state.js` gave for rows[i],
        ready to replay: every exported field is Showdown's, and the engine's
        own bookkeeping starts fresh with the log cursor at the first draw."""
        host = self.words()
        at = slice(at, at + len(rows))
        for f in layout.FIELDS:
            if f.exported:
                host[f.name][at] = np.array([row[f.name] for row in rows])
            else:
                host[f.name][at] = 0
        host['rng_mode'][at] = RNG_MODES['replay']
        for name, arr in host.items():
            self.arrays[name].assign(arr)

    def numpy(self, name: str) -> np.ndarray:
        """One state field as a host array; for tests and the NumPy mirror."""
        return self.arrays[name].numpy()

    def bytes_allocated(self) -> int:
        return sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in self.arrays.values())


def _view(arr: wp.array):
    """A zero-copy PyTorch view when torch is installed, else a NumPy one."""
    try:
        return wp.to_torch(arr)
    except ImportError:
        return arr.numpy()


def _index(x, device) -> wp.array:
    """An int32 slot list on the device, from a Warp array, a tensor or a list."""
    if isinstance(x, wp.array):
        return x
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return wp.array(np.asarray(x, dtype=np.int32), dtype=wp.int32, device=device)


def _write(dst: wp.array, src) -> None:
    """Copy actions in from a torch tensor, a NumPy array or a list."""
    try:
        import torch
        if isinstance(src, torch.Tensor):
            wp.to_torch(dst).copy_(src.to(torch.int32))
            return
    except ImportError:
        pass
    dst.assign(np.asarray(src, dtype=np.int32))
