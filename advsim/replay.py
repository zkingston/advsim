"""Showdown battles replayed in the engine, judged at every decision point.

The suite (`test_turn_l2.py`, `test_scenarios.py`) and the sweep
(`tools/parity.py`) all judge a battle here, so what counts as a match cannot
drift between them. A case is what `oracle.js` hands back: `before` is
`export_state.js`'s words for the position, and each segment carries the
choices, the raw draws, the words `after`, and the legal masks Showdown's
server would accept.

Draw counts come first, because a wrong count names a missing or extra call
site before any state is worth reading. Then the canonical hash: the engine
hashes its own state, the NumPy mirror hashes the same words with Showdown's
overwriting every exported field, so the two agree exactly when every field
Showdown can speak to does. Only on a mismatch are the fields diffed, to say
what moved.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import warp as wp

from advsim import statehash
from advsim.engine import kernels, layout
from advsim.env import Gen3Env

EXPORTED = tuple(f.name for f in layout.EXPORTED)


def replay(cases: list[dict], device: str = 'cpu', hashes: dict | None = None,
           after=None, fork: bool = False) -> Iterator[tuple[int, int, dict]]:
    """Yield (segment index, case index, problems) for every decision point.

    `problems` maps what disagreed to (engine, showdown): 'draws', 'err',
    'legal', or a state field. Empty means the decision point matched.
    `hashes`, if given, collects the engine's full hash at every point, keyed
    (case, segment): the cross-device check compares two of those.
    `fork` loads the shared `before` once and forks it into every case.
    `after(env, index, todo)`, if given, runs once before the first segment
    (index -1) and after every segment's step, for looking at the engine.
    """
    n = len(cases)
    env = Gen3Env(batch=n + fork, device=device, rng_mode='replay')
    env.reset(seed=0)
    if fork:
        # Perft: every case starts from one position. Load it once, in a spare
        # slot past the cases, and give each case the engine's own copy.
        root = cases[0]['before']
        assert all(c['before'] == root for c in cases), 'fork replay needs one shared position'
        env.load_words([root], at=n)
        env.fork([n] * n, list(range(n)))
    else:
        env.load_words([c['before'] for c in cases])
    streams = [sum((seg['draws'] for seg in c['segments']), []) for c in cases]
    log_host = np.zeros((n + fork, max(max(len(s) for s in streams), 1)), dtype=np.uint32)
    for b, draws in enumerate(streams):
        log_host[b, :len(draws)] = draws
    log = wp.array(log_host, dtype=wp.uint32, device=device)
    # One buffer each, reused: a sweep steps tens of thousands of segments, and
    # allocating a fresh Warp array each time was enough to corrupt the heap.
    acts = wp.zeros((n, 2), dtype=wp.int32, device=device)
    idx = wp.zeros(n, dtype=wp.int32, device=device)
    legal_out = wp.zeros((n, 2), dtype=wp.int32, device=device)
    seen = [0] * n
    if after:
        after(env, -1, list(range(n)))
    for index in range(max(len(c['segments']) for c in cases)):
        todo = [b for b, c in enumerate(cases) if index < len(c['segments'])]
        segs = {b: cases[b]['segments'][index] for b in todo}
        actions = np.zeros((n, 2), dtype=np.int32)
        for b in todo:
            actions[b] = segs[b]['choices']
        acts.assign(actions)
        idx.assign(np.array(todo + [0] * (n - len(todo)), dtype=np.int32))
        wp.launch(kernels.step_idx, dim=len(todo), device=device, inputs=[env.state, env.dex, acts, log, idx])
        wp.launch(kernels.legal_mask, dim=(n, 2), device=device, inputs=[env.state, env.dex, legal_out])
        legal = legal_out.numpy()
        if after:
            after(env, index, todo)

        engine = env.words()
        want = {name: arr.copy() for name, arr in engine.items()}
        for name in EXPORTED:
            want[name][todo] = np.array([segs[b]['after'][name] for b in todo])
        got_hash, want_hash = env.hash(), statehash.state_hash(want)

        for b in todo:
            seg, problems = segs[b], {}
            seen[b] += len(seg['draws'])
            drawn = int(engine['rng_ctr'][b])
            if drawn != seen[b]:
                problems['draws'] = (drawn, seen[b])
                seen[b] = drawn  # count the step, not the drift
            if engine['err'][b]:
                problems['err'] = (int(engine['err'][b]), 0)
            if legal[b].tolist() != seg['legal']:
                problems['legal'] = (legal[b].tolist(), seg['legal'])
            if got_hash[b] != want_hash[b]:
                for name in EXPORTED:
                    if not np.array_equal(engine[name][b], want[name][b]):
                        problems[name] = (engine[name][b].tolist(), want[name][b].tolist())
                if not any(name in problems for name in EXPORTED):
                    # Equal words and unequal hashes: Warp and the mirror disagree.
                    problems['hash'] = (f'{int(got_hash[b]):016x}', f'{int(want_hash[b]):016x}')
            if hashes is not None:
                hashes[(b, index)] = int(got_hash[b])
            yield index, b, problems


def damage(cases: list[dict], device: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    """L1: every case's hit computed by the engine from the words Showdown
    exported for it, next to getDamage's number. -1 marks an immunity."""
    n = len(cases)
    env = Gen3Env(batch=n, device=device, rng_mode='replay')
    host = {name: np.zeros_like(arr) for name, arr in env.words().items()}
    for name in cases[0]['words']:
        host[name][:] = np.array([c['words'][name] for c in cases])
    for name, arr in host.items():
        env.arrays[name].assign(arr)
    col = lambda key: wp.array(np.array([int(c[key]) for c in cases], dtype=np.int32), dtype=wp.int32, device=device)
    out = wp.zeros(n, dtype=wp.int32, device=device)
    wp.launch(kernels.damage_from_state, dim=n, device=device,
              inputs=[env.state, env.dex, col('slot'), col('crit'), col('roll'), out])
    return out.numpy().copy(), np.array([c['damage'] for c in cases], dtype=np.int32)
