"""The observation parity check: the engine's `obs` kernel against the live
converter, both players, at every decision point of a protocol case."""
from __future__ import annotations

import collections

import numpy as np
import warp as wp

from advsim.engine import kernels, obs_layout as L
from advsim.live.infostate import InfoState
from advsim.live.observation import Vocab, observe

NAMES = L.names()


def compare(cases: list[dict], replay, device: str = 'cpu') -> tuple[collections.Counter, list]:
    """Counts of differing columns, and the first few differences in full.
    `replay` is tests/replay.py's, which steps the engine through the cases."""
    vocab = Vocab()
    engine = {}
    buf = [None]

    def after(env, index, todo):
        if buf[0] is None:
            buf[0] = wp.zeros((env.batch, 2, L.OBS_DIM), dtype=wp.int16, device=device)
        wp.launch(kernels.obs, dim=(env.batch, 2), device=device, inputs=[env.state, env.dex, buf[0]])
        host = buf[0].numpy()
        for b in todo:
            engine[(b, index)] = host[b].copy()

    for _ in replay(cases, device, after=after):
        pass
    bad, examples = collections.Counter(), []
    for b, case in enumerate(cases):
        states = [InfoState(p, vocab.species_types) for p in ('p1', 'p2')]
        views = [case['start']] + [seg['view'] for seg in case['segments']]
        for index, view in enumerate(views, start=-1):
            for p, st in enumerate(states):
                st.feed(case['log'], view['cursor'])
                st.take_request(view['requests'][p])
                got = observe(st, vocab)
                want = engine[(b, index)][p]
                req = view['requests'][p] or {}
                if any(a.get('maybeTrapped') for a in req.get('active', [])):
                    mask = slice(L.MASK_BASE, L.MASK_BASE + 12)
                    got[mask] = want[mask]  # a hidden trap: see observation.py
                diff = np.flatnonzero(got != want)
                for i in diff:
                    bad[NAMES[i]] += 1
                if len(diff) and len(examples) < 5:
                    examples.append((b, index, p, [(NAMES[i], int(want[i]), int(got[i])) for i in diff[:8]]))
    return bad, examples
