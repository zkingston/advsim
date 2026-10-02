"""Every kernel that does not run the battle step: fork, determinize, the
device tree, and the baseline policies the Elo bracket plays.

A Warp module of its own so that editing any of them does not rebuild the
step's module (see kernels.py). Nothing here may call the battle step; lanes
and games are stepped by `kernels.step_idx`.
"""
import warp as wp

wp.set_module_options({"enable_backward": False})

from advsim.engine import determinize as det_  # noqa: E402
from advsim.engine import legal as legal_  # noqa: E402
from advsim.engine import mask as mask_  # noqa: E402
from advsim.engine import obs as obs_  # noqa: E402
from advsim.engine import policy_heuristic  # noqa: E402
from advsim.engine import rng as rng_  # noqa: E402
from advsim.engine import setdist as setdist_  # noqa: E402
from advsim.engine import tree as tree_  # noqa: E402
from advsim.engine._generated.state import State, copy_battle  # noqa: E402
from advsim.engine.dex import Dex  # noqa: E402


@wp.kernel
def fork(s: State, src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32), salt: wp.uint32):
    """Slot dst[i] becomes a copy of slot src[i], all but the RNG key, which
    each child re-derives from its parent's, its slot and `salt`. The draw
    counter carries over, so a replay-mode child reads its own log row from
    where the parent stood. No slot may be both a source and a destination."""
    i = wp.tid()
    d = dst[i]
    copy_battle(s, src[i], d)
    s.rng_key[d] = rng_.mix32(rng_.mix32(s.rng_key[d] ^ salt) ^ wp.uint32(d) * rng_.KNUTH)


@wp.kernel
def determinize(s: State, dex: Dex, sd: setdist_.SetDist, src: wp.array(dtype=wp.int32),
                dst: wp.array(dtype=wp.int32), side: int, key: wp.uint32):
    """World slot dst[i]: battle src[i] as `side` sees it, the rest drawn."""
    i = wp.tid()
    det_.determinize(s, dex, sd, src[i], dst[i], side, key)


# One search iteration is begin, depth x (select, kernels.step_idx), rollout
# x (search_rollout, kernels.step_idx), backup, end; fixed launches over fixed buffers, so it captures as one
# CUDA graph. The iteration count lives on the device for the same reason.

@wp.kernel
def tree_reset(t: tree_.Tree):
    """Every tree back to a bare root; other nodes are cleared as they are made."""
    tree = wp.tid()
    t.count[tree] = 1
    tree_.reset_node(t, tree * t.nodes)
    for c in range(tree_.CELLS):
        t.cell_n[tree, c] = 0
        t.cell_w[tree, c] = 0.0
    if tree == 0:
        t.counter[0] = 0


@wp.kernel
def search_begin(s: State, dex: Dex, sd: setdist_.SetDist, t: tree_.Tree, hidden: int):
    """Each lane starts from its tree's root: a fresh world as the tree's side
    sees it, or, with `hidden` 0, an exact copy with a fresh key."""
    lane = wp.tid()
    t.lane_depth[lane] = 0
    t.lane_live[lane] = 0
    t.lane_leaf[lane] = -1
    if not tree_.in_wave(t, lane):
        return
    tree = lane / t.lanes
    root = t.roots[tree]
    b = tree_.lane_slot(t, lane)
    k = rng_.mix32(t.key[0] ^ wp.uint32(t.counter[0]) * rng_.KNUTH)
    if hidden == 1:
        det_.determinize(s, dex, sd, root, b, t.side[tree], k)
    else:
        copy_battle(s, root, b)
        s.rng_key[b] = rng_.mix32(k ^ wp.uint32(b) * rng_.KNUTH)
        s.rng_mode[b] = wp.uint8(0)
    t.lane_node[lane] = tree * t.nodes
    t.lane_live[lane] = 1


@wp.kernel
def search_select(s: State, dex: Dex, t: tree_.Tree):
    tree_.select(s, dex, t, wp.tid())


@wp.kernel
def search_rollout(s: State, dex: Dex, t: tree_.Tree, r: int):
    """Random play for one decision past the tree: both sides' actions for
    `step_idx`, or -1 where the battle is over."""
    lane = wp.tid()
    b = tree_.lane_slot(t, lane)
    t.actions[b, 0] = -1
    if not tree_.in_wave(t, lane) or int(s.result[b]) != mask_.RESULT_ONGOING:
        return
    t.actions[b, 0] = tree_.random_legal(legal_.legal_actions(s, dex, b, 0), tree_.search_draw(t, lane, 1000 + 2 * r))
    t.actions[b, 1] = tree_.random_legal(legal_.legal_actions(s, dex, b, 1), tree_.search_draw(t, lane, 1001 + 2 * r))


@wp.kernel
def search_backup(s: State, t: tree_.Tree, use_value: int):
    tree_.backup(s, t, wp.tid(), use_value)


@wp.kernel
def lane_obs(s: State, dex: Dex, t: tree_.Tree, out: wp.array3d(dtype=wp.int16)):
    """Both players' observations of every lane in the wave, at the lane's
    arena row of `out`, for a value network to score."""
    lane, p = wp.tid()
    if not tree_.in_wave(t, lane):
        return
    b = tree_.lane_slot(t, lane)
    for i in range(out.shape[2]):
        out[b, p, i] = wp.int16(0)
    obs_.observe(s, dex, out, b, p)


@wp.kernel
def search_end(t: tree_.Tree):
    t.counter[0] = t.counter[0] + 1


@wp.kernel
def root_policy(t: tree_.Tree, iters: int, q: wp.array3d(dtype=wp.float32), out: wp.array3d(dtype=wp.float32),
                masks: wp.array2d(dtype=wp.int32)):
    """Each root's empirical matrix, solved by regret matching. Actions are the
    ones visited at the root; an unvisited cell between two of them takes the
    root's mean value."""
    tree = wp.tid()
    node = tree
    n = int(0)
    w = float(0.0)
    m0 = int(0)
    m1 = int(0)
    for c in range(tree_.CELLS):
        if t.cell_n[node, c] > 0:
            n += t.cell_n[node, c]
            w += t.cell_w[node, c]
            m0 = m0 | (1 << (c / tree_.A))
            m1 = m1 | (1 << (c % tree_.A))
    mean = w / float(wp.max(n, 1))
    for a in range(q.shape[1]):
        for c in range(q.shape[1]):
            k = t.cell_n[node, a * tree_.A + c]
            q[tree, a, c] = wp.where(k > 0, t.cell_w[node, a * tree_.A + c] / float(wp.max(k, 1)), mean)
    masks[tree, 0] = m0
    masks[tree, 1] = m1
    tree_.matrix_policy(q, m0, m1, tree, iters, out)


@wp.kernel
def solve_matrix(q: wp.array3d(dtype=wp.float32), masks: wp.array2d(dtype=wp.int32), iters: int,
                 out: wp.array3d(dtype=wp.float32)):
    """matrix_policy on given matrices, for testing it on known games."""
    i = wp.tid()
    tree_.matrix_policy(q, masks[i, 0], masks[i, 1], i, iters, out)


@wp.kernel
def heuristic_actions(s: State, dex: Dex, slot: wp.array(dtype=wp.int32), side: wp.array(dtype=wp.int32),
                      policy: wp.array(dtype=wp.int32), key: wp.uint32, tick: int, out: wp.array2d(dtype=wp.int32)):
    """out[slot[i], side[i]]: baseline policy[i]'s action there. `tick` keeps
    one decision's random choices apart from the next."""
    i = wp.tid()
    out[slot[i], side[i]] = policy_heuristic.act(s, dex, slot[i], side[i], policy[i], key, tick)
