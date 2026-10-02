"""The search tree on the device: simultaneous-move, open-loop MCTS.

A node is a decision point reached by a sequence of joint actions from the
root, not a state: every iteration re-plays the path from a freshly
determinized copy of the root, so chance and hidden information are sampled
rather than stored (SPEC §Search integration). Per node:

- `child[node, a0 * 12 + a1]`, the node a joint action leads to (-1 none);
- `regret[node, p, a]`: each player's cumulative regret, for decoupled regret
  matching (Lanctot et al., SM-MCTS with RM selection). Values are from p1's
  side; p2 plays -v.

Only a root keeps `cell_n`, `cell_w` (visits and summed value per joint
action), the empirical matrix `matrix_policy` solves, so an inner node is
768 bytes (children, regrets and priors). A tree's root is any arena slot
(`roots[tree]`), searched for `side[tree]`; trees at or past `n_trees[0]` sit
the iteration out, which is how one captured graph serves waves of any size
up to its capacity.

Lanes are the parallel descents of one tree. Selection is a sample from each
player's strategy, so lanes spread out on their own and need no virtual loss.
With a policy network, each node carries both players' policies as a prior
(`prior`): regret matching starts from it and explores with it, instead of
uniformly. The prior of the node a descent creates comes from the same
forward pass that scores its leaf.
Actions are the 12 engine codes: a node's matrix is 12 x 12, most of it unused.
Loops take their bounds from array shapes, not the constant 12, so Warp does
not unroll them: unrolled, matrix_policy alone was 10,000 lines of CUDA.
"""
import warp as wp

from advsim.engine import legal as legal_
from advsim.engine import mask as mask_
from advsim.engine import rng as rng_
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex

A = wp.constant(12)
CELLS = wp.constant(144)
# t.knobs, set by the host per search: the prior's weight in each player's sampling strategy, and with
# PRUNE > 0 an action whose prior is under PRUNE times the node's best is never tried, which spends the
# budget on fewer joint actions.
GAMMA = wp.constant(0)
PRUNE = wp.constant(1)


@wp.struct
class Tree:
    child: wp.array2d(dtype=wp.int32)
    cell_n: wp.array2d(dtype=wp.int32)  # [trees, 144], roots only
    cell_w: wp.array2d(dtype=wp.float32)
    regret: wp.array3d(dtype=wp.float32)
    prior: wp.array3d(dtype=wp.float32)  # [nodes, 2, 12] each player's policy there; all zero means uniform
    count: wp.array(dtype=wp.int32)  # [trees] nodes in use
    knobs: wp.array(dtype=wp.float32)  # [2] GAMMA, PRUNE
    roots: wp.array(dtype=wp.int32)  # [trees] the arena slot each tree searches
    side: wp.array(dtype=wp.int32)  # [trees] the side it searches for
    n_trees: wp.array(dtype=wp.int32)  # [1] trees in this wave
    nodes: int  # per tree
    lanes: int  # per tree
    depth: int  # the longest descent
    lane_base: int  # arena slot of lane 0
    # Per lane: where it is, how deep, whether it still descends, and its path.
    lane_node: wp.array(dtype=wp.int32)
    lane_depth: wp.array(dtype=wp.int32)
    lane_live: wp.array(dtype=wp.int32)  # 0 done, 1 descending, 2 steps once more, then done
    lane_leaf: wp.array(dtype=wp.int32)  # the node this iteration's descent made, or -1: its prior comes from the leaf
    path_node: wp.array2d(dtype=wp.int32)
    path_act: wp.array3d(dtype=wp.int32)
    path_prob: wp.array3d(dtype=wp.float32)
    path_mask: wp.array3d(dtype=wp.int32)
    value: wp.array(dtype=wp.float32)
    actions: wp.array2d(dtype=wp.int32)  # [slots, 2], what the lanes play next; -1 sits out
    counter: wp.array(dtype=wp.int32)  # iterations so far this wave
    key: wp.array(dtype=wp.uint32)  # [1] this wave's key, set by the host: a captured graph reads it at run time


@wp.func
def lane_slot(t: Tree, lane: int) -> int:
    return t.lane_base + lane


@wp.func
def in_wave(t: Tree, lane: int) -> bool:
    return lane / t.lanes < t.n_trees[0]


@wp.func
def prior_of(t: Tree, node: int, p: int, legal: int, a: int) -> float:
    """The node's prior over the legal actions; uniform where it has none."""
    total = float(0.0)
    n = int(0)
    for i in range(t.regret.shape[2]):
        if (legal >> i) & 1 == 1:
            total += t.prior[node, p, i]
            n += 1
    if (legal >> a) & 1 == 0:
        return 0.0
    if total <= 0.0:
        return 1.0 / float(n)
    return t.prior[node, p, a] / total


@wp.func
def strategy(t: Tree, node: int, p: int, legal: int, a: int) -> float:
    """Regret matching: positive regret, normalised; the prior when none is."""
    total = float(0.0)
    for i in range(t.regret.shape[2]):
        if (legal >> i) & 1 == 1:
            total += wp.max(t.regret[node, p, i], 0.0)
    if (legal >> a) & 1 == 0:
        return 0.0
    if total <= 0.0:
        return prior_of(t, node, p, legal, a)
    return wp.max(t.regret[node, p, a], 0.0) / total


@wp.func
def sampling(t: Tree, node: int, p: int, legal: int, a: int) -> float:
    """What selection samples from: the strategy, with some of the prior
    mixed in so every legal action keeps being tried."""
    if (legal >> a) & 1 == 0:
        return 0.0
    g = t.knobs[GAMMA]
    return (1.0 - g) * strategy(t, node, p, legal, a) + g * prior_of(t, node, p, legal, a)


@wp.func
def prune(t: Tree, node: int, p: int, legal: int) -> int:
    """`legal` without the actions whose prior is under PRUNE times the best;
    the best always stays. Unchanged with PRUNE 0 or no prior at the node."""
    rho = t.knobs[PRUNE]
    if rho <= 0.0:
        return legal
    best = float(0.0)
    for a in range(t.regret.shape[2]):
        if (legal >> a) & 1 == 1:
            best = wp.max(best, t.prior[node, p, a])
    if best <= 0.0:
        return legal
    kept = int(0)
    for a in range(t.regret.shape[2]):
        if (legal >> a) & 1 == 1 and t.prior[node, p, a] >= rho * best:
            kept = kept | (1 << a)
    return kept


@wp.func
def unit(raw: wp.uint32) -> float:
    return float(raw >> wp.uint32(8)) / 16777216.0


@wp.func
def sample(t: Tree, node: int, p: int, legal: int, raw: wp.uint32) -> int:
    """An action drawn from `sampling`. When rounding leaves u past the sum,
    the last action with any probability: the backup divides by it."""
    u = unit(raw)
    last = int(-1)
    for a in range(t.regret.shape[2]):
        if (legal >> a) & 1 == 1:
            pa = sampling(t, node, p, legal, a)
            if pa > 0.0 or last < 0:
                last = a
            u -= pa
            if u < 0.0:
                return a
    return last


@wp.func
def random_legal(legal: int, raw: wp.uint32) -> int:
    n = int(0)
    for a in range(A):
        n += (legal >> a) & 1
    k = rng_.random_n(raw, n)
    for a in range(A):
        if (legal >> a) & 1 == 1:
            if k == 0:
                return a
            k -= 1
    return mask_.ACTION_PASS


@wp.func
def search_draw(t: Tree, lane: int, what: int) -> wp.uint32:
    """The search's own stream, apart from every battle's."""
    it = wp.uint32(t.counter[0])
    return rng_.rng_u32(rng_.mix32(t.key[0] ^ it * rng_.KNUTH) ^ wp.uint32(lane), wp.uint32(what))


@wp.func
def reset_node(t: Tree, node: int):
    for c in range(CELLS):
        t.child[node, c] = -1
    for p in range(2):
        for a in range(t.regret.shape[2]):
            t.regret[node, p, a] = 0.0
            t.prior[node, p, a] = 0.0


@wp.func
def select(s: State, dex: Dex, t: Tree, lane: int):
    """One level of one lane's descent: sample a joint action at its node,
    note it on the path and move to the child, making it if it is new. A new
    node, a full tree, the depth limit or a finished battle end the descent."""
    b = lane_slot(t, lane)
    t.actions[b, 0] = -1  # step_idx skips the lane unless it moves on
    if t.lane_live[lane] != 1:
        t.lane_live[lane] = 0
        return
    d = t.lane_depth[lane]
    if int(s.result[b]) != mask_.RESULT_ONGOING or d >= t.depth:
        t.lane_live[lane] = 0
        return
    tree = lane / t.lanes
    node = t.lane_node[lane]
    m0 = prune(t, node, 0, legal_.legal_actions(s, dex, b, 0))
    m1 = prune(t, node, 1, legal_.legal_actions(s, dex, b, 1))
    a0 = sample(t, node, 0, m0, search_draw(t, lane, 2 * d))
    a1 = sample(t, node, 1, m1, search_draw(t, lane, 2 * d + 1))
    t.path_node[lane, d] = node
    t.path_act[lane, d, 0] = a0
    t.path_act[lane, d, 1] = a1
    t.path_prob[lane, d, 0] = sampling(t, node, 0, m0, a0)
    t.path_prob[lane, d, 1] = sampling(t, node, 1, m1, a1)
    t.path_mask[lane, d, 0] = m0
    t.path_mask[lane, d, 1] = m1
    t.actions[b, 0] = a0
    t.actions[b, 1] = a1
    t.lane_depth[lane] = d + 1

    cell = a0 * A + a1
    child = t.child[node, cell]
    stop = False
    if child < 0:
        stop = True
        local = wp.atomic_add(t.count, tree, 1)
        if local < t.nodes:
            fresh = tree * t.nodes + local
            reset_node(t, fresh)
            won = wp.atomic_cas(t.child, node, cell, -1, fresh)
            child = wp.where(won == -1, fresh, won)  # another lane made it first: use theirs
        else:
            child = -1
    t.lane_node[lane] = child
    t.lane_live[lane] = wp.where(stop or child < 0, 2, 1)
    if stop and child >= 0:
        t.lane_leaf[lane] = child


@wp.func
def leaf_value(s: State, b: int) -> float:
    """p1's value: the result if the battle is over, else the difference in
    remaining HP fractions summed over each party, over six."""
    result = int(s.result[b])
    if result == mask_.RESULT_P1:
        return 1.0
    if result == mask_.RESULT_P2:
        return -1.0
    if result == mask_.RESULT_TIE:
        return 0.0
    v = float(0.0)
    for i in range(6):
        v += float(s.hp[b, 0, i]) / float(wp.max(int(s.maxhp[b, 0, i]), 1))
        v -= float(s.hp[b, 1, i]) / float(wp.max(int(s.maxhp[b, 1, i]), 1))
    return v / 6.0


@wp.func
def backup(s: State, t: Tree, lane: int, use_value: int):
    """Every node on the lane's path: the cell's visits and value, and each
    player's regrets from the importance-weighted estimate of the one action
    it sampled (r(a) = x(a) - u, x(a) = u / p(a) for the sampled a, else 0).

    The leaf's value is the result of a finished battle; otherwise, with
    `use_value`, what the host wrote to t.value (a value network), else the
    HP-fraction score."""
    if not in_wave(t, lane):
        return
    b = lane_slot(t, lane)
    u = leaf_value(s, b)
    if use_value == 1 and int(s.result[b]) == mask_.RESULT_ONGOING:
        u = t.value[lane]
    t.value[lane] = u
    root = (lane / t.lanes) * t.nodes
    for d in range(t.lane_depth[lane]):
        node = t.path_node[lane, d]
        a0 = t.path_act[lane, d, 0]
        a1 = t.path_act[lane, d, 1]
        if node == root:
            wp.atomic_add(t.cell_n, lane / t.lanes, a0 * A + a1, 1)
            wp.atomic_add(t.cell_w, lane / t.lanes, a0 * A + a1, u)
        for p in range(2):
            up = wp.where(p == 0, u, -u)
            mine = t.path_act[lane, d, p]
            legal = t.path_mask[lane, d, p]
            for a in range(t.regret.shape[2]):
                if (legal >> a) & 1 == 1:
                    x = float(0.0)
                    if a == mine:
                        x = up / t.path_prob[lane, d, p]
                    wp.atomic_add(t.regret, node, p, a, x - up)


@wp.func
def matrix_policy(q: wp.array3d(dtype=wp.float32), m0: int, m1: int, i: int, iters: int,
                  out: wp.array3d(dtype=wp.float32)):
    """Regret matching (RM+) in self-play on one 12 x 12 zero-sum matrix, rows
    p1's actions in m0, columns p2's in m1, entries p1's value. out[i, 0..1]
    gets each player's average strategy, which converges to a Nash equilibrium;
    rows 2..3 hold the regrets and 4..5 the current strategies while it runs."""
    for r in range(out.shape[1]):
        for a in range(q.shape[1]):
            out[i, r, a] = 0.0
    for _ in range(iters):
        for p in range(2):
            legal = wp.where(p == 0, m0, m1)
            total = float(0.0)
            n = int(0)
            for a in range(q.shape[1]):
                if (legal >> a) & 1 == 1:
                    total += out[i, 2 + p, a]
                    n += 1
            for a in range(q.shape[1]):
                x = float(0.0)
                if (legal >> a) & 1 == 1:
                    x = wp.where(total > 0.0, out[i, 2 + p, a] / wp.max(total, 1e-30), 1.0 / float(n))
                out[i, 4 + p, a] = x
                out[i, p, a] += x
        v = float(0.0)
        for a in range(q.shape[1]):
            for c in range(q.shape[1]):
                v += out[i, 4, a] * out[i, 5, c] * q[i, a, c]
        for a in range(q.shape[1]):
            row = float(0.0)
            col = float(0.0)
            for c in range(q.shape[1]):
                row += out[i, 5, c] * q[i, a, c]
                col += out[i, 4, c] * q[i, c, a]
            if (m0 >> a) & 1 == 1:
                out[i, 2, a] = wp.max(out[i, 2, a] + row - v, 0.0)
            if (m1 >> a) & 1 == 1:
                out[i, 3, a] = wp.max(out[i, 3, a] - col + v, 0.0)
    for p in range(2):
        for a in range(q.shape[1]):
            out[i, p, a] = out[i, p, a] / float(wp.max(iters, 1))
