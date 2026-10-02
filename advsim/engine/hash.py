"""The canonical hash primitive.

`h = XOR over i in mask of splitmix64((i << 32) | w_i)`, where `w_i` is the i-th
u32 word of the canonical layout. Index-salted so the fold is order-free, and
XOR-folded so a single changed word updates a hash in place, Zobrist-style:
`h ^= word(i, old) ^ word(i, new)`.

The per-field walks are generated from `layout.py` into `_generated/hashes.py`,
because a hand-written field list would drift from the state layout. The NumPy
mirror is `advsim/statehash.py`.
"""
import warp as wp


@wp.func
def splitmix64(x: wp.uint64) -> wp.uint64:
    z = x + wp.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    return z ^ (z >> wp.uint64(31))


@wp.func
def word(i: int, w: wp.uint32) -> wp.uint64:
    """One word's contribution. The salt is the word index, not the field."""
    return splitmix64((wp.uint64(i) << wp.uint64(32)) | wp.uint64(w))
