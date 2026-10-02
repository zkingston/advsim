"""Random draws: one counter hash, and Showdown's mappings on top of it.

Showdown's own generator is ChaCha20-based and is not reimplemented here.
Replay mode reads its logged `rng.next()` outputs instead, so parity never
depends on matching a stream cipher. Train and paired modes use a counter hash,
which gives a battle one u32 of state and makes a draw a pure function of its
coordinates.

Every consumer maps a raw u32 exactly as `sim/prng.js` does, so switching modes
changes no other code.
"""
import warp as wp
from advsim.engine import mask as mask_
from advsim.engine._generated.state import State

MODE_PAIRED = wp.constant(1)
MODE_REPLAY = wp.constant(2)

GOLDEN = wp.constant(wp.uint32(0x9E3779B9))
# Named so the literals are converted once here rather than at each use, where
# Warp would parse them as signed and warn on the wrap.
MIX_A = wp.constant(wp.uint32(0x7FEB352D))
MIX_B = wp.constant(wp.uint32(0x846CA68B))

KNUTH = wp.constant(wp.uint32(2654435761))


@wp.func
def mix32(x: wp.uint32) -> wp.uint32:
    """Two multiply-xorshift rounds; avalanches every input bit."""
    x ^= x >> wp.uint32(16)
    x *= MIX_A
    x ^= x >> wp.uint32(15)
    x *= MIX_B
    x ^= x >> wp.uint32(16)
    return x


@wp.func
def rng_u32(key: wp.uint32, ctr: wp.uint32) -> wp.uint32:
    """Train mode: the draw is a pure function of (key, counter)."""
    return mix32(mix32(key ^ GOLDEN) ^ mix32(ctr))


@wp.func
def paired_u32(key: wp.uint32, turn: wp.uint32, index: wp.uint32) -> wp.uint32:
    """Paired mode: both games of a mirrored pair draw alike per turn, even after they diverge."""
    return mix32(mix32(key ^ GOLDEN) ^ mix32(turn * KNUTH ^ index))


# ---- Showdown's mappings (sim/prng.js), applied to a raw u32.
@wp.func
def random_n(raw: wp.uint32, n: int) -> int:
    """`random(n)`: (u64(x) * n) >> 32."""
    return int((wp.uint64(raw) * wp.uint64(n)) >> wp.uint64(32))


@wp.func
def random_range(raw: wp.uint32, m: int, n: int) -> int:
    """`random(m, n)`: m + ((u64(x) * (n - m)) >> 32)."""
    return m + int((wp.uint64(raw) * wp.uint64(n - m)) >> wp.uint64(32))


@wp.func
def random_chance(raw: wp.uint32, numerator: int, denominator: int) -> bool:
    """`randomChance(num, den)`: random(den) < num."""
    return random_n(raw, denominator) < numerator


@wp.func
def draw(s: State, log: wp.array2d(dtype=wp.uint32), b: int) -> wp.uint32:
    """The next raw u32 for this battle, from the counter hash or the replay log."""
    ctr = s.rng_ctr[b]
    s.rng_ctr[b] = ctr + wp.uint32(1)
    if int(s.rng_mode[b]) == MODE_REPLAY:
        if int(ctr) >= log.shape[1]:
            s.err[b] = wp.uint8(int(s.err[b]) | mask_.ERR_LOG_EXHAUSTED)
            return wp.uint32(0)
        return log[b, int(ctr)]
    if int(s.rng_mode[b]) == MODE_PAIRED:
        return paired_u32(s.rng_key[b], wp.uint32(s.turn[b]), ctr)
    return rng_u32(s.rng_key[b], ctr)
