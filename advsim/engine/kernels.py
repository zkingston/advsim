"""The engine's kernels: the battle step, dealing, observations, hashing and the
test hooks.

Search's kernels are in `kernels_search.py`: Warp compiles each module on its
own, and every kernel that calls the step gets its own inlined copy of it, so
`step_idx` is the only one here that does and the CUDA build stays at a few
minutes. None of search's kernels calls the step, so editing search rebuilds
only its own small module. Other engine modules export `@wp.func`s only."""
import warp as wp

# No gradients anywhere in this engine, and asking for them is not free:
# Warp emits an adjoint for every function it compiles, and those adjoints
# were two thirds of the generated CUDA.
wp.set_module_options({"enable_backward": False})

from advsim.engine import mask as mask_
from advsim.engine import obs as obs_
from advsim.engine import moves as mv
from advsim.engine import policy_emerald
from advsim.engine import rng as rng_
from advsim.engine import deal as deal_
from advsim.engine import execute as execute_
from advsim.engine import faint as faint_
from advsim.engine import legal as legal_
from advsim.engine import switch as switch_
from advsim.engine import turn as turn_
from advsim.engine._generated import hashes, ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex


@wp.kernel
def damage_from_state(s: State, dex: Dex, slot: wp.array(dtype=wp.int32), crit: wp.array(dtype=wp.int32),
                      roll: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    """L1: side 0's move `slot` hitting side 1, through `one_hit`, the path a
    move takes, with the crit and the roll given rather than drawn. -1 is an
    immunity, which Showdown's getDamage refuses before any damage."""
    b = wp.tid()
    att = int(s.active[b, 0])
    dfn = int(s.active[b, 1])
    move = int(s.moves[b, 0, att, slot[b]])
    move_type = mv.type_of(dex, move, int(s.hp_type[b, 0, att]))
    category = dex.move_category[move]
    if dex.move_type_from_mon[move] != 0:
        category = dex.type_is_special[move_type]
    if mv.type_multiplier_of(dex, move_type, int(s.types[b, 1, 0]), int(s.types[b, 1, 1])) == -99 or \
            (move_type == ids.TYPE_GROUND and not switch_.grounded(s, dex, b, 1, dfn)):
        out[b] = -1
        return
    # A raw draw of 0 always passes a crit roll and the top one never does.
    crit_raw = wp.uint32(0xFFFFFFFF)
    if crit[b] == 1:
        crit_raw = wp.uint32(0)
    out[b] = execute_.one_hit(s, dex, b, 0, move, move_type, category, False, crit_raw,
                              wp.uint32(roll[b]) * wp.uint32(0x10000000))


@wp.kernel
def prng_mappings(raw: wp.array(dtype=wp.uint32), n: wp.array(dtype=wp.int32),
                  m: wp.array(dtype=wp.int32), hi: wp.array(dtype=wp.int32),
                  num: wp.array(dtype=wp.int32), den: wp.array(dtype=wp.int32),
                  out_n: wp.array(dtype=wp.int32), out_range: wp.array(dtype=wp.int32),
                  out_chance: wp.array(dtype=wp.int32)):
    """Apply Showdown's three mappings to raw draws, for the parity test."""
    i = wp.tid()
    out_n[i] = rng_.random_n(raw[i], n[i])
    out_range[i] = rng_.random_range(raw[i], m[i], hi[i])
    if rng_.random_chance(raw[i], num[i], den[i]):
        out_chance[i] = 1


@wp.kernel
def reset(s: State, dex: Dex, pool: wp.array3d(dtype=wp.int16), pool_size: int, start: int,
          seed: wp.uint32, rng_mode: int):
    """Deal every battle two pool teams: battle b takes rows start + 2b and the
    one after, so a seed deals the same teams on any device."""
    b = wp.tid()
    deal_.deal(s, dex, pool, pool_size, start + 2 * b, seed, rng_mode, b)


@wp.kernel
def legal_mask(s: State, dex: Dex, out: wp.array2d(dtype=wp.int32)):
    """Legal actions per side, as a bitmask over the 12 action codes."""
    b, side = wp.tid()
    out[b, side] = legal_.legal_actions(s, dex, b, side)


@wp.kernel
def emerald_replacements(s: State, dex: Dex, out: wp.array2d(dtype=wp.int32)):
    """The Emerald cartridge AI's replacement per side, as an action code, or
    -1 where the side is not being asked for one."""
    b, side = wp.tid()
    out[b, side] = -1
    if int(s.request[b, side]) == mask_.REQUEST_SWITCH:
        out[b, side] = mask_.SWITCH_BASE + policy_emerald.choose_replacement(s, dex, b, side)


@wp.kernel
def obs(s: State, dex: Dex, out: wp.array3d(dtype=wp.int16)):
    """Each player's observation of each battle, obs_layout.py's vector."""
    b, p = wp.tid()
    for i in range(out.shape[2]):
        out[b, p, i] = wp.int16(0)
    obs_.observe(s, dex, out, b, p)


@wp.kernel
def settle(s: State, dex: Dex, pool: wp.array3d(dtype=wp.int16), pool_size: int, cursor: wp.array(dtype=wp.int32),
           seed: wp.uint32, rng_mode: int, max_steps: int, steps: wp.array(dtype=wp.int32),
           reward: wp.array(dtype=wp.float32), done: wp.array(dtype=wp.uint8),
           truncated: wp.array(dtype=wp.uint8)):
    """The training step's second half, after `step_idx` over every battle:
    score, and deal a finished battle again.

    `reward` is +1, -1 or 0 from p1's side at a game end and 0 otherwise; the
    turn-1,000 tie is a game end like any other. `truncated` is the trainer's
    cap on decisions (0 for none), not a game end. Either one deals the slot
    two fresh teams, so the observation that follows is the new battle's.
    """
    b = wp.tid()
    steps[b] = steps[b] + 1
    result = int(s.result[b])
    r = float(0.0)
    finished = result != mask_.RESULT_ONGOING
    if result == mask_.RESULT_P1:
        r = 1.0
    elif result == mask_.RESULT_P2:
        r = -1.0
    capped = not finished and max_steps > 0 and steps[b] >= max_steps
    reward[b] = r
    done[b] = wp.uint8(wp.where(finished, 1, 0))
    truncated[b] = wp.uint8(wp.where(capped, 1, 0))
    if finished or capped:
        row = wp.atomic_add(cursor, 0, 2)
        deal_.deal(s, dex, pool, pool_size, row, seed, rng_mode, b)
        steps[b] = 0


@wp.kernel
def step_idx(s: State, dex: Dex, actions: wp.array2d(dtype=wp.int32),
             log: wp.array2d(dtype=wp.uint32), idx: wp.array(dtype=wp.int32)):
    """Advance the listed battles to their next decision point, skipping any
    whose p1 action is -1. The only kernel that calls `advance`: a kernel
    inlines every function it calls, so each caller was another full copy of
    the step for nvcc to optimise. Training passes every slot, search its
    lanes, replay the cases still running."""
    b = idx[wp.tid()]
    if actions[b, 0] < 0:
        return
    turn_.advance(s, dex, log, b, actions[b, 0], actions[b, 1])
    faint_.close_requests(s, b)

@wp.kernel
def state_hash(s: State, position: int, out: wp.array(dtype=wp.uint64)):
    """The canonical hash of every battle: full, or position-only for search."""
    b = wp.tid()
    if position == 1:
        out[b] = hashes.position_hash(s, b)
    else:
        out[b] = hashes.full_hash(s, b)
