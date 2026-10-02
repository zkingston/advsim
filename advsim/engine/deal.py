"""Dealing a battle: two pool teams into a battle slot that `clear_battle`
has zeroed, which is what "no volatile, no weather, turn 0" means, so only
the rest is written. `reset` deals every slot and `settle` deals a finished one.
"""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import rng as rng_
from advsim.engine._generated import ids
from advsim.engine._generated.state import State, clear_battle
from advsim.engine.dex import Dex



@wp.func
def deal(s: State, dex: Dex, pool: wp.array3d(dtype=wp.int16), pool_size: int, row: int,
         seed: wp.uint32, rng_mode: int, b: int):
    """Battle b takes pool rows `row` and `row + 1` (wrapping), from scratch."""
    clear_battle(s, b)
    for side in range(2):
        team = (row + side) % pool_size
        for slot in range(6):
            s.species[b, side, slot] = wp.uint16(pool[team, slot, ids.POOL_SPECIES])
            s.level[b, side, slot] = wp.uint8(pool[team, slot, ids.POOL_LEVEL])
            s.gender[b, side, slot] = wp.uint8(pool[team, slot, ids.POOL_GENDER])
            s.ability[b, side, slot] = wp.uint8(pool[team, slot, ids.POOL_ABILITY])
            s.item[b, side, slot] = wp.uint8(pool[team, slot, ids.POOL_ITEM])
            s.hp_type[b, side, slot] = wp.uint8(pool[team, slot, ids.POOL_HP_TYPE])
            maxhp = int(pool[team, slot, ids.POOL_MAXHP])
            s.maxhp[b, side, slot] = wp.uint16(maxhp)
            s.hp[b, side, slot] = wp.uint16(maxhp)
            for j in range(5):
                s.stats[b, side, slot, j] = wp.uint16(pool[team, slot, ids.POOL_ATK + j])
            # setSpecies writes the cached Speed every sort reads, and nothing
            # else does until a Pokemon is active and updateSpeed runs.
            s.cached_spe[b, side, slot] = wp.uint16(pool[team, slot, ids.POOL_ATK + 4])
            for j in range(4):
                s.moves[b, side, slot, j] = wp.uint8(pool[team, slot, ids.POOL_MOVE1 + j])
                s.pp[b, side, slot, j] = wp.uint8(pool[team, slot, ids.POOL_PP1 + j])
                s.max_pp[b, side, slot, j] = wp.uint8(pool[team, slot, ids.POOL_PP1 + j])
            s.party_pos[b, side, slot] = wp.uint8(slot)
        s.request[b, side] = wp.uint8(mask_.REQUEST_MOVE)
        s.revealed[b, side, 0] = wp.uint16(1)  # the lead's switch line
        s.alive_mask[b, side] = wp.uint8(63)  # six living party members
        lead = int(pool[team, 0, ids.POOL_SPECIES])  # the active starts as party slot 0
        s.types[b, side, 0] = wp.uint8(dex.species_type1[lead])
        s.types[b, side, 1] = wp.uint8(dex.species_type2[lead])
    # A key per deal, from the team row: a battle slot dealt again does not
    # replay the stream it had last time.
    s.rng_key[b] = rng_.mix32(seed ^ wp.uint32(row) * rng_.KNUTH)
    s.rng_mode[b] = wp.uint8(rng_mode)
