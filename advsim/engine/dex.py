"""The dex tables, as one Warp struct passed to every kernel.

One `wp.array` per table column, loaded from `artifacts/dex.npz` by `env.py`.
Never `wp.constant`: that would bake the data into compiled code, so a rebuilt
dex would need a recompile. A test asserts these members match the built
arrays exactly, since a column added by the build and missing here would be
silently unavailable to the engine.
"""
import warp as wp


@wp.struct
class Dex:
    ability_family: wp.array(dtype=wp.int32)
    ability_p0: wp.array(dtype=wp.int32)
    ability_p1: wp.array(dtype=wp.int32)
    ability_p2: wp.array(dtype=wp.int32)
    ability_p3: wp.array(dtype=wp.int32)
    item_family: wp.array(dtype=wp.int32)
    item_p0: wp.array(dtype=wp.int32)
    item_p1: wp.array(dtype=wp.int32)
    item_p2: wp.array(dtype=wp.int32)
    item_p3: wp.array(dtype=wp.int32)
    move_accuracy: wp.array(dtype=wp.int32)
    move_ai_skip: wp.array(dtype=wp.int32)
    move_boosts: wp.array(dtype=wp.int32)
    move_category: wp.array(dtype=wp.int32)
    move_crit_stage: wp.array(dtype=wp.int32)
    move_drain_den: wp.array(dtype=wp.int32)
    move_drain_num: wp.array(dtype=wp.int32)
    move_family: wp.array(dtype=wp.int32)
    move_fixed_damage: wp.array(dtype=wp.int32)
    move_flags: wp.array(dtype=wp.int32)
    move_heal_den: wp.array(dtype=wp.int32)
    move_heal_num: wp.array(dtype=wp.int32)
    move_ignore_immunity: wp.array(dtype=wp.int32)
    move_multihit_min: wp.array(dtype=wp.int32)
    move_p0: wp.array(dtype=wp.int32)
    move_power: wp.array(dtype=wp.int32)
    move_pp: wp.array(dtype=wp.int32)
    move_priority: wp.array(dtype=wp.int32)
    move_recoil_den: wp.array(dtype=wp.int32)
    move_recoil_num: wp.array(dtype=wp.int32)
    move_sec_boosts: wp.array(dtype=wp.int32)
    move_sec_chance: wp.array(dtype=wp.int32)
    move_sec_self_boosts: wp.array(dtype=wp.int32)
    move_sec_status: wp.array(dtype=wp.int32)
    move_sec_volatile: wp.array(dtype=wp.int32)
    move_self_boosts: wp.array(dtype=wp.int32)
    move_selfdestruct: wp.array(dtype=wp.int32)
    move_status: wp.array(dtype=wp.int32)
    move_switch_mode: wp.array(dtype=wp.int32)
    move_target: wp.array(dtype=wp.int32)
    move_type: wp.array(dtype=wp.int32)
    move_type_from_mon: wp.array(dtype=wp.int32)
    move_volatile: wp.array(dtype=wp.int32)
    move_weather: wp.array(dtype=wp.int32)
    ability_acc_mod: wp.array(dtype=wp.int32)
    ability_acc_whose: wp.array(dtype=wp.int32)
    ability_acc_when: wp.array(dtype=wp.int32)
    ability_acc_prio: wp.array(dtype=wp.int32)
    species_base_stats: wp.array2d(dtype=wp.int32)
    species_num: wp.array(dtype=wp.int32)
    species_type1: wp.array(dtype=wp.int32)
    species_type2: wp.array(dtype=wp.int32)
    type_ai_order: wp.array2d(dtype=wp.int32)
    type_chart: wp.array2d(dtype=wp.int32)
    type_is_special: wp.array(dtype=wp.int32)
    type_status_immune: wp.array(dtype=wp.int32)


# Column groups, for the tests and for anyone reading the kernels.
