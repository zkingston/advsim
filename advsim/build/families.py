"""Hand-kept map from a dex entry to an effect family and its parameters.

Every entry in the vocabulary that carries a callback must appear here, either
with a family or in DEAD with a reason. `vocab.py` fails the build on
anything unmapped, so coverage holds by construction rather than by review.

Parameters are written symbolically ('Fire', 'atk', 'brn') and resolved to IDs
by `vocab.py`, which owns the vocabulary. Each family's parameter order is
in the comment above its members. A family named after a single entry is a
one-off: it gets its own `@wp.func` rather than sharing a parametric one.
"""

ONE_OFF = 'one_off'

# Move flags with runtime effect in this vocabulary. The dump carries 30, but most
# belong to later generations or to callers that do not exist here (Metronome,
# Mirror Move, Snatch, Magic Coat, Heal Block, Minimize), so they are dropped.
MOVE_FLAGS: dict[str, str] = {
    'protect': 'blocked by Protect and Detect',
    'contact': 'triggers Static, Rough Skin, Cute Charm, Effect Spore, Flame Body, Poison Point',
    'sound': 'blocked by Soundproof',
    'bypasssub': 'passes through Substitute',
    'nosleeptalk': 'Sleep Talk cannot call it',
    'failencore': 'Encore cannot lock it',
    'recharge': 'Hyper Beam spends the next turn recharging',
    'charge': 'Solar Beam spends a turn charging',
    'defrost': 'thaws a frozen user',
    'mustpressure': 'costs 2 PP against Pressure even when it does not target the foe',
}

# ---------------------------------------------------------------- moves
# Parameter order per family:
#   protect          (leaves_1hp,)
#   reflect_damage   (category to reflect,)
#   self_cure        (sleep counter Showdown writes; 0 = no sleep, Rest writes 3,)
#   cure_team        (1 if the deaf are passed over,)
#   the rest take no parameters.
MOVES: dict[str, tuple[str, tuple]] = {
    'flail': ('hp_scaled_power', ()),
    'reversal': ('hp_scaled_power', ()),
    'facade': ('status_boosted_power', ()),
    'synthesis': ('heal_weather', ()),
    'morningsun': ('heal_weather', ()),
    'moonlight': ('heal_weather', ()),
    # Heal Bell is a sound move and passes over a Soundproof holder;
    # Aromatherapy is not, and cures the whole party.
    'healbell': ('cure_team', (1,)),
    'aromatherapy': ('cure_team', (0,)),
    'protect': ('protect', (0,)),
    'endure': ('protect', (1,)),
    'meanlook': ('trap_foe', ()),
    'spiderweb': ('trap_foe', ()),
    'counter': ('reflect_damage', ('Physical',)),
    'mirrorcoat': ('reflect_damage', ('Special',)),
    'thunder': ('weather_accuracy', ()),
    'solarbeam': ('charge_turn', ()),
    'rest': ('self_cure', (3,)),
    'refresh': ('self_cure', (0,)),
    # One-offs: each gets its own function.
    'transform': (ONE_OFF, ()),
    'sleeptalk': (ONE_OFF, ()),
    'batonpass': (ONE_OFF, ()),
    'substitute': (ONE_OFF, ()),
    'trick': (ONE_OFF, ()),
    'knockoff': (ONE_OFF, ()),
    'pursuit': (ONE_OFF, ()),
    'focuspunch': (ONE_OFF, ()),
    'rapidspin': (ONE_OFF, ()),
    'bellydrum': (ONE_OFF, ()),
    'curse': (ONE_OFF, ()),
    'painsplit': (ONE_OFF, ()),
    'haze': (ONE_OFF, ()),
    'perishsong': (ONE_OFF, ()),
    'destinybond': (ONE_OFF, ()),
    'encore': (ONE_OFF, ()),
    'wish': (ONE_OFF, ()),
    'yawn': (ONE_OFF, ()),
    'leechseed': (ONE_OFF, ()),
    'spikes': (ONE_OFF, ()),
}

# ---------------------------------------------------------------- abilities
# Parameter order per family:
#   stat_mult        (stat mask, modifier, condition: 0 always, 1 when statused,
#                     2 when another active holds the other half of the pair,
#                     mode: 0 chains with other modifiers, 1 applies on its own)
#   weather_speed    (weather,)
#   pinch_type_boost (move type,)                 x1.5 at a third HP or less
#   status_immune    (condition it blocks,)
#   type_immune      (move type, effect: 0 none, 1 heal a quarter, 2 Flash Fire)
#   contact_punish   (numerator, denominator, effect)
#   contact_recoil   (denominator of the attacker's max HP,)
#   block_drops      (stat mask the foe cannot lower,)
#   weather_setter   (weather,)
#   trap             (0 anything, 1 grounded only, 2 Steel only)
#   residual_self    (0 Speed Boost, 1 Shed Skin)
#   crit_immune, weather_suppress take none.
#
# Hustle is the one stat multiplier Showdown applies with `modify` rather than
# `chainModify`, so it rounds separately from an item on the same stat.
ABILITIES: dict[str, tuple[str, tuple]] = {
    'hugepower': ('stat_mult', ('stats:atk', 'x2/1', 0, 0)),
    'purepower': ('stat_mult', ('stats:atk', 'x2/1', 0, 0)),
    'hustle': ('stat_mult', ('stats:atk', 'x3/2', 0, 1)),
    'guts': ('stat_mult', ('stats:atk', 'x3/2', 1, 0)),
    'marvelscale': ('stat_mult', ('stats:def', 'x3/2', 1, 0)),
    # getAllActive spans both sides, so in singles these pair across the field.
    'plus': ('stat_mult', ('stats:spa', 'x3/2', 2, 0)),
    'minus': ('stat_mult', ('stats:spa', 'x3/2', 2, 0)),
    'swiftswim': ('weather_speed', ('rain',)),
    'chlorophyll': ('weather_speed', ('sun',)),
    'overgrow': ('pinch_type_boost', ('Grass',)),
    'blaze': ('pinch_type_boost', ('Fire',)),
    'torrent': ('pinch_type_boost', ('Water',)),
    'swarm': ('pinch_type_boost', ('Bug',)),
    'insomnia': ('status_immune', ('slp',)),
    'vitalspirit': ('status_immune', ('slp',)),
    'limber': ('status_immune', ('par',)),
    'immunity': ('status_immune', ('psn',)),
    'waterveil': ('status_immune', ('brn',)),
    'magmaarmor': ('status_immune', ('frz',)),
    'owntempo': ('status_immune', ('confusion',)),
    'oblivious': ('status_immune', ('attract',)),
    'innerfocus': ('status_immune', ('flinch',)),
    'voltabsorb': ('type_immune', ('Electric', 1)),
    'waterabsorb': ('type_immune', ('Water', 1)),
    'flashfire': ('type_immune', ('Fire', 2)),
    # No callbacks in the dump: Showdown implements it in Pokemon#isGrounded, so
    # the engine still needs the behaviour.
    'levitate': ('type_immune', ('Ground', 0)),
    'static': ('contact_punish', (1, 3, 'par')),
    'flamebody': ('contact_punish', (1, 3, 'brn')),
    'poisonpoint': ('contact_punish', (1, 3, 'psn')),
    'cutecharm': ('contact_punish', (1, 3, 'attract')),
    'effectspore': ('contact_punish', (1, 10, 'spore')),
    'roughskin': ('contact_recoil', (16,)),
    'clearbody': ('block_drops', ('stats:atk+def+spa+spd+spe+acc+eva',)),
    'whitesmoke': ('block_drops', ('stats:atk+def+spa+spd+spe+acc+eva',)),
    'hypercutter': ('block_drops', ('stats:atk',)),
    'keeneye': ('block_drops', ('stats:acc',)),
    'drizzle': ('weather_setter', ('rain',)),
    'drought': ('weather_setter', ('sun',)),
    'sandstream': ('weather_setter', ('sand',)),
    'cloudnine': ('weather_suppress', ()),
    'airlock': ('weather_suppress', ()),
    'shadowtag': ('trap', (0,)),
    'arenatrap': ('trap', (1,)),
    'magnetpull': ('trap', (2,)),
    'speedboost': ('residual_self', (0,)),
    'shedskin': ('residual_self', (1,)),
    # onCriticalHit is a data field set to false, not a function, so these carry
    # no callbacks either; they still need the check.
    'shellarmor': ('crit_immune', ()),
    'battlearmor': ('crit_immune', ()),
    'intimidate': (ONE_OFF, ()),
    'trace': (ONE_OFF, ()),
    'forecast': (ONE_OFF, ()),
    'wonderguard': (ONE_OFF, ()),
    'soundproof': (ONE_OFF, ()),
    'truant': (ONE_OFF, ()),
    'naturalcure': (ONE_OFF, ()),
    'synchronize': (ONE_OFF, ()),
    'colorchange': (ONE_OFF, ()),
    'stickyhold': (ONE_OFF, ()),
    'suctioncups': (ONE_OFF, ()),
    'liquidooze': (ONE_OFF, ()),
    'rockhead': (ONE_OFF, ()),
    'shielddust': (ONE_OFF, ()),
    'serenegrace': (ONE_OFF, ()),
    'compoundeyes': (ONE_OFF, ()),
    'sandveil': (ONE_OFF, ()),
    'thickfat': (ONE_OFF, ()),
    'pressure': (ONE_OFF, ()),
    # Early Bird carries no callbacks of its own; the sleep condition reads it.
    'earlybird': (ONE_OFF, ()),
}

# ---------------------------------------------------------------- accuracy
# Scaling a move's accuracy is its own event, separate from the stat families
# above, which is why Hustle appears here as well as there. Showdown chains
# these in handler-priority order and truncates at each step, so the order is
# part of the data.
#   modifier, whose: 0 the Pokemon using the move, 1 the one it is aimed at,
#   when: 0 always, 1 the move's type is one gen 3 treats as physical, 2 sandstorm,
#   priority: Showdown's handler priority, highest first.
ACCURACY: dict[str, tuple] = {
    'compoundeyes': ('x13/10', 0, 0, 9),
    'sandveil': ('x4/5', 1, 2, 8),
    'hustle': ('x3277/4096', 0, 1, 7),
}

# ---------------------------------------------------------------- items
# Parameter order per family:
#   stat_item     (stat mask, modifier, lowest dex number, highest dex number)
#   type_item     (stat mask, modifier, move type)
#   crit_item     (added crit stages, lowest dex number, highest dex number)
#   residual_heal (denominator: heals maxhp over this)
#   pinch_berry   (stat mask to raise by one, denominator of the HP threshold)
#   cure_item     (what to clear: 'status' or 'boosts')
#
# A dex-number range of 0 means any species. Two items need a range rather than
# one species: Thick Club works for Cubone and Marowak, Soul Dew for Latias and
# Latios, and each pair is consecutive in the dex.
ITEMS: dict[str, tuple[str, tuple]] = {
    'choiceband': ('stat_item', ('stats:atk', 'x3/2', 0, 0)),
    'lightball': ('stat_item', ('stats:spa', 'x2/1', 25, 25)),
    'thickclub': ('stat_item', ('stats:atk', 'x2/1', 104, 105)),
    'souldew': ('stat_item', ('stats:spa+spd', 'x3/2', 380, 381)),
    'stick': ('crit_item', (2, 83, 83)),
    'silkscarf': ('type_item', ('stats:atk', 'x11/10', 'Normal')),
    'twistedspoon': ('type_item', ('stats:spa', 'x11/10', 'Psychic')),
    'leftovers': ('residual_heal', (16,)),
    'salacberry': ('pinch_berry', ('stats:spe', 4)),
    'liechiberry': ('pinch_berry', ('stats:atk', 4)),
    'petayaberry': ('pinch_berry', ('stats:spa', 4)),
    'lumberry': ('cure_item', ('status',)),
    'whiteherb': ('cure_item', ('boosts',)),
}

# ---------------------------------------------------------------- conditions
# Durations come from the dump, not from here; this only fixes the family.
CONDITIONS: dict[str, str] = {
    'brn': 'status', 'par': 'status', 'psn': 'status', 'tox': 'status', 'slp': 'status', 'frz': 'status',
    'sunnyday': 'weather', 'raindance': 'weather', 'sandstorm': 'weather',
    'spikes': 'side', 'wish': 'slot',
    'confusion': 'volatile_timer', 'encore': 'volatile_timer', 'yawn': 'volatile_timer',
    'partiallytrapped': 'volatile_timer', 'mustrecharge': 'volatile_timer', 'perishsong': 'volatile_timer',
    'flinch': 'volatile_flag', 'protect': 'volatile_flag', 'endure': 'volatile_flag',
    'substitute': 'volatile_flag', 'leechseed': 'volatile_flag', 'destinybond': 'volatile_flag',
    'curse': 'volatile_flag', 'attract': 'volatile_flag', 'trapped': 'volatile_flag',
}

# Entries whose callbacks cannot fire in this vocabulary. The build reports these
# so a callback is never ignored silently.
DEAD: dict[str, str] = {
    'brickbreak': 'removes Reflect and Light Screen; neither move is in the vocabulary',
    'return': 'basePowerCallback over happiness, which randbats leaves at 255, so power is fixed at 102',
    'hiddenpower': 'basePowerCallback and onModifyType read the set; carried as the type_from_mon flag and power 70',
    'toxic': 'gen3 blanks the inherited onPrepareHit, so its lock-on condition can never be added',
    'struggle': 'onModifyMove only makes it typeless, which the type column carries as ???',
    'sturdy': 'gen 3 Sturdy only refuses an OHKO move, and this vocabulary has none',
}

MOVE_FAMILIES = tuple(sorted({f for f, _ in MOVES.values()} - {ONE_OFF}))
ABILITY_FAMILIES = tuple(sorted({f for f, _ in ABILITIES.values()} - {ONE_OFF}))
ITEM_FAMILIES = tuple(sorted({f for f, _ in ITEMS.values()}))


def uncovered(seen: set[str]) -> list[str]:
    """Families and one-offs that nothing in `seen` (dex ids a battle touched)
    reaches: a family counts once any of its members is touched."""
    gap = set()
    for table in (MOVES, ABILITIES, ITEMS, CONDITIONS):
        # CONDITIONS carries a bare family name; the rest carry (family, params).
        fam_of = {n: (v[0] if isinstance(v, tuple) else v) for n, v in table.items()}
        covered = {fam for name, fam in fam_of.items() if name in seen}
        for name, fam in fam_of.items():
            if fam == ONE_OFF and name not in seen:
                gap.add(name)
            elif fam != ONE_OFF and fam not in covered:
                gap.add(fam)
    return sorted(gap)
