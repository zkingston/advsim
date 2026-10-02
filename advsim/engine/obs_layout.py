"""The observation layout (OBS_VERSION below): one definition, plain Python, read by
the `obs` kernel (engine/obs.py), the live converter and anyone decoding.

One int16 vector per battle per player. Everything in it is something that
player's Showdown stream shows or lets it count, so the converter can rebuild
it from the protocol and the request JSON alone:

- 12 Pokemon tokens: the player's own party in its fixed order, then the
  foe's seen Pokemon in species-id order and zeros for the rest: the foe's
  party order is never shown. Own moves are in slot order; a foe's revealed
  moves are in move-id order, since the protocol names a move but never its
  slot.
- 2 active tokens, own then foe: boosts, current types and the visible
  volatiles.
- 1 field token.
- Matchup: each of the own active's moves, and each revealed foe move, against
  the other active's current types.
- The legal-action mask, one entry per action code.
- History (version 2): the move each active Pokemon last used since it came
  in, own then foe, or 0. Appended after the mask, so every earlier column
  keeps its place and version-1 networks read version-2 vectors unchanged.
- Own stats (version 3): Atk, Def, SpA, SpD and Spe of each of the player's own
  Pokemon in party order, as the request lists them (a transformed Pokemon's
  own, not its copy's). Appended after the history, so earlier columns keep
  their place. Networks derive damage ranges from these (advsim/damage.py).
- Move order (version 4): +1 if the player's move went first in the turn in
  progress, or the last one until the next begins, -1 if the foe's did, 0
  unless both sides used a move of their own action at the same priority, the
  one case the order tells Speed. A move line without `[from]` in the protocol
  is such a use; a called move, a reflected one and a Pursuit chase are not.
"""
OBS_VERSION = 4

MON = ('present', 'species', 'level', 'hp_pct', 'hp', 'maxhp', 'status', 'toxic_stage', 'fainted',
       'active', 'ability', 'ability_known', 'item', 'item_known',
       'move0', 'move1', 'move2', 'move3', 'pp0', 'pp1', 'pp2', 'pp3')
ACTIVE = ('boost_atk', 'boost_def', 'boost_spa', 'boost_spd', 'boost_spe', 'boost_acc', 'boost_eva',
          'type0', 'type1', 'substitute', 'confusion', 'leech_seed', 'encore', 'partial_trap', 'yawn',
          'perish', 'attract', 'transformed', 'flash_fire', 'destiny_bond', 'charging', 'recharge',
          'trapped', 'choice_move')
FIELD = ('weather', 'weather_turns', 'turn', 'spikes_own', 'spikes_foe', 'wish_own', 'wish_foe')
MATCHUP = tuple(f'{who}_move{i}_{what}' for who in ('own', 'foe') for i in range(4) for what in ('eff', 'stab'))
MASK = tuple(f'legal{a}' for a in range(12))
HISTORY = ('own_last_move', 'foe_last_move')
STATS = ('atk', 'def', 'spa', 'spd', 'spe')
ORDER = ('moved_first',)

MON_BASE = 0
ACTIVE_BASE = MON_BASE + 12 * len(MON)
FIELD_BASE = ACTIVE_BASE + 2 * len(ACTIVE)
MATCHUP_BASE = FIELD_BASE + len(FIELD)
MASK_BASE = MATCHUP_BASE + len(MATCHUP)
HISTORY_BASE = MASK_BASE + len(MASK)
STATS_BASE = HISTORY_BASE + len(HISTORY)
ORDER_BASE = STATS_BASE + 6 * len(STATS)
OBS_DIM = ORDER_BASE + len(ORDER)


def names() -> list[str]:
    """Every column's name, for decoding a vector or diffing two."""
    out = [f'{who}{i}.{n}' for who in ('own', 'foe') for i in range(6) for n in MON]
    out += [f'{who}_active.{n}' for who in ('own', 'foe') for n in ACTIVE]
    out += [f'field.{n}' for n in FIELD] + list(MATCHUP) + list(MASK) + list(HISTORY)
    out += [f'own{i}.{n}' for i in range(6) for n in STATS]
    out += list(ORDER)
    assert len(out) == OBS_DIM
    return out
