"""The single definition of battle state.

Warp arrays, the NumPy mirror, the hash word order and `artifacts/layout.json`
for `export_state.js` are all derived from FIELDS. Nothing else may declare a
state field, because the exporter and the hash would drift from the engine.

`scope` gives an array's shape: field [B], side [B, 2], slot [B, 2] for the
active slot, mon [B, 2, 6]. `width` is a trailing per-entry width, so the stats
of one Pokemon are [B, 2, 6, 5]. A field owned by a flag that is off must be
zero; the canonical hash depends on it.
"""
from __future__ import annotations

from dataclasses import dataclass

FIELD, SIDE, SLOT, MON = 'field', 'side', 'slot', 'mon'
SHAPES = {FIELD: (), SIDE: (2,), SLOT: (2,), MON: (2, 6)}


@dataclass(frozen=True)
class Field:
    """`hashed` means the position hash; the full hash covers everything but `err`.

    `exported` means Showdown can speak to the field, so `export_state.js` fills
    it and a replay compares it. The rest is the engine's own bookkeeping, which
    a battle loaded from an export starts fresh.

    `hidden` is who cannot see it, which `determinize` must resample and the
    leak test scrambles (SPEC §Determinization): '' anyone; 'unseen' the foe,
    until the Pokemon is seen; 'reveal' the foe, until its own `revealed` bit
    (a move slot, the ability, the item); 'foe' the foe, always; 'both' either
    player; 'party' a party slot or a per-slot bitmask, which the foe's hidden
    party order moves.
    """

    name: str
    dtype: str
    scope: str
    hashed: bool = True
    exported: bool = True
    width: int = 1
    note: str = ''
    hidden: str = ''

    def shape(self, batch: int) -> tuple[int, ...]:
        trailing = (self.width,) if self.width > 1 else ()
        return (batch,) + SHAPES[self.scope] + trailing

    def elems(self) -> int:
        n = self.width
        for d in SHAPES[self.scope]:
            n *= d
        return n


FIELDS: tuple[Field, ...] = (
    # ---- field
    Field('weather', 'u8', FIELD, note='none, sun, rain, sand'),
    Field('weather_turns', 'u8', FIELD, note='0 turns means ability-set and permanent'),
    Field('turn', 'u16', FIELD, hashed=False, note='excluded from the position hash; Endless Battle Clause ends at 1,000'),
    Field('result', 'u8', FIELD, note='ongoing, p1, p2, tie'),
    Field('last_used', 'u8', FIELD, note="Showdown's battle.lastMove: the last move past BeforeMove and PP, Sleep Talk rather than its call; the Emerald chooser reads it"),
    Field('rng_key', 'u32', FIELD, exported=False, hashed=False, hidden='both'),
    Field('rng_ctr', 'u32', FIELD, exported=False, hashed=False, note='replay mode uses this as a log cursor', hidden='both'),
    Field('rng_mode', 'u8', FIELD, exported=False, hashed=False, note='train, paired, replay'),
    Field('err', 'u8', FIELD, exported=False, hashed=False, note='bits: illegal action (mask.ERR_ILLEGAL), replay log exhausted (mask.ERR_LOG_EXHAUSTED)'),
    Field('phase', 'u8', FIELD, exported=False, note='how far into the turn: a faint pauses it for a replacement'),
    Field('turn_first', 'u8', FIELD, exported=False, note='which side won the order roll for the turn in progress'),
    Field('pending_act', 'u8', FIELD, exported=False, width=2, note='the actions chosen for the turn in progress'),
    # ---- side
    Field('active', 'u8', SIDE, note='index into the party', hidden='party'),
    Field('request', 'u8', SIDE, note='none, move, forced switch, wait'),
    Field('spikes', 'u8', SIDE, note='0 to 3 layers'),
    Field('wish_turns', 'u8', SIDE, note='gen 3 heals half the max HP of whoever is in the slot, so no amount is stored'),
    Field('alive_mask', 'u8', SIDE, note='6 bits', hidden='party'),
    Field('knocked_mask', 'u8', SIDE,
          note='6 bits: gen 3 Knock Off leaves a Pokemon unable to take an item for the rest of the battle', hidden='party'),
    Field('truant_mask', 'u8', SIDE,
          note='6 bits: `truantTurn` is a property of the Pokemon, not a volatile, so it survives a '
               'switch. Only a traced Truant can see that, since a real one resets it on the way in', hidden='party'),
    # ---- active slot, cleared on switch unless Baton Pass
    Field('boosts', 'u32', SLOT, note='7 signed nibbles: atk, def, spa, spd, spe, acc, eva'),
    Field('vflags', 'u32', SLOT, note='one bit per volatile'),
    Field('sub_hp', 'u16', SLOT, hidden='foe'),
    Field('confusion_turns', 'u8', SLOT, hidden='both'),
    Field('encore_turns', 'u8', SLOT, hidden='both'),
    Field('encore_move', 'u8', SLOT),
    Field('choice_move', 'u8', SLOT, hidden='reveal', note='the foe knows the lock only once it knows the item'),
    Field('trap_turns', 'u8', SLOT, hidden='both'),
    Field('trap_source', 'u8', SLOT, note='party slot of the Pokemon holding the trap, plus one; the trap ends when it is no longer the one across the field. The holder is on the other side, so it moves with that side\'s order', hidden='party'),
    Field('perish_count', 'u8', SLOT),
    Field('yawn_turns', 'u8', SLOT),
    Field('stall_ctr', 'u8', SLOT, note='consecutive Protect or Endure'),
    Field('twoturn_move', 'u8', SLOT, note='Solar Beam charge'),
    Field('last_move', 'u8', SLOT, note='Encore target'),
    Field('dmg_taken', 'u16', SLOT, exported=False, note='Counter and Mirror Coat', hidden='foe'),
    Field('dmg_cat', 'u8', SLOT, exported=False),
    Field('types', 'u8', SLOT, width=2, note='current types: Forecast, Color Change, Transform'),
    Field('turn_flags', 'u16', SLOT, exported=False, note='what this side has already done this turn: switched in, released a charge, queued a beforeTurnMove, and so on'),
    Field('moved_at', 'u8', SLOT, exported=False, note='this turn (or the last, until the next starts): 1 if this side used a move of its own action first, 2 second, 0 not at all; what the protocol shows as its move line'),
    Field('moved_prio', 'u8', SLOT, exported=False, note="that move's priority plus 8"),
    Field('xf_stats', 'u16', SLOT, width=5,
          note='what a transformed Pokemon was before it copied anything; the live fields carry the copy, the way Showdown mutates the Pokemon itself', hidden='foe'),
    Field('xf_moves', 'u8', SLOT, width=4, hidden='reveal'),
    Field('xf_pp', 'u8', SLOT, width=4, hidden='reveal'),
    Field('xf_max_pp', 'u8', SLOT, width=4,
          note='Showdown keeps the PP Ups on the Pokemon and rebuilds a transformed '
               "slot's maxpp from them, so a copy needs the original back to revert to", hidden='foe'),
    Field('xf_ability', 'u8', SLOT, hidden='reveal'),
    Field('xf_species', 'u16', SLOT),
    Field('xf_hp_type', 'u8', SLOT, hidden='foe'),
    Field('base_ability', 'u8', SLOT,
          note='what a Trace holder had before it copied, plus one; the copy goes when it leaves'),

    # ---- per Pokemon
    Field('cached_spe', 'u16', MON,
          note="Showdown's `pokemon.speed`, which every sort reads. setSpecies writes the raw stat "
               'and only updateSpeed refreshes it, and only for the actives, so a Pokemon keeps the '
               'Speed it last had on the field while it sits on the bench', hidden='foe'),
    Field('hp', 'u16', MON, note='absolute; the observation converts to percent', hidden='foe'),
    Field('status', 'u8', MON, hidden='unseen'),
    Field('status_ctr', 'u8', MON, note='sleep turns or toxic stage. Sleep a foe caused lasts a hidden roll; Rest and the toxic stage are counted in the open', hidden='both'),
    Field('sleep_skipped', 'u8', MON,
          note='gen 3 gives back the turns Sleep Talk spent asleep when the Pokemon switches back in'),
    Field('slept_by_foe', 'u8', MON, note='Sleep Clause Mod; Rest does not count'),
    Field('species', 'u16', MON, hidden='unseen'),
    Field('level', 'u8', MON, hidden='unseen'),
    Field('gender', 'u8', MON, note='Cute Charm', hidden='unseen'),
    Field('ability', 'u8', MON, hidden='reveal'),
    Field('item', 'u8', MON, note='mutable: Trick, Knock Off', hidden='reveal'),
    Field('hp_type', 'u8', MON, hidden='foe'),
    Field('stats', 'u16', MON, width=5, hidden='foe'),
    Field('maxhp', 'u16', MON, hidden='foe'),
    Field('moves', 'u8', MON, width=4, hidden='reveal'),
    Field('pp', 'u8', MON, width=4, hidden='reveal'),
    Field('max_pp', 'u8', MON, width=4, hidden='foe'),
    Field('party_pos', 'u8', MON,
          note='where this Pokemon sits in Showdown\'s own party array, which a drag samples over', hidden='foe'),
    Field('revealed', 'u16', MON, hashed=False,
          note='what the foe has seen: bit 0 the Pokemon, 1-4 its move slots, 5 ability, 6 item; '
               'excluded from the position hash, which is about the game and not who knows what'),
)

BY_NAME = {f.name: f for f in FIELDS}
# The two hash masks. `err` is out of both: it reports a harness fault, not state.
FULL = tuple(f for f in FIELDS if f.name != 'err')
POSITION = tuple(f for f in FIELDS if f.hashed)
EXPORTED = tuple(f for f in FIELDS if f.exported)

assert len(BY_NAME) == len(FIELDS), 'duplicate state field name'
assert {f.hidden for f in FIELDS} <= {'', 'unseen', 'reveal', 'foe', 'both', 'party'}


def word_offsets() -> dict[str, int]:
    """Where each field starts in the canonical word order, which salts the hash.

    One u32 word per scalar, fields in declaration order, row-major within a
    field. Offsets count every field, masked or not, so a field leaving the
    position hash cannot move another field's salt.
    """
    offsets, at = {}, 0
    for f in FIELDS:
        offsets[f.name] = at
        at += f.elems()
    return offsets


N_WORDS = sum(f.elems() for f in FIELDS)


def as_json() -> list[dict]:
    """artifacts/layout.json: what export_state.js and the NumPy mirror read."""
    return [{'name': f.name, 'dtype': f.dtype, 'scope': f.scope, 'width': f.width,
             'hashed': f.hashed, 'exported': f.exported, 'hidden': f.hidden, 'shape': list(f.shape(1)[1:]),
             'note': f.note} for f in FIELDS]


def bytes_per_battle() -> int:
    size = {'u8': 1, 'u16': 2, 'u32': 4}
    return sum(size[f.dtype] * f.elems() for f in FIELDS)
