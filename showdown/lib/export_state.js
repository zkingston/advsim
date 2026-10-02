// A Showdown `Battle` as engine state: one value per field of the layout.
//
// This is the converter the hash compare needs. `advsim/statehash.py` hashes
// what this returns and the engine hashes its own arrays; the two must agree at
// every decision point, so a field this file gets wrong shows up as a hash
// mismatch instead of as silence.
//
// Fields the layout marks `exported: false` are engine bookkeeping with no
// Showdown counterpart: the RNG cursor, how far into a turn the step is, what
// the foe has seen. A replay starts those fresh and never compares them.
'use strict';
const fs = require('fs');
const path = require('path');
const { toID } = require('pokemon-showdown').Dex;

const ART = path.join(__dirname, '..', '..', 'artifacts');
const ids = JSON.parse(fs.readFileSync(path.join(ART, 'ids.json'), 'utf8'));

// Fields Showdown cannot speak to are marked in the layout itself, so this file
// and the Python side cannot disagree on them. dmg_taken and dmg_cat are among
// them because Showdown keeps the same fact inside the Counter volatile's
// effectState, doubled, and only while that volatile is up; Counter and Mirror
// Coat are still checked through the HP they deal.
const layout = JSON.parse(fs.readFileSync(path.join(ART, 'layout.json'), 'utf8'));
const EXPORTED = layout.filter((f) => f.exported).map((f) => f.name);

// Volatile bit positions: engine/mask.py owns them, and
// tests/test_scenarios.py checks this copy against it.
const VFLAGS = {
  mustrecharge: 0, twoturnmove: 1, trapped: 2, flashfire: 3, attract: 4, protect: 5,
  endure: 6, leechseed: 7, flinch: 9, destinybond: 10, truant: 12,
};
// Transform has no volatile of its own: Showdown sets a flag on the Pokemon.
const VF_TRANSFORMED = 11;
const WEATHER = { '': 0, sunnyday: 1, raindance: 2, sandstorm: 3 };
const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const STATS = ['atk', 'def', 'spa', 'spd', 'spe'];

const index = (table, name) => {
  const i = ids[table].indexOf(name);
  if (i < 0) throw new Error(`${name} is not in the ${table} vocabulary`);
  return i;
};
const baseFormeId = (species) => species.baseSpecies.toLowerCase().replace(/[^a-z0-9]/g, '');
const moveId = (m) => (m ? index('moves', typeof m === 'string' ? m : m.id) : 0);
const pad = (list, n) => list.concat(Array(Math.max(0, n - list.length)).fill(0)).slice(0, n);

// Party order is the order the battle started in, captured once: Showdown
// swaps the incoming Pokemon into slot 0 of its own array and rewrites
// `position`, while the engine keeps the party fixed and moves an index.
function partyOrders(battle) {
  return battle.sides.map((side) => new Map([...side.pokemon].map((p, i) => [p, i])));
}

// What each Pokemon has shown its opponent, from Showdown's public log: bit 0
// seen, bits 1-4 its own move slots, 5 its ability, 6 its item. A `[from]
// ability:` names its holder in `[of]` when there is one (weather, a contact
// punisher, Cute Charm, Synchronize), except that an absorbing heal names the
// absorber first; Trace reveals both the tracer and what it copied. The private half of a `split`
// line is skipped. Parsed incrementally, since exports come every decision.
const REVEAL_SEEN = 1, REVEAL_ABILITY = 1 << 5, REVEAL_ITEM = 1 << 6;
const HOLDER_FIRST = new Set(['-heal']);
const revealCache = new WeakMap();
function reveals(battle) {
  let st = revealCache.get(battle);
  if (!st) revealCache.set(battle, st = { pos: 0, skip: false, bits: new Map(), copies: new Set() });
  const who = (ident) => {
    const m = /^(p[12])[a-z]?: (.*)$/.exec((ident || '').trim());
    if (!m) return null;
    return battle.sides.find((sd) => sd.id === m[1])?.pokemon.find((p) => p.name === m[2]) || null;
  };
  // An ability shown while transformed is the copy's, not the Pokemon's own.
  const set = (p, bit) => {
    if (p && !(bit === REVEAL_ABILITY && st.copies.has(p))) st.bits.set(p, (st.bits.get(p) || 0) | bit);
  };
  for (; st.pos < battle.log.length; st.pos++) {
    const line = battle.log[st.pos];
    if (line.startsWith('|split|')) { st.skip = true; continue; }
    if (st.skip) { st.skip = false; continue; }
    const part = line.split('|');
    const tag = part[1];
    const of = (/\[of\] ([^|]+)/.exec(line) || [])[1];
    if (tag === 'switch' || tag === 'drag') {
      const p = who(part[2]);
      for (const q of [...st.copies]) if (q.side === p?.side) st.copies.delete(q);
      set(p, REVEAL_SEEN);
    }
    if (tag === '-transform') st.copies.add(who(part[2]));
    if (tag === 'faint') st.copies.delete(who(part[2]));
    if (tag === 'move') {
      const p = who(part[2]);
      // A copy's moves are the copy's: nothing a transformed Pokemon uses counts.
      const k = p && !st.copies.has(p) ? p.baseMoveSlots.findIndex((ms) => ms.id === toID(part[3])) : -1;
      if (k >= 0) set(p, 1 << (1 + k));
    }
    if (tag === '-item' || tag === '-enditem' || /\[from\] item:/.test(line)) set(who(part[2]), REVEAL_ITEM);
    if (/ability: /.test(line)) {
      set(of && !HOLDER_FIRST.has(tag) ? who(of) : who(part[2]), REVEAL_ABILITY);
      if (tag === '-ability' && /\[from\] ability: Trace/.test(line)) set(who(of), REVEAL_ABILITY);
    }
    if (tag === '-ability') set(who(part[2]), REVEAL_ABILITY);
  }
  return st.bits;
}

// A copy of `battle` carries what was shown so far, Pokemon matched by array
// index; its own log starts wherever the copy's does.
function carryReveals(battle, copy) {
  const st = reveals(battle);
  const map = new Map(battle.sides.flatMap((sd, s) => sd.pokemon.map((p, i) => [p, copy.sides[s].pokemon[i]])));
  const old = revealCache.get(battle);
  revealCache.set(copy, {
    pos: copy.log.length, skip: false, bits: new Map([...st].map(([p, v]) => [map.get(p), v])),
    copies: new Set([...old.copies].map((p) => map.get(p))),
  });
}

function packBoosts(boosts) {
  let word = 0;
  BOOSTS.forEach((stat, i) => { word |= ((boosts[stat] || 0) & 0xF) << (4 * i); });
  return word >>> 0;
}

function vflagsOf(p) {
  let word = p.transformed ? 1 << VF_TRANSFORMED : 0;
  for (const [name, bit] of Object.entries(VFLAGS)) if (p.volatiles[name]) word |= 1 << bit;
  return word >>> 0;
}

// One Pokemon's row: what it is, what it has left, and what it is carrying.
function monWords(p, pos) {
  return {
    // The transformed copy if there is one, and either way the base forme: the
    // vocabulary has no id for Castform's weather formes, which is the engine's
    // position too, since it models Forecast as a type change.
    species: index('species', baseFormeId(p.transformed ? p.species : p.baseSpecies)),
    level: p.level,
    gender: index('genders', p.gender || 'N'),
    hp: p.hp,
    maxhp: p.maxhp,
    // 'fnt' is Showdown's way of saying fainted; the engine keeps that in the
    // alive mask and clears the status.
    status: p.status && p.status !== 'fnt' ? index('conditions', p.status) : 0,
    // Sleep counts down in `time` and toxic up in `stage`, from 0 until its
    // first residual. The other statuses carry no counter, and the engine holds
    // a 1 for them.
    status_ctr: p.status === 'slp' ? (p.statusState?.time || 0)
      : p.status === 'tox' ? (p.statusState?.stage || 0)
        : p.status && p.status !== 'fnt' ? 1 : 0,
    // Only a sleep has a skipped count. A Pokemon that fainted asleep keeps its
    // statusState under 'fnt', stale count and all.
    sleep_skipped: p.status === 'slp' ? (p.statusState?.skippedTime || 0) : 0,
    slept_by_foe: p.status === 'slp' && p.statusState?.source && p.statusState.source.side !== p.side ? 1 : 0,
    // The ability in play, copy and all; what it reverts to is base_ability or
    // xf_ability on the slot below.
    ability: index('abilities', p.ability),
    item: p.item ? index('items', p.item) : 0,
    hp_type: p.hpType ? index('types', p.hpType) : 0,
    stats: STATS.map((s) => p.storedStats[s]),
    moves: pad(p.moveSlots.map((m) => moveId(m.id)), 4),
    pp: pad(p.moveSlots.map((m) => m.pp), 4),
    max_pp: pad(p.moveSlots.map((m) => m.maxpp), 4),
    // `pokemon.speed` is a cache every sort reads: setSpecies writes the raw
    // stat and only updateSpeed refreshes it, and only for the actives.
    cached_spe: p.speed,
    party_pos: pos,
    revealed: reveals(p.battle).get(p) || 0,
  };
}

// What a transformed Pokemon was before it copied anything. The engine keeps
// the copy in the live fields, the way Showdown mutates the Pokemon itself.
function overlay(p) {
  if (!p.transformed) {
    return { xf_stats: [0, 0, 0, 0, 0], xf_moves: [0, 0, 0, 0], xf_pp: [0, 0, 0, 0],
             xf_max_pp: [0, 0, 0, 0], xf_ability: 0, xf_species: 0, xf_hp_type: 0 };
  }
  return {
    xf_stats: STATS.map((s) => p.baseStoredStats[s]),
    xf_moves: pad(p.baseMoveSlots.map((m) => moveId(m.id)), 4),
    xf_pp: pad(p.baseMoveSlots.map((m) => m.pp), 4),
    xf_max_pp: pad(p.baseMoveSlots.map((m) => m.maxpp), 4),
    xf_ability: index('abilities', p.baseAbility),
    xf_species: index('species', baseFormeId(p.baseSpecies)),
    xf_hp_type: p.baseHpType ? index('types', p.baseHpType) : 0,
  };
}

function slotWords(side, orders) {
  const p = side.active[0];
  const v = p.volatiles;
  const trapper = v.partiallytrapped?.source;
  const types = p.getTypes();
  return {
    boosts: packBoosts(p.boosts),
    vflags: vflagsOf(p),
    sub_hp: v.substitute?.hp || 0,
    confusion_turns: v.confusion?.time || 0,
    encore_turns: v.encore?.duration || 0,
    encore_move: moveId(v.encore?.move),
    choice_move: moveId(v.choicelock?.move),
    trap_turns: v.partiallytrapped?.duration || 0,
    trap_source: trapper ? orders[trapper.side.n].get(trapper) + 1 : 0,
    perish_count: v.perishsong?.duration || 0,
    yawn_turns: v.yawn?.duration || 0,
    stall_ctr: v.stall?.counter || 0,
    twoturn_move: moveId(v.twoturnmove?.move),
    last_move: moveId(p.lastMove?.id),
    types: [index('types', types[0]), index('types', types[types.length - 1])],
    ...overlay(p),
    // Trace: the original, plus one. Zero means the ability in play is its own.
    // A transformed Pokemon reports the same fact in xf_ability, because
    // Showdown keeps one `baseAbility` where the engine keeps two slots.
    base_ability: p.transformed || p.ability === p.baseAbility ? 0 : index('abilities', p.baseAbility) + 1,
  };
}

function sideWords(side, orders) {
  const req = side.activeRequest;
  let alive = 0, knocked = 0, truant = 0;
  for (const [mon, i] of orders[side.n]) {
    if (!mon.fainted) alive |= 1 << i;
    if (mon.itemKnockedOff) knocked |= 1 << i;
    // truantTurn is a property of the Pokemon, not a volatile, so it survives a
    // switch; only a traced Truant can see that.
    if (mon.truantTurn) truant |= 1 << i;
  }
  return {
    active: orders[side.n].get(side.active[0]),
    // A finished battle asks nothing: the request Showdown still holds is the
    // stale one from before the last action.
    request: side.battle.ended ? 0 : req?.forceSwitch ? 2 : (req?.wait ? 3 : 1),
    spikes: side.sideConditions.spikes?.layers || 0,
    wish_turns: side.slotConditions[0]?.wish?.duration || 0,
    alive_mask: alive,
    knocked_mask: knocked,
    truant_mask: truant,
  };
}

function result(battle) {
  if (!battle.ended) return 0;
  if (!battle.winner) return 3;
  return battle.winner === battle.p1.name ? 1 : 2;
}

// One battle as `{field: value}` in the shapes layout.json declares: a scalar
// for a field, `[p1, p2]` for a side or the active slot, `[[...6], [...6]]` per
// Pokemon. Unexported fields are absent. `orders` comes from partyOrders at the
// start of the battle and must not be rebuilt mid-battle.
function exportState(battle, orders) {
  const out = {
    weather: WEATHER[battle.field.weather || ''],
    weather_turns: battle.field.weatherState?.duration || 0,
    turn: battle.turn,
    result: result(battle),
    last_used: moveId(battle.lastMove?.id),
  };
  const sides = battle.sides.map((side) => ({ ...sideWords(side, orders), ...slotWords(side, orders) }));
  for (const key of Object.keys(sides[0])) out[key] = [sides[0][key], sides[1][key]];

  const parties = battle.sides.map((side, i) => {
    const rows = Array(6).fill(null);
    for (const [mon, at] of orders[i]) rows[at] = monWords(mon, side.pokemon.indexOf(mon));
    return rows;
  });
  for (const key of Object.keys(parties[0][0])) {
    // An empty party slot is zero in the shape the field declares, not a bare 0.
    const blank = Array.isArray(parties[0][0][key]) ? parties[0][0][key].map(() => 0) : 0;
    out[key] = parties.map((rows) => rows.map((row) => (row ? row[key] : blank)));
  }
  // A layout field this file forgot would hash as zero on one side only and
  // show up as a mismatch nobody could explain, so it fails here instead.
  const missing = EXPORTED.filter((k) => !(k in out));
  const extra = Object.keys(out).filter((k) => !EXPORTED.includes(k));
  if (missing.length || extra.length) {
    throw new Error(`export_state.js is out of step with layout.json: missing [${missing}], extra [${extra}]`);
  }
  return out;
}

// The actions the server would accept, as the engine's bitmask: bits 0-3 the
// move slots, 4-9 the party in its stable order, 10 pass, 11 the one action a
// lock leaves. This is Showdown's legality, and the harness's random play picks
// from it. Only `trapped` stops a switch; the
// client warns on `maybeTrapped` but the server still accepts one. Arena Trap
// and Magnet Pull trap in hiding: the request says only `maybeTrapped`, and
// the server rejects the switch anyway, so the Pokemon is what to ask.
function legalMask(side, orders) {
  const req = side.activeRequest;
  if (!req || req.wait) return 1 << 10;
  const switches = side.pokemon
    .filter((p) => !p.fainted && !p.isActive)
    .reduce((m, p) => m | (1 << (4 + orders[side.n].get(p))), 0);
  if (req.forceSwitch) return switches || (1 << 10);
  const active = side.active[0];
  const entry = req.active?.[0];
  // A locked move leaves one action and no way out: a recharge, the second
  // turn of a charge. Struggle also arrives as a request of one entry, but it
  // is not a lock, so a Pokemon with nothing left to use can still switch.
  if (active.getLockedMove()) return 1 << 11;
  if (entry && entry.moves.length === 1 && active.moveSlots.length > 1 && !entry.moves[0].disabled) {
    return (1 << 11) | (active.trapped ? 0 : switches);
  }
  let moves = 0;
  (entry?.moves || []).forEach((m, i) => {
    if (!m.disabled && active.moveSlots[i]?.pp > 0) moves |= 1 << i;
  });
  if (!moves) moves = 1 << 11;
  return moves | (active.trapped ? 0 : switches);
}

module.exports = { exportState, legalMask, partyOrders, carryReveals };
