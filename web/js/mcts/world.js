// A world for the search: the battle as one side sees it, everything hidden from it drawn
// afresh. A port of advsim/engine/determinize.py onto Showdown's own objects:
//
// - The copy gets a fresh PRNG seed: the real one would tell the search every later roll.
// - An unseen foe Pokemon becomes a fresh species (Species Clause) with a set from the set
//   table. A seen one keeps what it revealed (species, level, status, the moves it used, a
//   shown ability or item) and takes the rest of a set that contains it, and an exact HP
//   showing the same percent.
// - A Choice lock the true item implies goes with it; a Substitute takes the new max HP.
// - Hidden durations on both sides (a foe's sleep, confusion, a partial trap) are drawn again.
//
// Positive evidence only, as in the engine: a set must contain what was revealed, and
// nothing is ruled out by what failed to appear. Simplifications against the engine: the
// true party order is kept (switching does not depend on it, and the PRNG is fresh), a
// transformed foe keeps its copy untouched, and Encore keeps its duration.
import { hpPercent } from '../observation.js';

// mulberry32: the search's own stream, apart from every battle's.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const below = (r, n) => Math.floor(r() * n);
const hex = (r) => Array.from({ length: 8 }, () => (below(r, 2 ** 32) >>> 0).toString(16).padStart(8, '0')).join('');

// web/data/setdist.json: the pool's distinct sets by species, with counts (build/setdist.py).
export class SetTable {
  constructor(sd, vocab) {
    this.sd = sd;
    this.v = vocab.v;
    this.col = Object.fromEntries(sd.columns.map((c, i) => [c, i]));
    this.speciesId = vocab.index.species;
  }

  // A set of species `sp` weighted by count among those containing the revealed moves and
  // ability; the ability is let go first, then the moves, if none does.
  pickSet(sp, revealed, r) {
    const { sets, weight, start, count } = this.sd, c = this.col;
    const fits = (row, level) => {
      if (level === 0 || !revealed) return true;
      if (level === 2 && revealed.ability && this.v.abilities[row[c.ability]] !== revealed.ability) return false;
      const moves = [0, 1, 2, 3].map((k) => this.v.moves[row[c[`move${k + 1}`]]]);
      return [...revealed.moves].every((m) => moves.includes(m));
    };
    for (const level of [2, 1, 0]) {
      let total = 0;
      for (let i = start[sp]; i < start[sp] + count[sp]; i++) if (fits(sets[i], level)) total += weight[i];
      if (!total) continue;
      let pick = below(r, total);
      for (let i = start[sp]; i < start[sp] + count[sp]; i++) {
        if (fits(sets[i], level) && (pick -= weight[i]) < 0) return this.row(i);
      }
    }
    return this.row(start[sp] + count[sp] - 1);
  }

  row(i) {
    const x = this.sd.sets[i], c = this.col, v = this.v;
    return {
      species: v.species[x[c.species]], level: x[c.level], ability: v.abilities[x[c.ability]], item: v.items[x[c.item]],
      hpType: v.types[x[c.hp_type]], maxhp: x[c.maxhp],
      stats: { atk: x[c.atk], def: x[c.def], spa: x[c.spa], spd: x[c.spd], spe: x[c.spe] },
      moves: [1, 2, 3, 4].map((k) => v.moves[x[c[`move${k}`]]]).filter(Boolean),
      pp: [1, 2, 3, 4].map((k) => x[c[`pp${k}`]]),
    };
  }

  // A species for an unseen Pokemon, weighted by the pool, not one already on the side.
  pickSpecies(taken, r) {
    const w = this.sd.species_weight;
    let total = 0;
    for (let sp = 1; sp < w.length; sp++) if (!taken.has(sp)) total += w[sp];
    let pick = below(r, total);
    for (let sp = 1; sp < w.length; sp++) if (!taken.has(sp) && (pick -= w[sp]) < 0) return sp;
    return w.length - 1;
  }

  pickGender(sp, r) {
    const g = this.sd.gender[sp], total = g.reduce((a, b) => a + b, 0);
    let pick = below(r, total);
    for (let i = 0; i < 4; i++) if ((pick -= g[i]) < 0) return this.sd.genders[i];
    return '';
  }
}

// An HP out of `max` that shows `pct` (Showdown's percent), uniformly; the nearest if none does.
function hpLike(pct, max, r) {
  if (pct <= 0) return 0;
  if (pct >= 100) return max;
  const ok = [];
  for (let h = 1; h < max; h++) if (hpPercent(h, max) === pct) ok.push(h);
  return ok.length ? ok[below(r, ok.length)] : Math.min(Math.max(Math.floor((pct * max) / 100), 1), max - 1);
}

// The set's moves as Showdown move slots; a move the Pokemon revealed keeps its current PP.
function slots(world, row, keepPP) {
  return row.moves.map((id, k) => {
    const move = world.dex.moves.get(id);
    return { move: move.name, id: move.id, pp: keepPP.get(move.id) ?? row.pp[k], maxpp: row.pp[k],
      target: move.target, disabled: false, disabledSource: '', used: false };
  });
}

// Write a set into a Pokemon: moves, stats, max HP, Hidden Power, and the ability and item
// unless `keep` says they were revealed.
function applyRow(world, q, row, keep) {
  const moveSlots = slots(world, row, keep.pp);
  q.set = { ...q.set, moves: row.moves, level: row.level, hpType: row.hpType };
  q.baseMoveSlots = moveSlots;
  q.moveSlots = moveSlots.map((s) => ({ ...s }));
  q.hpType = q.baseHpType = row.hpType;
  q.hpPower = q.baseHpPower = 70;
  if (!keep.item) {
    q.item = row.item;
    q.set.item = row.item;
    q.itemState = world.initEffectState({ id: row.item, target: q });
  }
  if (!keep.ability) {
    q.ability = q.baseAbility = row.ability;
    q.set.ability = row.ability;
    q.abilityState = world.initEffectState({ id: row.ability, target: q });
  }
  q.storedStats = { ...row.stats };
  q.baseStoredStats = { hp: row.maxhp, ...row.stats };
  q.speed = row.stats.spe;
  q.maxhp = q.baseMaxhp = row.maxhp;
}

// `root`: State.serializeBattle of the true battle (log dropped). `me`: the searching side's
// index. `view`: its InfoState, whose knowledge of the foe is all the world may use.
export function buildWorld(root, me, view, table, r) {
  const { State, PRNG } = globalThis.Showdown;
  const world = State.deserializeBattle(root);
  world.log = [];  // deserializing hands over the root's own array; a world needs its own
  world.inputLog = [];
  world.sentLogPos = 0;
  world.prng = new PRNG(`sodium,${hex(r)}`);
  const foe = world.sides[1 - me], known = view.sides[foe.id].mons;
  const Pokemon = foe.pokemon[0].constructor;
  const taken = new Set();
  const unseen = [];
  foe.pokemon.forEach((q, i) => {
    const k = known.get(q.name);
    if (k && k.species) taken.add(table.speciesId.get(q.baseSpecies.id) ?? 0);
    else unseen.push(i);
  });
  foe.pokemon.forEach((q) => {
    const k = known.get(q.name);
    if (!k || !k.species || q.transformed) return;
    const sp = table.speciesId.get(q.baseSpecies.id);
    if (sp == null) return;
    const revealed = { moves: new Set([...k.moves].filter((m) => q.baseMoveSlots.some((s) => s.id === m))),
      ability: k.ability_known ? q.baseAbility : null };
    const row = table.pickSet(sp, revealed, r);
    const pp = new Map(q.baseMoveSlots.filter((s) => revealed.moves.has(s.id)).map((s) => [s.id, s.pp]));
    applyRow(world, q, row, { pp, item: k.item_known, ability: k.ability_known });
    q.hp = q.fainted ? 0 : hpLike(k.hp, row.maxhp, r);
    if (q.volatiles.substitute) q.volatiles.substitute.hp = Math.max(Math.floor(q.maxhp / 4), 1);
    if (q.volatiles.choicelock && q.item !== 'choiceband') delete q.volatiles.choicelock;
  });
  for (const i of unseen) {
    const sp = table.pickSpecies(taken, r);
    taken.add(sp);
    const row = table.pickSet(sp, null, r);
    const name = world.dex.species.get(row.species).name;
    const fresh = new Pokemon({ name, species: name, item: row.item, ability: row.ability, moves: row.moves,
      level: row.level, gender: table.pickGender(sp, r), hpType: row.hpType, evs: {}, ivs: {} }, foe);
    applyRow(world, fresh, row, { pp: new Map() });
    fresh.hp = row.maxhp;
    fresh.position = foe.pokemon[i].position;
    foe.pokemon[i] = fresh;
  }
  for (const side of world.sides) {
    for (const q of side.pokemon) {
      if (q.status === 'slp' && q.statusState.source !== q) q.statusState.time = 1 + below(r, 4);  // not Rest's
      if (q.volatiles.confusion) q.volatiles.confusion.time = 1 + below(r, 4);
      if (q.volatiles.partiallytrapped) q.volatiles.partiallytrapped.duration = 1 + below(r, 5);
    }
  }
  if (world.requestState) world.makeRequest();
  return world;
}
