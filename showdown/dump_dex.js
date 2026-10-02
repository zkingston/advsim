// Dump resolved Gen 3 dex data + callback sources from pokemon-showdown 0.11.11.
// Usage: node showdown/dump_dex.js  -> artifacts/dex_raw.json
'use strict';
const fs = require('fs');
const path = require('path');
const { Dex, Teams } = require('pokemon-showdown');

const OUT = path.join(__dirname, '..', 'artifacts', 'dex_raw.json');
const FORMAT = 'gen3randombattle';
const DROP = new Set(['desc', 'shortDesc', 'fullname', 'exists', 'effectType', 'isNonstandard', 'contestType',
  'pokemonGoData', 'tier', 'doublesTier', 'natDexTier', 'spriteid', 'color', 'tags', 'eggGroups', 'canHatch',
  'realMove', 'inherit', 'sourceEffect']);
const EFFECT_KEYS = new Set(['condition', 'secondary', 'secondaries', 'self', 'selfBoost', 'fling']);

const hasFn = (o) => o && typeof o === 'object' && Object.values(o).some((v) => typeof v === 'function');

// Split an effect-like object into data fields + callbacks: {event: source}.
// Recurses only into known effect-like sub-objects; any other object holding a function throws,
// so a callback can never be dropped silently by JSON.stringify.
function split(o, where) {
  const out = {}, callbacks = {};
  for (const [k, v] of Object.entries(o)) {
    if (v === undefined || DROP.has(k)) continue;
    if (typeof v === 'function') callbacks[k] = v.toString();
    else if (EFFECT_KEYS.has(k) && v && typeof v === 'object') {
      out[k] = Array.isArray(v) ? v.map((x, i) => split(x, `${where}.${k}[${i}]`)) : split(v, `${where}.${k}`);
    } else if (hasFn(v) || (Array.isArray(v) && v.some(hasFn))) {
      throw new Error(`unexpected callback under ${where}.${k}`);
    } else out[k] = v;
  }
  out.callbacks = callbacks;
  return out;
}

// Scripts: functions and objects of functions, arbitrary nesting.
function srcTree(o, where) {
  const out = {};
  for (const [k, v] of Object.entries(o)) {
    if (DROP.has(k)) continue;
    if (typeof v === 'function') out[k] = v.toString();
    else if (v && typeof v === 'object' && !Array.isArray(v)) out[k] = srcTree(v, `${where}.${k}`);
    else out[k] = v;
  }
  return out;
}

// Every dumped callback must still parse; a truncated or mangled source would silently break the M1 build.
function checkParses(v, where, out) {
  if (Array.isArray(v)) { v.forEach((x, i) => checkParses(x, `${where}[${i}]`, out)); return out; }
  if (v && typeof v === 'object') { for (const [k, x] of Object.entries(v)) checkParses(x, `${where}.${k}`, out); return out; }
  if (typeof v !== 'string' || !/^(\w+\s*\(|function\b|\(|async\b)/.test(v) || !v.includes('{')) return out;
  try { new Function(`return ({${v}})`); } catch (e) { try { new Function(`return (${v})`); } catch (e2) { out.push(where); } }
  return out;
}

function sortKeys(v) {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === 'object') return Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])]));
  return v;
}

function table(entries, where) {
  const out = {};
  for (const e of entries) out[Dex.toID(e.name)] = split(e, `${where}.${e.name}`);  // toID(name): 17 hiddenpower* keys share id 'hiddenpower'
  return out;
}

// The random-battle vocabulary, enumerated rather than sampled. The set table
// lists every species, level, movepool and ability the generator can produce;
// items are not data, so they are read off getItem's own source, keeping every
// quoted string that resolves to a real item.
function randbats(dex) {
  const gen = Teams.getGenerator(FORMAT, null);
  const sets = gen.randomSets;
  const source = gen.getItem.toString();
  const quoted = [...new Set((source.match(/"[^"]+"/g) || []).map((q) => q.slice(1, -1)))];
  const items = quoted.filter((name) => dex.items.get(name).exists).map((name) => Dex.toID(name)).sort();
  const species = new Set(), moves = new Set(), abilities = new Set(), levels = {};
  for (const [id, entry] of Object.entries(sets)) {
    levels[id] = entry.level;
    species.add(id);
    for (const n of dex.species.get(id).cosmeticFormes || []) species.add(Dex.toID(n));
    for (const set of entry.sets) {
      set.movepool.forEach((m) => moves.add(m));
      (set.abilities || []).forEach((a) => abilities.add(Dex.toID(a)));
    }
  }
  return {
    generator: gen.constructor.name,
    species: [...species].sort(), moves: [...moves].sort(), abilities: [...abilities].sort(), items,
    levels, sets, getItemSource: source,
  };
}

// species.all() omits cosmetic formes (the 27 Unown letters); the team pool uses their ids, so expand them.
function speciesList(dex, std) {
  const base = dex.species.all().filter((s) => std(s) && s.num >= 1);
  return [...base, ...base.flatMap((s) => (s.cosmeticFormes || []).map((n) => dex.species.get(n)))];
}

function main() {
  const dex = Dex.mod('gen3');
  const std = (e) => e.gen <= 3 && !e.isNonstandard;
  const chain = [];
  for (let m = dex; m; m = m.parentMod ? Dex.mod(m.parentMod) : null) { chain.push(m.currentMod); if (m.currentMod === 'base') break; }
  const format = Dex.formats.get(FORMAT);

  const raw = {
    meta: { showdown: require('pokemon-showdown/package.json').version, mod: 'gen3', modChain: chain, generated: new Date().toISOString() },
    format: { id: format.id, mod: format.mod, team: format.team, ruleset: format.ruleset, rules: [...Dex.formats.getRuleTable(format).keys()] },
    species: table(speciesList(dex, std), 'species'),
    moves: table(dex.moves.all().filter(std), 'moves'),
    abilities: table(dex.abilities.all().filter(std), 'abilities'),
    items: table(dex.items.all().filter(std), 'items'),
    conditions: table(Object.keys(dex.data.Conditions).map((id) => dex.conditions.get(id)), 'conditions'),
    typechart: dex.data.TypeChart,
    scripts: srcTree(dex.data.Scripts, 'scripts'),
    randbats: randbats(dex),
  };

  const bad = checkParses([raw.scripts, ...['moves', 'abilities', 'items', 'conditions'].map((t) => raw[t])], 'dex', []);
  if (bad.length) throw new Error(`${bad.length} callbacks failed to parse, e.g. ${bad.slice(0, 3).join(', ')}`);

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(sortKeys(raw), null, 2) + '\n');

  const n = (t) => Object.keys(raw[t]).length;
  const cb = (t) => Object.values(raw[t]).filter((e) => Object.keys(e.callbacks).length).length;
  const nested = Object.values(raw.moves).filter((m) => m.condition && Object.keys(m.condition.callbacks).length).length;
  const sec = Object.values(raw.moves).filter((m) => (m.secondaries || []).some((s) => Object.keys(s.callbacks).length)).length;
  console.log(JSON.stringify({
    out: path.relative(process.cwd(), OUT), bytes: fs.statSync(OUT).size,
    species: n('species'), moves: n('moves'), abilities: n('abilities'), items: n('items'), conditions: n('conditions'),
    withCallbacks: { moves: cb('moves'), abilities: cb('abilities'), items: cb('items'), conditions: cb('conditions'), moveConditions: nested, moveSecondaries: sec },
    scripts: Object.keys(raw.scripts),
    randbats: {species: raw.randbats.species.length, moves: raw.randbats.moves.length,
               abilities: raw.randbats.abilities.length, items: raw.randbats.items.length,
               sets: Object.values(raw.randbats.sets).reduce((n, e) => n + e.sets.length, 0)},
  }));
}
main();
