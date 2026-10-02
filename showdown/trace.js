// Trace Gen 3 Random Battle games: raw RNG draws with call sites, choices, protocol log.
// Usage: node showdown/trace.js --games 1000 --policy random|greedy --seed 1 [--verify 50]
//   -> traces/<policy>-<seed>.jsonl (one battle per line), artifacts/catalog.json (site key -> fn, args, counts)
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { Dex, Teams } = require('pokemon-showdown');
const { Battle } = require('pokemon-showdown/dist/sim/battle');
const { PRNG } = require('pokemon-showdown/dist/sim/prng');

const FORMAT = 'gen3randombattle';
const ROOT = path.join(__dirname, '..');
const CATALOG = path.join(ROOT, 'artifacts', 'catalog.json');
const { instrument, siteName } = require('./lib/scripted_rng.js');
const gen3 = Dex.mod('gen3');
const seedFor = (seed, g, tag) => 'sodium,' + crypto.createHash('sha256').update(`${seed}:${g}:${tag}`).digest('hex');

// The PRNG patch, the site keys and the `force` names live in lib/scripted_rng.js,
// because the scenario runner has to name a call site exactly the way a trace does.

function choose(side, policy, pol) {
  const r = side.activeRequest; if (!r || r.wait) return null;
  const bench = () => side.pokemon.map((p, i) => (!p.fainted && !p.isActive) ? `switch ${i + 1}` : null).filter(Boolean);
  if (r.forceSwitch) { const o = bench(); return o.length ? pol.sample(o) : 'pass'; }
  const a = r.active[0];
  const mv = a.moves.map((m, i) => [m, i]).filter(([m]) => !m.disabled);
  const sw = (!a.trapped && !a.maybeTrapped) ? bench() : [];
  if (policy === 'greedy' && mv.length) {
    let best = mv[0], bp = -1;
    for (const [m, i] of mv) { const v = (gen3.moves.get(m.id).basePower || 0) + pol.random(); if (v > bp) { bp = v; best = [m, i]; } }
    if (pol.random() < 0.9) return `move ${best[1] + 1}`;
  }
  const o = [...mv.map(([, i]) => `move ${i + 1}`), ...sw];
  return o.length ? pol.sample(o) : 'move 1';
}

// Play (or replay from `script` draws + `choicesIn`) one battle. Returns the battle plus what was logged.
function run(teams, seed, policy, pol, script, choicesIn) {
  const b = new Battle({ formatid: FORMAT, seed });
  const draws = [], sites = [], siteKeys = [], keyIdx = new Map(), calls = [];
  instrument(b, script, script ? null : (v, key, call) => {
    if (!keyIdx.has(key)) { keyIdx.set(key, siteKeys.length); siteKeys.push(key); }
    draws.push(v); sites.push(keyIdx.get(key)); calls.push(call);
  });
  b.setPlayer('p1', { team: teams[0] }); b.setPlayer('p2', { team: teams[1] });
  const choices = []; let ci = 0, guard = 0;
  while (!b.ended && guard++ < 3000) {
    for (const s of b.sides) {
      let c;
      if (choicesIn) { if (ci < choicesIn.length && choicesIn[ci][0] === s.id && s.activeRequest && !s.activeRequest.wait) c = choicesIn[ci++][1]; }
      else c = choose(s, policy, pol);
      if (!c) continue;
      if (!s.choose(c)) { c = 'default'; s.choose(c); }
      if (!choicesIn) choices.push([s.id, c]);
    }
    if (b.allChoicesDone()) b.commitChoices();
  }
  return { b, draws, sites, siteKeys, calls, choices };
}

const stripLog = (b) => b.log.filter((l) => !l.startsWith('|t:|'));
// Only species that would otherwise make Showdown draw get an explicit gender.
// A genderless species resolves to '' on its own, and forcing 'N' would put a
// gender marker in the log that a plain Showdown run would not have.
const withGender = (team, pol) => team.map((p) => (
  p.gender || Dex.species.get(p.species).gender ? p : { ...p, gender: pol.sample(['M', 'F']) }));
function arg(name, dflt) { const i = process.argv.indexOf('--' + name); return i < 0 ? dflt : process.argv[i + 1]; }

function main() {
  const games = +arg('games', 100), policy = arg('policy', 'random'), seed = +arg('seed', 1);
  const verify = +arg('verify', Math.min(games, 50));
  const out = path.join(ROOT, 'traces', `${policy}-${seed}.jsonl`);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  const catalog = fs.existsSync(CATALOG) ? JSON.parse(fs.readFileSync(CATALOG, 'utf8')) : {};
  for (const e of Object.values(catalog)) delete e.count[policy];
  const fd = fs.openSync(out, 'w'), t0 = Date.now();
  let verified = 0, setupDraws = 0, quickclawMissing = 0;
  for (let g = 0; g < games; g++) {
    const pol = new PRNG(seedFor(seed, g, 'policy'));
    const gen = Teams.getGenerator(FORMAT, seedFor(seed, g, 'teams'));
    const teams = [withGender(gen.getTeam(), pol), withGender(gen.getTeam(), pol)];
    const battleSeed = seedFor(seed, g, 'battle');
    const { b, draws, sites, siteKeys, calls, choices } = run(teams, battleSeed, policy, pol, null, null);
    const log = stripLog(b);
    let qc = 0;
    sites.forEach((si, i) => {
      const key = siteKeys[si], call = calls[i] || ['?'];
      const e = catalog[key] || (catalog[key] = { name: siteName(key), fn: call[0], args: [], count: {} });
      const a = JSON.stringify(call.slice(1)); if (!e.args.includes(a) && e.args.length < 8) e.args.push(a);
      e.count[policy] = (e.count[policy] || 0) + 1;
      if (key.startsWith('new Pokemon<')) setupDraws++;
      if (key.startsWith('Battle.endTurn<')) qc++;
    });
    if (qc < b.turn - 1) quickclawMissing++;
    if (g < verify) {
      const r2 = run(teams, battleSeed, policy, pol, draws, choices);
      if (stripLog(r2.b).join('\n') === log.join('\n')) verified++; else console.error(`game ${g}: replay from draws diverged`);
    }
    fs.writeSync(fd, JSON.stringify({ g, seed: battleSeed, policySeed: seedFor(seed, g, 'policy'), teams, choices, draws, siteKeys, sites, log, turns: b.turn, ended: b.ended, winner: b.winner || '' }) + '\n');
    if (g % 100 === 99) console.error(`${g + 1}/${games} games, ${(g + 1) / ((Date.now() - t0) / 1000) | 0}/s`);
  }
  fs.closeSync(fd);
  const sorted = Object.fromEntries(Object.keys(catalog).sort().map((k) => [k, catalog[k]]));
  fs.writeFileSync(CATALOG, JSON.stringify(sorted, null, 2) + '\n');
  if (setupDraws) throw new Error(`${setupDraws} setup gender draws; teams must carry explicit gender`);
  if (quickclawMissing) throw new Error(`${quickclawMissing} games without a Quick Claw roll every turn`);
  if (verified !== verify) throw new Error(`replay verified ${verified}/${verify}`);
  console.log(JSON.stringify({ out: path.relative(process.cwd(), out), games, policy, seed, verified, secs: +((Date.now() - t0) / 1000).toFixed(1), sites: Object.keys(sorted).length, catalog: path.relative(process.cwd(), CATALOG) }));
}
main();
