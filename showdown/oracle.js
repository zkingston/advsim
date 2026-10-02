// JSONL server: one command per stdin line, one JSON object per stdout line,
// each command closed by {"done": true}. Commands: damage (L1 getDamage cases),
// turn (whole battles for the sweep), prng (the mappings), scenario (one L2
// scenario), emerald (the Emerald chooser on hand-built positions), observe and
// forward (the browser page's converter and network, web/js/, for their parity).
// This is the only process Python talks to; advsim/oracle.py is the other end.
'use strict';
const readline = require('readline');
const { Dex, Teams } = require('pokemon-showdown');
const { Battle } = require('pokemon-showdown/dist/sim/battle');
const { PRNG } = require('pokemon-showdown/dist/sim/prng');
const { exportState, legalMask, partyOrders } = require('./lib/export_state.js');
const { instrument, seedString, debugSorts } = require('./lib/scripted_rng.js');
const emerald = require('./lib/policy.js');
const play = require('./lib/play.js');
const { runScenario, touched } = require('./lib/scenario.js');
const { perft } = require('./lib/perft.js');
const web = require('./lib/web.js');

const gen3 = Dex.mod('gen3');
const FORMAT = 'gen3customgame';
const WEATHERS = ['', 'sunnyday', 'raindance', 'sandstorm'];
// A raw u32 that makes random(16) return exactly r.
const rollSeed = (r) => r * 0x10000000;

// L1: getDamage on random pairs from the pool, with their own abilities and
// items, random boosts, statuses, HP, weather, crit and roll. Each case carries
// the words the hit reads, through the same exporter L2 and L3 use, so the
// engine computes it from state through the path a move takes.
const L1_FIELDS = ['species', 'level', 'ability', 'item', 'hp_type', 'stats', 'hp', 'maxhp', 'status',
  'types', 'boosts', 'active', 'weather', 'vflags', 'moves'];
function damageCases(n, seed) {
  const prng = new PRNG(seedString(seed));
  const pick = (a) => a[prng.random(a.length)];
  const pool = Teams.getGenerator('gen3randombattle', seedString(seed + 1));
  const out = [];
  while (out.length < n) {
    const battle = new Battle({ formatid: FORMAT, seed: seedString(seed + 2) });
    battle.setPlayer('p1', { team: pool.getTeam() });
    battle.setPlayer('p2', { team: pool.getTeam() });
    const orders = partyOrders(battle);
    const source = battle.p1.active[0];
    const target = battle.p2.active[0];
    for (let k = 0; k < 8 && out.length < n; k++) {
      const slot = prng.random(source.moveSlots.length);
      const move = battle.dex.getActiveMove(gen3.moves.get(source.moveSlots[slot].id));
      if (!move.basePower || move.category === 'Status' || move.damage || move.ohko) continue;
      if (move.basePowerCallback || move.onModifyMove || move.onBasePower) continue;
      const crit = prng.random(2) === 0;
      const roll = prng.random(16);
      const weather = pick(WEATHERS);
      for (const [p, stats] of [[source, ['atk', 'spa']], [target, ['def', 'spd']]]) {
        for (const stat of stats) p.boosts[stat] = prng.random(-6, 7);
      }
      source.status = pick(['', '', 'brn', 'par', 'psn']);
      target.status = pick(['', '', '', 'par', 'psn']);
      source.hp = 1 + prng.random(source.maxhp);  // Blaze and the other pinch abilities
      battle.field.weather = weather;
      battle.field.weatherState = weather ? { id: weather, duration: 5 } : { id: '' };
      move.willCrit = crit;
      battle.prng.rng.next = () => rollSeed(roll);  // the only draw left in getDamage is the roll
      const damage = battle.actions.getDamage(source, target, move, true);
      const words = exportState(battle, orders);
      out.push({
        slot, crit, roll, damage: damage === false ? -1 : (damage === undefined ? -2 : damage),
        words: Object.fromEntries(L1_FIELDS.map((f) => [f, words[f]])),
      });
    }
  }
  return out;
}

// Whole battles for the sweep: the state before, then per decision point the
// choices, every raw draw with its call site, the state after and the legal
// masks. `policy` names how both sides play (lib/play.js); 'mix' draws one
// per side per battle. None of them touches the battle's PRNG.
function turnCases(n, seed, turns, policy, protocol) {
  const prng = new PRNG(seedString(seed));
  const pool = Teams.getGenerator('gen3randombattle', seedString(seed + 1));
  const withGender = (team) => team.map((p) => ({ ...p, gender: p.gender || Dex.species.get(p.species).gender || 'M' }));
  const out = [];
  for (let attempt = 1; out.length < n; attempt++) {
    const battle = new Battle({ formatid: 'gen3randombattle', seed: seedString(seed + attempt) });
    battle.setPlayer('p1', { team: withGender(pool.getTeam()) });
    battle.setPlayer('p2', { team: withGender(pool.getTeam()) });
    debugSorts(battle, `case=${out.length}.${attempt}`);
    const orders = partyOrders(battle);
    const before = exportState(battle, orders);
    const draws = [], sites = [];
    instrument(battle, null, (v, key) => { draws.push(v); sites.push(key); }, __filename);
    const policies = play.policiesFor(policy, prng);
    const ctx = { battle, prng, orders };


    const segments = [];
    const seen = new Set();
    // With `protocol`, what each player's client was given at every decision:
    // the log position and the request JSON, for the live converter's parity.
    const view = () => ({ cursor: battle.log.length,
      requests: battle.sides.map((side) => JSON.parse(JSON.stringify(side.activeRequest || null))) });
    const start = protocol ? view() : undefined;
    while (segments.length < turns && !battle.ended) {
      const choices = battle.sides.map((side, i) => play.pick(policies[i], ctx, side));
      draws.length = 0; sites.length = 0;
      battle.sides.forEach((side, i) => { const c = play.command(side, choices[i], orders); if (c) side.choose(c); });
      battle.commitChoices();
      touched(battle, seen, false);
      segments.push({ after: exportState(battle, orders), legal: battle.sides.map((side) => legalMask(side, orders)),
                      choices, draws: [...draws], sites: [...sites], ...(protocol ? { view: view() } : {}) });
    }
    const c = { before, segments, policies, touched: [...touched(battle, seen)] };
    if (process.env.ADVSIM_LOG || protocol) c.log = battle.log;
    if (protocol) c.start = start;
    out.push(c);
  }
  return out;
}


// A perft position: one battle from `seed`, played `turns` decisions with the
// policy mix and then on to the first turn where both sides face an ordinary
// move request, which is where the engine's own bookkeeping is zero and the
// exported words are the whole state. Emits nothing if the battle ends first.
function perftCases(seed, turns, depth, budget, stride, emit) {
  const prng = new PRNG(seedString(seed));
  const pool = Teams.getGenerator('gen3randombattle', seedString(seed + 1));
  const withGender = (team) => team.map((p) => ({ ...p, gender: p.gender || Dex.species.get(p.species).gender || 'M' }));
  const battle = new Battle({ formatid: 'gen3randombattle', seed: seedString(seed + 2) });
  battle.setPlayer('p1', { team: withGender(pool.getTeam()) });
  battle.setPlayer('p2', { team: withGender(pool.getTeam()) });
  const orders = partyOrders(battle);
  const policies = play.policiesFor('mix', prng);
  const ctx = { battle, prng, orders };
  const ordinary = () => battle.sides.every((side) => side.activeRequest?.active && !side.activeRequest.forceSwitch);
  for (let t = 0; !battle.ended && (t < turns || !ordinary()); t++) {
    const choices = battle.sides.map((side, i) => play.pick(policies[i], ctx, side));
    battle.sides.forEach((side, i) => { const c = play.command(side, choices[i], orders); if (c) side.choose(c); });
    battle.commitChoices();
  }
  if (battle.ended) return 0;
  return perft(battle, orders, depth, budget, stride, emit);
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (!line.trim()) return;
  let req;
  try { req = JSON.parse(line); } catch (e) { console.log(JSON.stringify({ error: `bad request: ${e.message}` })); return; }
  try {
    if (req.cmd === 'damage') {
      for (const c of damageCases(req.n || 100, req.seed || 1)) console.log(JSON.stringify(c));
      console.log(JSON.stringify({ done: true, cmd: 'damage', n: req.n }));
    } else if (req.cmd === 'turn') {
      for (const c of turnCases(req.n || 50, req.seed || 1, req.turns || 1, req.policy || 'random', !!req.protocol)) console.log(JSON.stringify(c));
      console.log(JSON.stringify({ done: true, cmd: 'turn' }));
    } else if (req.cmd === 'prng') {
      const prng = new PRNG(seedString(req.seed || 1));
      for (let i = 0; i < (req.n || 100); i++) {
        // Draw the raw u32 first, then reseed so each mapping sees that same raw value.
        const before = prng.getSeed();
        const raw = prng.rng.next();
        const n = 2 + (raw % 97), m = raw % 7, hi = m + 1 + (raw % 23);
        const num = 1 + (raw % 5), den = num + (raw % 11);
        prng.setSeed(before); const a = prng.random(n);
        prng.setSeed(before); const b = prng.random(m, hi);
        prng.setSeed(before); const c = prng.randomChance(num, den);
        prng.setSeed(before); prng.rng.next();
        console.log(JSON.stringify({ raw, n, random_n: a, m, hi, random_range: b, num, den, chance: c }));
      }
      console.log(JSON.stringify({ done: true, cmd: 'prng' }));
    } else if (req.cmd === 'perft') {
      const depth1 = perftCases(req.seed, req.turns || 10, req.depth || 1, req.budget || 100000, req.stride || 1,
        (c) => console.log(JSON.stringify(c)));
      console.log(JSON.stringify({ depth1 }));
      console.log(JSON.stringify({ done: true, cmd: 'perft' }));
    } else if (req.cmd === 'scenario') {
      console.log(JSON.stringify(runScenario(req.scenario)));
      console.log(JSON.stringify({ done: true, cmd: 'scenario' }));
    } else if (req.cmd === 'emerald') {
      // The chooser's two sections on hand-built positions, for the write-up's examples.
      const types = (name) => gen3.species.get(name).types;
      for (const c of req.cases) {
        const foe = { types: c.foe.types || types(c.foe.species), ability: c.foe.ability || '' };
        const party = c.party.map((m) => m && { types: types(m.species), moves: m.moves.map(emerald.moveInfo) });
        console.log(JSON.stringify({ section1: emerald.section1(foe, party),
          section2: emerald.section2(foe, party, c.base || 0, c.stab || []) }));
      }
      console.log(JSON.stringify({ mismatches: emerald.tableMismatches() }));
      console.log(JSON.stringify({ done: true, cmd: 'emerald' }));
    } else if (req.cmd === 'observe') {
      console.log(JSON.stringify({ obs: web.observeCase(req.case) }));
      console.log(JSON.stringify({ done: true, cmd: 'observe' }));
    } else if (req.cmd === 'forward') {
      console.log(JSON.stringify(web.forward(req.model, req.obs)));
      console.log(JSON.stringify({ done: true, cmd: 'forward' }));
    } else {
      console.log(JSON.stringify({ error: `unknown command ${req.cmd}` }));
    }
  } catch (e) {
    console.log(JSON.stringify({ error: e.message, stack: (e.stack || '').split('\n')[1] }));
  }
});
