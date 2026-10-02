// The page's MCTS (web/js/mcts) under Node, on the bundled simulator:
//   node showdown/check_mcts.js time [descents]    one decision's cost from a mid-game position
//   node showdown/check_mcts.js leak               hidden fields cannot change the search
//   node showdown/check_mcts.js match N [descents] [seed] [mcts|ppo]   p2 (MCTS, or PPO as the
//     paired baseline) against PPO (T 0.25) as p1, N games from battle `seed` on
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEB = path.resolve(__dirname, '../web');
vm.runInThisContext(fs.readFileSync(path.join(WEB, 'dist/showdown.js'), 'utf8'));
const { Battle, Teams, State } = globalThis.Showdown;
const { InfoState } = require('../web/js/infostate.js');
const { Vocab, observe, legalMask } = require('../web/js/observation.js');
const { loadPolicy, sample } = require('../web/js/policy.js');
const { command } = require('../web/js/command.js');
const { SetTable } = require('../web/js/mcts/world.js');
const { Search, decide, publicView } = require('../web/js/mcts/search.js');

const read = (f) => JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));
const vocab = new Vocab(read('data/vocab.json'));
const table = new SetTable(read('data/setdist.json'), vocab);
const policy = loadPolicy(read(`models/${read('models/index.json')[0]}.json`), vocab.L);  // the page's network
const seedString = (n) => `sodium,${n.toString(16).padStart(64, '0')}`;

// A battle and each side's InfoState, fed as the page feeds them.
function newGame(n) {
  const gen = Teams.getGenerator('gen3randombattle', seedString(n));
  const battle = new Battle({ formatid: 'gen3randombattle', seed: seedString(n) });
  battle.setPlayer('p1', { name: 'p1', team: gen.getTeam() });
  battle.setPlayer('p2', { name: 'p2', team: gen.getTeam() });
  const states = ['p1', 'p2'].map((p) => new InfoState(p, vocab.speciesTypes));
  const sync = () => states.forEach((s, p) => { s.feed(battle.log, battle.log.length); s.takeRequest(battle.sides[p].activeRequest); });
  sync();
  return { battle, states, sync };
}

function play(g, p, code) {
  const side = g.battle.sides[p], req = side.activeRequest;
  if (!req || req.wait) return;
  const c = command(code, g.states[p], req);
  if (c && !side.choose(c)) side.autoChoose();
}

function ppo(g, p) {
  const { logits } = policy(observe(g.states[p], vocab));
  return sample(logits, 0.25);
}

function root(g, me) {
  const json = State.serializeBattle(g.battle);
  json.log = []; json.inputLog = [];
  const views = [0, 1].map((p) => (p === me ? g.states[p] : publicView(g.battle.log, p === 0 ? 'p1' : 'p2', vocab.speciesTypes)));
  return { json, me, views };
}

const needs = (g, p) => { const r = g.battle.sides[p].activeRequest; return !g.battle.ended && r && !r.wait; };

function advance(g, turns) {
  for (let t = 0; t < turns && !g.battle.ended; t++) {
    for (let p = 0; p < 2; p++) if (needs(g, p)) play(g, p, ppo(g, p));
    g.battle.commitChoices();
    g.sync();
  }
}

const mode = process.argv[2] || 'time';
const search = new Search(table, vocab, policy);

if (mode === 'time') {
  const descents = +(process.argv[3] || 512);
  const g = newGame(48);
  advance(g, 6);
  const t0 = Date.now();
  const r0 = root(g, 1), before = JSON.stringify(r0.json);
  const stats = search.run(r0, { descents, seed: 7 });
  console.log('root untouched by the search:', JSON.stringify(r0.json) === before);
  const d = decide([stats], 1, legalMask(g.states[1]), { final: 'mix', temperature: 0.25 });
  console.log(`turn ${g.battle.turn}: ${stats.descents} descents in ${Date.now() - t0} ms (${((Date.now() - t0) / stats.descents).toFixed(1)} ms each), `
    + `${search.count} nodes; plays ${d.action}, root value ${d.value.toFixed(3)}, dist ${d.dist.map((x) => x.toFixed(2)).join(' ')}`);
} else if (mode === 'leak') {
  // Two true battles p2 cannot tell apart: p1's unseen Pokemon swapped for another species,
  // and a seen one's unrevealed item and moves changed. Same seed, so the search must agree.
  let checked = 0;
  for (let n = 1; checked < 5 && n < 60; n++) {
    const g = newGame(n);
    advance(g, 4 + (n % 5));
    if (g.battle.ended) continue;
    const a = root(g, 1);
    const b = { ...a, json: JSON.parse(JSON.stringify(a.json)) };
    const tb = State.deserializeBattle(b.json);
    const known = g.states[1].sides.p1.mons;
    const unseen = tb.sides[0].pokemon.findIndex((q) => !known.get(q.name)?.species);
    const seen = tb.sides[0].pokemon.find((q) => known.get(q.name)?.species && !known.get(q.name).item_known && !q.fainted);
    if (unseen < 0 || !seen) continue;
    const q = tb.sides[0].pokemon[unseen];
    q.moveSlots = q.baseMoveSlots = q.baseMoveSlots.slice(0, 2);
    q.storedStats.spe += 40;
    seen.item = seen.item === 'leftovers' ? 'choiceband' : 'leftovers';
    const hidden = seen.baseMoveSlots.filter((s) => !known.get(seen.name).moves.has(s.id));
    for (const s of hidden) s.pp = 1;
    seen.storedStats.atk += 17;
    b.json = State.serializeBattle(tb);
    b.json.log = []; b.json.inputLog = [];
    const sa = search.run(a, { descents: 48, seed: 11 });
    const sb = search.run(b, { descents: 48, seed: 11 });
    const same = sa.cellN.every((x, i) => x === sb.cellN[i]) && sa.cellW.every((x, i) => Math.abs(x - sb.cellW[i]) < 1e-9);
    console.log(`battle ${n} turn ${g.battle.turn}: ${same ? 'same' : 'DIFFERENT'} (changed ${q.name} and ${seen.name}'s item and ${hidden.length} hidden moves)`);
    if (!same) process.exitCode = 1;
    checked++;
  }
} else if (mode === 'match') {
  const games = +(process.argv[3] || 4), descents = +(process.argv[4] || 256), seed = +(process.argv[5] || 1000);
  const who = process.argv[6] || 'mcts';
  let wins = 0, ties = 0, decisions = 0, ms = 0;
  for (let i = 0; i < games; i++) {
    const g = newGame(seed + i);
    for (let t = 0; t < 400 && !g.battle.ended; t++) {
      if (needs(g, 0)) play(g, 0, ppo(g, 0));
      if (needs(g, 1) && who === 'ppo') play(g, 1, ppo(g, 1));
      else if (needs(g, 1)) {
        const t0 = Date.now();
        const stats = search.run(root(g, 1), { descents, seed: seed * 7919 + t });
        play(g, 1, decide([stats], 1, legalMask(g.states[1]), { final: 'mix', temperature: 0.25 }).action);
        ms += Date.now() - t0;
        decisions++;
      }
      g.battle.commitChoices();
      g.sync();
    }
    if (g.battle.winner === 'p2') wins++;
    else if (!g.battle.winner) ties++;
    console.log(`game ${i + 1}: ${g.battle.winner || 'tie'} in ${g.battle.turn} turns`);
  }
  console.log(`${who === 'ppo' ? 'PPO' : `MCTS-${descents}`} as p2 vs PPO: ${wins} wins, ${ties} ties, ${games - wins - ties} losses of ${games}; ${(ms / Math.max(decisions, 1)).toFixed(0)} ms a decision`);
}
