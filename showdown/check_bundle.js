// The browser bundle against the package: the same teams from the same seed,
// then the same battle log, choice for choice, with random play.
//   node showdown/check_bundle.js [battles] [seed]
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const native = require('pokemon-showdown');
const { Battle } = require('pokemon-showdown/dist/sim/battle');
const { PRNG } = require('pokemon-showdown/dist/sim/prng');
const { seedString } = require('./lib/scripted_rng.js');

// No require, process or fs: what a browser page has.
const sandbox = { console, Date, Math, JSON };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../web/dist/showdown.js'), 'utf8'), sandbox);
const web = sandbox.Showdown;

const clean = (log) => log.filter((l) => !l.startsWith('|t:|'));

// A random legal choice from the request alone (no artifacts, so CI can run
// this); a switch refused for a hidden trap is chosen again from the updated request.
function choose(side, prng) {
  for (let tries = 0; tries < 5; tries++) {
    const r = side.activeRequest;
    if (!r || r.wait) return null;
    const bench = r.side.pokemon.map((p, i) => i + 1).filter((i) => !r.side.pokemon[i - 1].active && !r.side.pokemon[i - 1].condition.endsWith(' fnt'));
    const opts = r.forceSwitch ? bench.map((i) => `switch ${i}`)
      : [...r.active[0].moves.map((m, i) => (m.disabled ? null : `move ${i + 1}`)).filter(Boolean),
        ...(r.active[0].trapped ? [] : bench.map((i) => `switch ${i}`))];
    const c = opts.length ? opts[prng.random(opts.length)] : (r.forceSwitch ? 'pass' : 'move 1');
    if (side.choose(c)) return c;
  }
  throw new Error(`no legal choice for ${side.id}`);
}

const n = +(process.argv[2] || 100), seed = +(process.argv[3] || 1);
let decisions = 0;
for (let b = 0; b < n; b++) {
  const s = seedString(seed + b);
  const gen = [native.Teams, web.Teams].map((T) => T.getGenerator('gen3randombattle', s));
  const teams = gen.map((g) => [g.getTeam(), g.getTeam()]);
  if (JSON.stringify(teams[0]) !== JSON.stringify(teams[1])) throw new Error(`battle ${b}: teams differ`);
  const battles = [new Battle({ formatid: 'gen3randombattle', seed: s }), new web.Battle({ formatid: 'gen3randombattle', seed: s })];
  battles.forEach((bt, k) => { bt.setPlayer('p1', { team: teams[k][0] }); bt.setPlayer('p2', { team: teams[k][1] }); });
  const prng = new PRNG(s);
  for (let t = 0; t < 300 && !battles[0].ended; t++, decisions++) {
    const picks = battles[0].sides.map((side) => choose(side, prng));
    battles[0].commitChoices();
    battles[1].sides.forEach((side, i) => { if (picks[i] && !side.choose(picks[i])) throw new Error(`bundle refused ${picks[i]}`); });
    battles[1].commitChoices();
    const [a, c] = battles.map((bt) => clean(bt.log));
    if (JSON.stringify(a) !== JSON.stringify(c)) {
      const i = a.findIndex((l, j) => l !== c[j]);
      throw new Error(`battle ${b} decision ${t}: logs differ at line ${i}: ${a[i]} vs ${c[i]}`);
    }
  }
}
console.log(`${n} battles, ${decisions} decisions: bundle and package agree`);
