// One L2 scenario: build the position, play the listed turns with steered
// draws, and report what Showdown's state was at every decision point.
//
// The engine is loaded from `before` and replays the same raw draws, so forcing
// an outcome can only change which code path the scenario reaches, never what
// either side computes once it is there.
'use strict';
const { Dex } = require('pokemon-showdown');
const { Battle } = require('pokemon-showdown/dist/sim/battle');
const { exportState, legalMask, partyOrders } = require('./export_state.js');
const { scriptedRNG, seedString, debugSorts } = require('./scripted_rng.js');

// What a scenario may write instead of a bare number, so `force` reads as
// English: {"crit": false, "accuracy": "hit"}.
const FORCE_WORDS = { hit: true, miss: false, yes: true, no: false, min: 0, mid: 8, max: 15 };
const forceValue = (v) => (Array.isArray(v) ? v.map(forceValue) : (typeof v === 'string' ? FORCE_WORDS[v] : v));
const forceMap = (force) => Object.fromEntries(Object.entries(force).map(([k, v]) => [k, forceValue(v)]));

// A scenario set is species, level, ability, item and moves; everything else is
// a Showdown default. Gender is filled the way the pool does, so no set-up draw
// happens before the battle.
function scenarioTeam(sets) {
  return sets.map((s) => ({
    name: s.species, species: s.species, level: s.level || 80, ability: s.ability,
    item: s.item || '', moves: s.moves, nature: s.nature || 'Serious',
    evs: s.evs || { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 },
    ivs: s.ivs || { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
    gender: s.gender || Dex.species.get(s.species).gender || 'M',
    shiny: false, happiness: 255,
  }));
}

// `switch N` counts in the order the scenario lists the team, which never
// moves; Showdown counts in its own array, which a switch reorders.
function command(side, orders, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return { code: 10, cmd: null };
  if (want === 'pass' || want === null) {
    // The side has something to decide and the scenario declined to decide it.
    throw new Error(`${side.id} has a ${req.forceSwitch ? 'replacement' : 'move'} to choose, and the scenario passes`);
  }
  const [verb, arg] = String(want).split(' ');
  // A clear failure beats Showdown's "Not all choices done" three turns later.
  if (req.forceSwitch && verb !== 'switch') {
    throw new Error(`${side.id} has to replace a fainted Pokemon, and the scenario says ${want}`);
  }
  if (!req.forceSwitch && verb === 'switch' && !side.pokemon[0].fainted && req.active?.[0]?.trapped) {
    throw new Error(`${side.id} is trapped and cannot ${want}`);
  }
  if (verb === 'switch') {
    const mon = [...orders[side.n]].find(([, at]) => at === Number(arg) - 1)[0];
    return { code: 4 + Number(arg) - 1, cmd: `switch ${side.pokemon.indexOf(mon) + 1}` };
  }
  const active = side.active[0];
  const entry = req.active?.[0];
  // A lock (a recharge, the second turn of Solar Beam) arrives as a request of
  // one move that does not line up with the slots: that is the forced action.
  if (entry && entry.moves.length === 1 && active.moveSlots.length > 1 && !entry.moves[0].disabled) {
    return { code: 11, cmd: 'move 1' };
  }
  const slot = active.moveSlots.findIndex((m) => m.id === Dex.toID(arg));
  if (slot < 0) throw new Error(`${side.id} has no move ${arg}`);
  return { code: slot, cmd: `move ${slot + 1}` };
}

// What the battle actually exercised, as dex ids. The coverage report reads it,
// so a scenario cannot claim a family it never reached: this comes from
// Showdown's own log, not from what the sets happen to list.
function touched(battle, seen, log = true) {
  const add = (s) => { if (s) seen.add(Dex.toID(String(s).replace(/^(move|item|ability|Ability):\s*/, ''))); };
  for (const line of log ? battle.log : []) {
    const part = line.split('|');
    const tag = part[1];
    if (['move', '-ability', '-item', '-enditem', '-status', '-start', '-singlemove',
         '-activate', '-sidestart', '-end'].includes(tag)) add(part[3]);
    if (tag === '-weather') add(part[2]);
    // Conditions the log names differently: a flinch or a recharge shows as the
    // reason a Pokemon cannot move, a partial trap as a tag on its damage.
    if (tag === 'cant') add(part[3] === 'recharge' ? 'mustrecharge' : part[3]);
    if (tag === '-mustrecharge') add('mustrecharge');
    if (line.includes('[partiallytrapped]')) add('partiallytrapped');
    for (const m of line.matchAll(/\[from\] (?:move|item|ability): ([^|[\]]+)/g)) add(m[1]);
  }
  // Passive abilities and held items never reach the log; anything that stood
  // on the field was in play. `log` false adds just those, for a caller that
  // samples the actives as the battle goes and reads the log once at the end.
  for (const side of battle.sides) for (const p of [side.active[0]]) { add(p.ability); add(p.item); }
  return seen;
}

function runScenario(spec) {
  const seed = seedString(spec.seed || 1);
  const battle = new Battle({ formatid: 'gen3randombattle', seed });
  // Patch before the teams go in: the leads switch in as the battle starts, and
  // a scenario may want to steer that too.
  const rng = scriptedRNG(battle, forceMap(spec.force || {}));
  battle.setPlayer('p1', { team: scenarioTeam(spec.p1) });
  battle.setPlayer('p2', { team: scenarioTeam(spec.p2) });
  debugSorts(battle, spec.name);
  const orders = partyOrders(battle);
  const before = exportState(battle, orders);
  const seen = touched(battle, new Set());
  let at = rng.draws.length;  // the start-up draws are already in `before`

  const segments = [];
  for (const turn of spec.turns) {
    if (battle.ended) break;
    const picked = battle.sides.map((side, i) => command(side, orders, turn[i]));
    for (const [i, side] of battle.sides.entries()) {
      if (picked[i].cmd && !side.choose(picked[i].cmd)) {
        throw new Error(`${side.id} refused ${picked[i].cmd}: ${battle.log.slice(-3).join(' | ')}`);
      }
    }
    battle.commitChoices();
    const drawn = rng.draws.slice(at);
    at = rng.draws.length;
    segments.push({
      choices: picked.map((p) => p.code),
      draws: drawn.map((d) => d.raw),
      names: drawn.map((d) => d.name || d.site),
      after: exportState(battle, orders),
      legal: battle.sides.map((side) => legalMask(side, orders)),
    });
    touched(battle, seen);
  }
  return { name: spec.name, before, segments, unused: rng.unused(), log: battle.log,
           touched: [...seen].sort(),
           turns: battle.turn, ended: battle.ended, winner: battle.winner || '' };
}

module.exports = { runScenario, touched };
