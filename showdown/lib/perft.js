// Perft: from a position, every joint action and every chance outcome, to a
// depth of one or two decisions. Each leaf is a case in the shape replay()
// judges (the position's words, then per decision the choices, the raw draws,
// the words after and the legal masks), so the engine is compared leaf by leaf.
//
// Chance is enumerated depth first over Showdown's own draws: run with a
// prefix of forced raws, take the first outcome at every draw past it, then
// branch on every other outcome of each of those draws. Every outcome class of
// a draw gets one representative raw; damage rolls only min, mid and max.
'use strict';
const { State } = require('pokemon-showdown/dist/sim/state');
const { exportState, legalMask, carryReveals } = require('./export_state.js');
const { instrument, siteName } = require('./scripted_rng.js');
const play = require('./play.js');

const U32 = 2 ** 32;
const rawFor = (k, n) => Math.min(Math.ceil((k * U32) / n), U32 - 1);

// A secondary or self-drop roll is random(100) compared with that one
// secondary's chance, so 0 (it fires) and 99 (it does not, unless the chance
// is 100) cover both outcomes whatever the chance is.
const SECONDARY_KS = [0, 99];

// One representative raw per outcome class of the call that drew.
function outcomes(call, name) {
  const [fn, a, b, c] = call;
  if (fn === 'randomChance') return [0, rawFor(a, b)];
  const n = fn === 'random' ? (b === undefined ? a : b - a) : fn === 'sample' ? a : c - b;
  let ks = [...Array(n).keys()];
  if (name === 'damage_roll') ks = [0, n >> 1, n - 1];
  else if ((name === 'secondary' || name === 'self_drop') && n === 100) ks = SECONDARY_KS;
  return [...new Set(ks)].map((k) => rawFor(k, n));
}

// A copy of the battle with the party order carried over by array index,
// which serialisation keeps.
function clone(battle, orders) {
  const copy = State.deserializeBattle(State.serializeBattle(battle));
  // The copy inherits the whole log but not how much of it was sent, and
  // Showdown calls a thousand unsent lines an infinite loop. Nothing reads it.
  copy.log = [];
  copy.sentLogPos = 0;
  carryReveals(battle, copy);
  const copyOrders = copy.sides.map((side, s) => new Map(side.pokemon.map((p, i) =>
    [p, orders[s].get(battle.sides[s].pokemon[i])])));
  return [copy, copyOrders];
}

const actionsOf = (side, orders) => {
  const mask = legalMask(side, orders);
  return [...Array(12).keys()].filter((a) => mask & (1 << a));
};

// Every leaf one decision below `battle`, handed to `onLeaf(copy, orders,
// segment)` as it is found rather than kept. Returns the leaf count. `budget`
// is shared by the whole position ({ left }), so the cap bounds its time.
function expand(battle, orders, budget, onLeaf) {
  let count = 0;
  const [a0, a1] = battle.sides.map((side) => actionsOf(side, orders));
  for (const x of a0) {
    for (const y of a1) {
      const stack = [[]];
      while (stack.length) {
        const prefix = stack.pop();
        const [copy, copyOrders] = clone(battle, orders);
        const draws = [];
        instrument(copy, null, (v, key, call) => {
          if (!call) throw new Error(`perft: ${key} drew outside a PRNG wrapper`);
          const raw = draws.length < prefix.length ? prefix[draws.length] : outcomes(call, siteName(key))[0];
          draws.push({ raw, key, call });
          return raw;
        });
        const choices = [x, y];
        copy.sides.forEach((side, i) => { const c = play.command(side, choices[i], copyOrders); if (c) side.choose(c); });
        copy.commitChoices();
        for (let j = draws.length - 1; j >= prefix.length; j--) {
          const head = draws.slice(0, j).map((d) => d.raw);
          for (const raw of outcomes(draws[j].call, siteName(draws[j].key)).slice(1)) stack.push([...head, raw]);
        }
        count++;
        if (--budget.left < 0) throw new Error(`perft: over ${budget.total} leaves`);
        onLeaf(copy, copyOrders, {
          choices, draws: draws.map((d) => d.raw), sites: draws.map((d) => d.key),
          after: exportState(copy, copyOrders), legal: copy.sides.map((side) => legalMask(side, copyOrders)),
        });
      }
    }
  }
  return count;
}

// The leaves of one position, each handed to `emit` as a replay case. Depth 2
// expands every `stride`-th depth-1 leaf that is still a decision, in full.
// `budget` caps the leaves of the whole position, depth 2 included. Returns
// the depth-1 leaf count.
function perft(battle, orders, depth, budget, stride, emit) {
  const before = exportState(battle, orders);
  const shared = { left: budget, total: budget };
  let i = 0;
  return expand(battle, orders, shared, (leaf, leafOrders, seg) => {
    if (depth < 2 || leaf.ended || i++ % stride) {
      emit({ before, segments: [seg] });
      return;
    }
    expand(leaf, leafOrders, shared, (_, __, seg2) => emit({ before, segments: [seg, seg2] }));
  });
}

module.exports = { perft, outcomes };
