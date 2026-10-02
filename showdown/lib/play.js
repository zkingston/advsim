// The policies the sweep plays with: L3's policy mix. Each picks one action
// code per side from Showdown's legal mask, using the harness PRNG only, so
// the battle's own stream is never touched.
//
// A policy is a move chooser, whether it switches of its own accord (one turn
// in three, like random play), and a replacement chooser.
'use strict';
const { legalMask } = require('./export_state.js');
const emerald = require('./policy.js');
const { maxDamageMove } = require('./estimate.js');

const bits = (mask, lo, hi) => [...Array(hi - lo).keys()].map((i) => i + lo).filter((i) => mask & (1 << i));

const MOVES = {
  random: (ctx, side, moves) => moves[ctx.prng.random(moves.length)],
  maxdamage: (ctx, side, moves) => maxDamageMove(ctx.battle, side, moves),
  // A status move when one is legal, otherwise any move.
  status: (ctx, side, moves) => {
    const active = side.active[0];
    const status = moves.filter((m) => m < 4 && ctx.battle.dex.moves.get(active.moveSlots[m].id).category === 'Status');
    const list = status.length ? status : moves;
    return list[ctx.prng.random(list.length)];
  },
};

const REPLACE = {
  random: (ctx, side, switches) => switches[ctx.prng.random(switches.length)],
  emerald: (ctx, side) => 4 + ctx.orders[side.n].get(side.pokemon[emerald.chooseReplacement(ctx.battle, side, ctx.orders[side.n])]),
};

const POLICIES = {
  random: { move: MOVES.random, switches: true, replace: REPLACE.random },
  emerald: { move: MOVES.random, switches: true, replace: REPLACE.emerald },
  maxdamage: { move: MOVES.maxdamage, switches: false, replace: REPLACE.emerald },
  status: { move: MOVES.status, switches: true, replace: REPLACE.random },
  switchaverse: { move: MOVES.random, switches: false, replace: REPLACE.random },
};
const NAMES = Object.keys(POLICIES);

// Each side's policy for one battle; 'mix' draws one per side.
function policiesFor(name, prng) {
  if (name === 'mix') return [0, 1].map(() => NAMES[prng.random(NAMES.length)]);
  if (!POLICIES[name]) throw new Error(`unknown policy ${name}; expected mix or one of ${NAMES}`);
  return [name, name];
}

// One action code for `side`. `ctx` is { battle, prng, orders }.
function pick(name, ctx, side) {
  const policy = POLICIES[name];
  const mask = legalMask(side, ctx.orders);
  const moves = bits(mask, 0, 4).concat(bits(mask, 11, 12));
  const switches = bits(mask, 4, 10);
  if (!moves.length && !switches.length) return 10;
  if (!moves.length) return policy.replace(ctx, side, switches);
  if (policy.switches && switches.length && ctx.prng.random(3) === 0) return switches[ctx.prng.random(switches.length)];
  return policy.move(ctx, side, moves);
}

// The Showdown choice for an action code, or null for a pass. Switches name the
// party slot in the stable order; Showdown's own array moves the active to 0.
function command(side, action, orders) {
  if (action === 10) return null;
  if (action === 11) return 'move 1';
  if (action < 4) return `move ${action + 1}`;
  return `switch ${side.pokemon.findIndex((p) => orders[side.n].get(p) === action - 4) + 1}`;
}

module.exports = { pick, policiesFor, command, NAMES };
