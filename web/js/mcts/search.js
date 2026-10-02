// Open-loop simultaneous-move MCTS over Showdown worlds: a port of advsim/engine/tree.py
// and advsim/search/mcts.py. A node is a sequence of joint actions from the root, not a
// state: every descent plays the path again in a freshly drawn world (world.js), so chance
// and hidden information are sampled rather than stored.
//
// Per node, each player's cumulative regret for decoupled regret matching, and a prior (the
// policy network's, from the forward pass that scored the node's leaf). Selection samples
// each player's strategy with GAMMA of the prior mixed in; backup is the engine's
// importance-weighted regret update. The root also keeps visits and summed value per joint
// action, the empirical matrix `decide` solves with RM+. Values are p1's; p2 plays -v.
import { InfoState } from '../infostate.js';
import { observe, legalMask } from '../observation.js';
import { command } from '../command.js';
import { buildWorld, rng } from './world.js';
import { rootLabels } from './labels.js';

// Worlds whose network prior of the other side is averaged for the Bot panel's "likely" column.
const LIKELY_WORLDS = 16;

const A = 12;
const bits = (mask) => { const out = []; for (let a = 0; a < A; a++) if (mask & (1 << a)) out.push(a); return out; };

function softmax(logits, mask) {
  const legal = bits(mask), top = Math.max(...legal.map((a) => logits[a]));
  const p = new Float64Array(A);
  let z = 0;
  for (const a of legal) z += (p[a] = Math.exp(logits[a] - top));
  for (const a of legal) p[a] /= z;
  return p;
}

// An InfoState copied for one descent. Messages and structuredClone drop the class, so
// both go through here; the species table is shared, not copied.
export function cloneView(st) {
  const { speciesTypes, request, ...rest } = st;
  const c = Object.assign(Object.create(InfoState.prototype), structuredClone(rest));
  c.speciesTypes = speciesTypes;
  c.request = request;
  return c;
}

// What `player` knows from the public log alone: its InfoState fed a log whose split lines
// all show the public half. The searcher gets this for the other side, whose private lines
// (exact HP, its own moves) it must not read; a world's request then fills in that side's
// own team as the world has it.
export function publicView(log, player, speciesTypes) {
  const pub = [];
  for (let i = 0; i < log.length; i++) {
    if (log[i].startsWith('|split|')) {
      pub.push(log[i], log[i + 2], log[i + 2]);
      i += 2;
    } else {
      pub.push(log[i]);
    }
  }
  const st = new InfoState(player, speciesTypes);
  st.feed(pub, pub.length);
  return st;
}

class Node {
  constructor() {
    this.child = new Map();
    this.regret = [new Float64Array(A), new Float64Array(A)];
    this.prior = [null, null];  // none: uniform over the legal actions
  }

  priorOf(p, legal, a) {
    if (!(legal & (1 << a))) return 0;
    const pr = this.prior[p];
    let total = 0;
    if (pr) for (const i of bits(legal)) total += pr[i];
    return total > 0 ? pr[a] / total : 1 / bits(legal).length;
  }

  strategy(p, legal, a) {
    if (!(legal & (1 << a))) return 0;
    let total = 0;
    for (const i of bits(legal)) total += Math.max(this.regret[p][i], 0);
    return total > 0 ? Math.max(this.regret[p][a], 0) / total : this.priorOf(p, legal, a);
  }

  // `legal` without the actions whose prior is under `rho` times the best (engine/tree.py's
  // prune): the budget goes to fewer joint actions. The best always stays; no prior, no change.
  prune(p, legal, rho) {
    const pr = this.prior[p];
    if (!(rho > 0) || !pr) return legal;
    let best = 0;
    for (const a of bits(legal)) best = Math.max(best, pr[a]);
    if (best <= 0) return legal;
    let kept = 0;
    for (const a of bits(legal)) if (pr[a] >= rho * best) kept |= 1 << a;
    return kept;
  }

  sampling(p, legal, a, gamma) {
    return (1 - gamma) * this.strategy(p, legal, a) + gamma * this.priorOf(p, legal, a);
  }

  sample(p, legal, gamma, r) {
    let u = r(), last = 10;
    for (const a of bits(legal)) {
      last = a;
      if ((u -= this.sampling(p, legal, a, gamma)) < 0) return a;
    }
    return last;
  }
}

export class Search {
  // `table`: world.js's SetTable. `vocab`: observation.js's Vocab. `policy`: policy.js's
  // forward, obs -> { logits, value } from the viewer's side.
  constructor(table, vocab, policy) {
    Object.assign(this, { table, vocab, policy });
  }

  // `root`: { json (State.serializeBattle, log dropped), me (the searching side), views
  // ([p1, p2] InfoStates at the root: the searcher's own, and the other side's public one) }.
  // params: { descents, ms, depth, gamma, prune, nodes, seed }. Returns the root's statistics.
  run(root, params) {
    const { descents = 512, ms = Infinity, depth = 6, gamma = 0.1, prune = 0, nodes = 4096, seed = 1 } = params;
    const r = rng(seed);
    this.root = new Node();
    this.count = 1;
    // For the Bot panel only (nothing below reads them): the root's statistics tallied by what
    // the other side's code was in each world (labels.js), and the network's prior over those.
    this.keyed = {};
    this.likely = {};
    this.likelyWorlds = 0;
    const cellN = new Float64Array(A * A), cellW = new Float64Array(A * A);
    const t0 = Date.now();
    let done = 0;
    for (; done < descents && Date.now() - t0 < ms; done++) {
      this.descend(root, { depth, gamma, prune, nodes }, r, cellN, cellW, done === 0);
    }
    return { cellN, cellW, prior: this.root.prior.map((p) => (p ? Array.from(p) : null)), descents: done, ms: Date.now() - t0,
      keyed: this.keyed, likely: this.likely, likelyWorlds: this.likelyWorlds };
  }

  // Both players' value (p1's) and priors at a position, from their own views.
  evaluate(states) {
    const f = states.map((s) => this.policy(observe(s, this.vocab)));
    return { value: Math.max(-1, Math.min(1, (f[0].value - f[1].value) / 2)),
      prior: f.map((x, p) => softmax(x.logits, legalMask(states[p]))) };
  }

  descend(root, { depth, gamma, prune, nodes }, r, cellN, cellW, first) {
    const world = buildWorld(root.json, root.me, root.views[root.me], this.table, r);
    const states = root.views.map((v, p) => {
      const s = cloneView(v);
      s.pos = 0;  // the world's log starts empty
      s.takeRequest(world.sides[p].activeRequest);
      return s;
    });
    if (first) this.root.prior = this.evaluate(states).prior;
    const me = root.me, opp = 1 - me, labels = rootLabels(world, states[opp], opp, root.views[me]);
    if (this.likelyWorlds < LIKELY_WORLDS) {
      const pr = first ? this.root.prior[opp] : softmax(this.policy(observe(states[opp], this.vocab)).logits, legalMask(states[opp]));
      for (let a = 0; a < A; a++) if (labels[a] && pr[a]) this.likely[labels[a]] = (this.likely[labels[a]] || 0) + pr[a];
      this.likelyWorlds++;
    }
    const path = [];
    let node = this.root, leaf = null;
    for (let d = 0; d < depth && !world.ended; d++) {
      const m = states.map((s, p) => node.prune(p, legalMask(s), prune));
      const a = [node.sample(0, m[0], gamma, r), node.sample(1, m[1], gamma, r)];
      path.push({ node, a, m, prob: a.map((x, p) => node.sampling(p, m[p], x, gamma)) });
      world.sides.forEach((side, p) => {
        const req = side.activeRequest;
        if (!req || req.wait) return;
        const c = command(a[p], states[p], req);
        if (c && !side.choose(c)) side.autoChoose();  // a world can refuse what the view allowed
      });
      world.commitChoices();
      states.forEach((s, p) => { s.feed(world.log, world.log.length); s.takeRequest(world.sides[p].activeRequest); });
      const cell = a[0] * A + a[1];
      let child = node.child.get(cell);
      if (!child) {
        if (this.count < nodes) {
          child = new Node();
          node.child.set(cell, child);
          this.count++;
          leaf = child;
        }
        break;
      }
      node = child;
    }
    let u;
    if (world.ended) {
      u = world.winner === world.sides[0].name ? 1 : world.winner === world.sides[1].name ? -1 : 0;
    } else {
      const e = this.evaluate(states);
      u = e.value;
      if (leaf) leaf.prior = e.prior;
    }
    for (const step of path) {
      if (step.node === this.root) {
        cellN[step.a[0] * A + step.a[1]] += 1;
        cellW[step.a[0] * A + step.a[1]] += u;
        const k = this.keyed[labels[step.a[opp]]] ??= { n: new Float64Array(A), w: new Float64Array(A), codes: new Float64Array(A) };
        k.n[step.a[me]] += 1;
        k.w[step.a[me]] += u;
        k.codes[step.a[opp]] += 1;
      }
      for (let p = 0; p < 2; p++) {
        const up = p === 0 ? u : -u;
        for (const a of bits(step.m[p])) step.node.regret[p][a] += (a === step.a[p] ? up / step.prob[p] : 0) - up;
      }
    }
  }
}

// RM+ in self-play on the root's empirical matrix (tree.py's matrix_policy, over root_policy's
// q: an unvisited cell between two visited actions takes the root's mean). -> [p1, p2] strategies.
function solveRoot(cellN, cellW, iters = 1000) {
  let n = 0, w = 0, m = [0, 0];
  for (let c = 0; c < A * A; c++) {
    if (cellN[c] > 0) { n += cellN[c]; w += cellW[c]; m[0] |= 1 << Math.floor(c / A); m[1] |= 1 << (c % A); }
  }
  const mean = w / Math.max(n, 1);
  const q = (a, c) => (cellN[a * A + c] > 0 ? cellW[a * A + c] / cellN[a * A + c] : mean);
  const regret = [new Float64Array(A), new Float64Array(A)], avg = [new Float64Array(A), new Float64Array(A)];
  const cur = [new Float64Array(A), new Float64Array(A)];
  for (let it = 0; it < iters; it++) {
    for (let p = 0; p < 2; p++) {
      const legal = bits(m[p]);
      const total = legal.reduce((s, a) => s + regret[p][a], 0);
      for (let a = 0; a < A; a++) {
        cur[p][a] = m[p] & (1 << a) ? (total > 0 ? regret[p][a] / total : 1 / legal.length) : 0;
        avg[p][a] += cur[p][a];
      }
    }
    let v = 0;
    for (let a = 0; a < A; a++) for (let c = 0; c < A; c++) v += cur[0][a] * cur[1][c] * q(a, c);
    for (let a = 0; a < A; a++) {
      let row = 0, col = 0;
      for (let c = 0; c < A; c++) { row += cur[1][c] * q(a, c); col += cur[0][c] * q(c, a); }
      if (m[0] & (1 << a)) regret[0][a] = Math.max(regret[0][a] + row - v, 0);
      if (m[1] & (1 << a)) regret[1][a] = Math.max(regret[1][a] - col + v, 0);
    }
  }
  return { strategy: avg.map((x) => Array.from(x, (y) => y / iters)), value: mean };
}

// `weights` over the legal bitmask, normalised; uniform where no legal action has weight.
function restrict(weights, legal) {
  const w = Array.from({ length: A }, (_, a) => (legal & (1 << a) ? Math.max(weights[a] || 0, 0) : 0));
  const total = w.reduce((s, x) => s + x, 0);
  return total > 0 ? w.map((x) => x / total) : w.map((_, a) => (legal & (1 << a) ? 1 / bits(legal).length : 0));
}

// mcts.py's decide over merged root statistics: the searcher's root strategy on its true
// legal actions, then `final` ('eq' or 'mix', the geometric mean with the root
// prior), sharpened by `temperature` (0: the top action). -> { action, dist, play, value (p1's),
// visits, and for the page's Bot panel: strategy (both sides' root equilibrium), cellN, cellW }.
export function decide(stats, me, legal, { final = 'mix', temperature = 0.25 }, random = Math.random) {
  const cellN = new Float64Array(A * A), cellW = new Float64Array(A * A);
  const prior = new Float64Array(A);
  let priors = 0;
  for (const s of stats) {
    for (let c = 0; c < A * A; c++) { cellN[c] += s.cellN[c]; cellW[c] += s.cellW[c]; }
    if (s.prior[me]) { priors++; for (let a = 0; a < A; a++) prior[a] += s.prior[me][a]; }
  }
  const solved = solveRoot(cellN, cellW);
  const dist = restrict(solved.strategy[me], legal);
  let play = dist;
  if (final === 'mix' && priors) {
    const pr = restrict(prior, legal);
    play = restrict(dist.map((x, a) => Math.sqrt(x * pr[a])), legal);
  }
  let action;
  if (temperature <= 0) {
    action = play.indexOf(Math.max(...play));
  } else {
    const sharp = restrict(play.map((x) => x ** (1 / temperature)), legal);
    let u = random();
    action = bits(legal).at(-1);
    for (const a of bits(legal)) if ((u -= sharp[a]) < 0) { action = a; break; }
  }
  return { action, dist, play, value: solved.value, visits: cellN.reduce((s, x) => s + x, 0), strategy: solved.strategy, cellN, cellW };
}
