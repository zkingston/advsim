// The bots: the PPO network (models/, the first listed), the MCTS on top of it in a pool of
// workers, the Emerald-style scripted player, and random play. Each picks an action code for
// player p of a Game; the network's own probabilities are the human's assist.
import { observe, legalMask } from './observation.js';
import { loadPolicy, sample } from './policy.js';
import { Mcts } from './mcts/client.js';
import { decide } from './mcts/search.js';
import { vocab } from './game.js';
import { settings } from './settings.js';
import { json } from './fetch.js';
import { summarize } from './thinking.js';

const { chooseReplacement, maxDamageMove } = window.Showdown;
const PPO = (await json('models/index.json'))[0];
const model = json(`models/${PPO}.json`);
const network = model.then((m) => loadPolicy(m, vocab.L));

// The MCTS workers, made on first use and remade when the Settings' worker count changes. The
// scripts' URLs are index.html's, carrying the deploy's stamps; the data is fetched once here.
let pool = null, mctsData = null;
function mcts() {
  if (pool && pool.workers.length === settings.mcts.workers) return pool;
  pool?.close();
  mctsData ??= Promise.all([json('data/setdist.json'), model]).then(([setdist, m]) => ({ vocab: vocab.v, setdist, model: m }));
  pool = new Mcts({ worker: document.querySelector('link[href*="dist/worker.js"]').href,
    showdown: document.querySelector('script[src*="dist/showdown.js"]').src, data: mctsData }, settings.mcts.workers);
  return pool;
}

const legal = (mask) => [...Array(12).keys()].filter((a) => mask & (1 << a));

// The sweep's `maxdamage` player (showdown/lib/estimate.js, policy.js), which reads the
// battle itself as a cartridge AI does. Replacement codes index the first request's party
// order, so the chooser gets that order.
function emerald(game, p, st) {
  const side = game.battle.sides[p], mask = legalMask(st);
  if (game.views.at(-1).requests[p].forceSwitch) {
    const order = new Map(side.pokemon.map((q) => [q, st.sides[st.me].order.indexOf(q.name)]));
    return 4 + order.get(side.pokemon[chooseReplacement(game.battle, side, order)]);
  }
  const moves = [0, 1, 2, 3, 11].filter((a) => mask & (1 << a));
  return moves.length ? maxDamageMove(game.battle, side, moves) : legal(mask)[0];
}

// Player p's choice: { code, value, thought }: value its own win estimate where it has one,
// thought the MCTS's reasoning for the Bot panel (thinking.js). The battle is read before the
// first await, so the other side may choose meanwhile; the caller commits only after this.
export async function bot(game, p) {
  const who = game.players[p], st = game.state(p);
  if (who === 'random') { const l = legal(legalMask(st)); return { code: l[Math.floor(Math.random() * l.length)] }; }
  if (who === 'emerald') return { code: emerald(game, p, st) };
  if (who === 'mcts') {
    const m = settings.mcts;
    try {
      const stats = await mcts().search(game.battle, p, st, vocab.speciesTypes,
        { descents: m.descents, ms: m.seconds * 1000, depth: m.depth, gamma: m.gamma, prune: m.prune, seed: Math.floor(Math.random() * 2 ** 32) });
      const d = decide(stats, p, legalMask(st), { final: m.final, temperature: settings.temperature });
      return { code: d.action, value: p === 0 ? d.value : -d.value, thought: summarize(game, p, st, d, stats) };  // decide's value is p1's; this, p's
    } catch (e) {
      console.error(e);  // the network plays this decision; the next one starts a fresh pool
      pool?.close();
      pool = null;
    }
  }
  const { logits, value } = (await network)(observe(st, vocab));
  return { code: sample(logits, settings.temperature), value };
}

// The network's probabilities (temperature 1) for each of player p's 12 action codes.
export const assist = (game, p) => probsOf(game.state(p));

// The same from any InfoState, such as p1's at a past view (watching, the scrubbed view).
export async function probsOf(st) {
  const { logits } = (await network)(observe(st, vocab));
  const top = Math.max(...logits), e = logits.map((v) => Math.exp(v - top)), z = e.reduce((a, b) => a + b);
  return e.map((v) => v / z);
}
