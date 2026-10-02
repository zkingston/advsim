// The page's side of the MCTS: a pool of workers (worker.js), each running an independent
// search of the same root with its own seed (root parallelism); their root statistics are
// merged by search.js's decide.
import { publicView } from './search.js';

export class Mcts {
  // `worker`, `showdown`: the two scripts' URLs, as the page loaded them (so the browser's copy
  // serves every worker). `data`: a promise of the parsed { vocab, setdist, model } JSON, which
  // the page fetches once and hands to each worker rather than every worker downloading it.
  constructor({ worker, showdown, data }, n) {
    this.workers = Array.from({ length: n }, () => new Worker(worker));
    this.pending = new Map();
    this.next = 0;
    this.ready = data.then((d) => Promise.all(this.workers.map((w) => new Promise((resolve, reject) => {
      // A failure rejects whatever waits on it: the start-up, or every search in flight.
      w.onerror = (e) => { const err = new Error(`MCTS worker: ${e.message}`); reject(err); this.fail(err); };
      w.onmessage = ({ data: m }) => {
        if (m.type === 'ready') resolve();
        else if (m.id === undefined) reject(new Error(`MCTS worker: ${m.message}`));
        else this.settle(m);
      };
      w.postMessage({ type: 'init', showdown, ...d });
    }))));
  }

  settle(m) {
    const p = this.pending.get(m.id);
    this.pending.delete(m.id);
    if (m.type === 'stats') p.resolve(m.stats);
    else p.reject(new Error(`MCTS worker: ${m.message}`));
  }

  fail(err) {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  // Root statistics for side `me` of `battle`, whose own view is `view`: each worker searches
  // descents / n (or until params.ms) with its own seed.
  // The battle is read before the first await: the caller may go on to choose and commit.
  async search(battle, me, view, speciesTypes, params) {
    const json = globalThis.Showdown.State.serializeBattle(battle);
    json.log = [];
    json.inputLog = [];
    // The other side as the public log shows it; the class and the species table stay behind.
    const other = publicView(battle.log, me === 0 ? 'p2' : 'p1', speciesTypes);
    const strip = (st) => ({ ...st, speciesTypes: null });
    const views = me === 0 ? [strip(view), strip(other)] : [strip(other), strip(view)];
    await this.ready;
    const n = this.workers.length, each = Math.max(1, Math.ceil(params.descents / n));
    return Promise.all(this.workers.map((w, i) => new Promise((resolve, reject) => {
      const id = this.next++;
      this.pending.set(id, { resolve, reject });
      w.postMessage({ type: 'search', id, root: { json, me, views },
        params: { ...params, descents: each, seed: (params.seed + i * 7919) >>> 0 } });
    })));
  }

  close() { for (const w of this.workers) w.terminate(); }
}
