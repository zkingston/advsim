// The MCTS in a Web Worker (bundled to dist/worker.js by showdown/bundle.js). It loads the
// simulator and builds the set table and the network once, from the data the page sends, then
// runs one search per message and posts the root's statistics back; client.js splits a
// decision across workers and merges them.
import { Vocab } from '../observation.js';
import { loadPolicy } from '../policy.js';
import { SetTable } from './world.js';
import { Search, cloneView } from './search.js';

let search = null;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      self.importScripts(data.showdown);
      const vocab = new Vocab(data.vocab);
      search = new Search(new SetTable(data.setdist, vocab), vocab, loadPolicy(data.model, vocab.L));
      self.postMessage({ type: 'ready' });
    } else if (data.type === 'search') {
      // Views arrive without their class or species table (client.js strips it); put both back.
      const views = data.root.views.map((st) => { const c = cloneView(st); c.speciesTypes = search.vocab.speciesTypes; return c; });
      const stats = search.run({ ...data.root, views }, data.params);
      self.postMessage({ type: 'stats', id: data.id, stats });
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: data.id, message: String(e && e.stack || e) });
  }
};
