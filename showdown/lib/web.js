// The browser page's converter and network (web/js/), run under Node so
// tests/test_web.py can hold them to the Python ones.
'use strict';
const fs = require('fs');
const path = require('path');
const { InfoState } = require('../../web/js/infostate.js');
const { Vocab, observe } = require('../../web/js/observation.js');
const { loadPolicy } = require('../../web/js/policy.js');

const WEB = path.resolve(__dirname, '../../web');
const read = (f) => JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));
let vocab = null;
const getVocab = () => (vocab = vocab || new Vocab(read('data/vocab.json')));
const policies = new Map();

// Both players' vectors at the start and after every segment, as live/parity.py walks a case.
function observeCase(c) {
  const v = getVocab();
  const states = ['p1', 'p2'].map((p) => new InfoState(p, v.speciesTypes));
  return [c.start, ...c.segments.map((s) => s.view)].map((view) => states.map((st, p) => {
    st.feed(c.log, view.cursor);
    st.takeRequest(view.requests[p]);
    return Array.from(observe(st, v));
  }));
}

function forward(model, rows) {
  if (!policies.has(model)) policies.set(model, loadPolicy(read(`models/${model}.json`), getVocab().L));
  const f = policies.get(model);
  const out = rows.map((r) => f(Int16Array.from(r)));
  return { logits: out.map((o) => o.logits), value: out.map((o) => o.value) };
}

module.exports = { observeCase, forward };
