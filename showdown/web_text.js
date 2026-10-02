// Gen 3 descriptions for the page's tooltips: the moves, abilities and items in
// web/data/vocab.json, as Showdown 0.11.11 words them for gen 3. The bundle
// leaves Showdown's text out (2 MB for every generation); this is the part used.
//   node showdown/web_text.js   → web/data/text.json
'use strict';
const fs = require('fs');
const path = require('path');
const { Dex } = require('pokemon-showdown');

const gen3 = Dex.mod('gen3');
const WEB = path.resolve(__dirname, '../web/data');
const vocab = JSON.parse(fs.readFileSync(path.join(WEB, 'vocab.json'), 'utf8'));
const sorted = (entries) => Object.fromEntries(entries.sort(([a], [b]) => (a < b ? -1 : 1)));
const desc = (e) => e.shortDesc || e.desc || '';

const out = {
  moves: sorted(vocab.moves.filter(Boolean).map((id) => {
    const m = gen3.moves.get(id);
    return [id, { desc: desc(m), category: m.category, power: m.basePower, accuracy: m.accuracy === true ? 0 : m.accuracy,
      priority: m.priority, pp: m.pp }];
  })),
  abilities: sorted(vocab.abilities.filter(Boolean).map((id) => [id, desc(gen3.abilities.get(id))])),
  items: sorted(vocab.items.filter(Boolean).map((id) => [id, desc(gen3.items.get(id))])),
};
const missing = ['moves', 'abilities', 'items'].flatMap((t) => Object.entries(out[t]).filter(([, v]) => !(v.desc ?? v)).map(([k]) => `${t}/${k}`));
if (missing.length) throw new Error(`no description: ${missing.join(', ')}`);
fs.writeFileSync(path.join(WEB, 'text.json'), `${JSON.stringify(out, null, 2)}\n`);
console.log(`web/data/text.json: ${Object.keys(out.moves).length} moves, ${Object.keys(out.abilities).length} abilities, ${Object.keys(out.items).length} items`);
