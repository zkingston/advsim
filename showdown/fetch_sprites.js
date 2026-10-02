// The page's sprites, served from its own origin: Showdown's Gen 3 front and
// back sprites for every species in web/data/vocab.json.
//   node showdown/fetch_sprites.js   → web/sprites/gen3/<id>.png, web/sprites/gen3-back/<id>.png,
//   web/sprites/categories/{Physical,Special,Status}.png
// Files are named by our species id; Showdown names formes with a hyphen
// (deoxys-attack), its spriteid. Same-origin images also let the page read their pixels, which it does to
// stand each Pokemon on its platform.
'use strict';
const fs = require('fs');
const path = require('path');
const { Dex } = require('pokemon-showdown');

const WEB = path.resolve(__dirname, '../web');
const species = JSON.parse(fs.readFileSync(path.join(WEB, 'data/vocab.json'), 'utf8')).species.filter(Boolean);

(async () => {
  let bytes = 0;
  for (const dir of ['gen3', 'gen3-back']) {
    fs.mkdirSync(path.join(WEB, 'sprites', dir), { recursive: true });
    for (const id of species) {
      const res = await fetch(`https://play.pokemonshowdown.com/sprites/${dir}/${Dex.species.get(id).spriteid}.png`, { headers: { 'User-Agent': 'advsim fetch_sprites' } });
      if (!res.ok) throw new Error(`${dir}/${id}: HTTP ${res.status}`);
      const png = Buffer.from(await res.arrayBuffer());
      fs.writeFileSync(path.join(WEB, 'sprites', dir, `${id}.png`), png);
      bytes += png.length;
    }
  }
  // The move-category icons. Gen 3 shows none (its split is by type); these are Showdown's, in Gen 4's style.
  fs.mkdirSync(path.join(WEB, 'sprites', 'categories'), { recursive: true });
  for (const c of ['Physical', 'Special', 'Status']) {
    const res = await fetch(`https://play.pokemonshowdown.com/sprites/categories/${c}.png`, { headers: { 'User-Agent': 'advsim fetch_sprites' } });
    if (!res.ok) throw new Error(`categories/${c}: HTTP ${res.status}`);
    fs.writeFileSync(path.join(WEB, 'sprites', 'categories', `${c}.png`), Buffer.from(await res.arrayBuffer()));
  }
  // The Substitute doll, front and back.
  for (const dir of ['gen3', 'gen3-back']) {
    const res = await fetch(`https://play.pokemonshowdown.com/sprites/substitutes/${dir}/substitute.png`, { headers: { 'User-Agent': 'advsim fetch_sprites' } });
    if (!res.ok) throw new Error(`substitutes/${dir}: HTTP ${res.status}`);
    fs.writeFileSync(path.join(WEB, 'sprites', dir, '_substitute.png'), Buffer.from(await res.arrayBuffer()));
  }
  console.log(`${2 * species.length} sprites, ${(bytes / 1e6).toFixed(2)} MB, 3 category icons and the Substitute doll`);
})();
