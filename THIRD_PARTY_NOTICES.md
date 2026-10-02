# Third-party notices

advsim reimplements Pokémon Showdown's Gen 3 battle rules and ships data derived
from it. Pokémon is © Nintendo, Game Freak and Creatures Inc. This is a
non-commercial fan project, not affiliated with or endorsed by them or by
Pokémon Showdown.

## Pokémon Showdown

`artifacts/dex_raw.json` (Showdown's Gen 3 data, including the source of its
callbacks), `artifacts/catalog.json`, `web/data/text.json` and
`web/data/setdist.json` are dumped from the `pokemon-showdown` npm package,
0.11.11, and the engine in `advsim/engine/` ports its battle logic. The web
page's `dist/showdown.js` (built by `showdown/bundle.js`, not committed) bundles
the package itself, with ts-chacha20 below.

```
The MIT License (MIT)

Copyright (c) 2011-2026 Guangcong Luo and other contributors http://pokemonshowdown.com/

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## ts-chacha20

Bundled into the web page's `dist/showdown.js` as Showdown's PRNG dependency.

```
MIT License

Copyright (c) 2017 Mykola Bubelich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Sprites

The web page shows Showdown's Gen 3 sprites and move-category icons. They are
not in this repository: `showdown/fetch_sprites.js` downloads them from
play.pokemonshowdown.com when the page is deployed. The sprites are images from
the games, © Nintendo, Game Freak and Creatures Inc.

## The Emerald AI write-up

The Emerald-style baseline (`showdown/lib/policy.js`, `showdown/lib/emerald.json`,
`advsim/engine/policy_emerald.py`) follows a public write-up of the Pokémon
Emerald cartridge's battle AI:
https://docs.google.com/document/d/13E61Jj4KwhIy3ZKgLjPY-_uWKWklfBR_zUlhZHUZqik.
Only facts about the game's behaviour are taken from it.
