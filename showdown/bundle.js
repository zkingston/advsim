// The page's two scripts: web/dist/app.js (web/js bundled, at the end) and
// Showdown 0.11.11's simulator as one browser script, web/dist/showdown.js,
// defining `Showdown` = { Battle, Teams, PRNG, Dex, State, chooseReplacement, maxDamageMove }.
//
// The dex finds its data with fs.readdirSync and require(path) at run time,
// which a bundler cannot follow. The entry below requires every file gen 3
// reaches statically, under the path dex.js computes for it, and a global
// `require` serves them from that map. Everything else (learnsets, text) is
// MODULE_NOT_FOUND, which the dex already tolerates, or an empty stub.
'use strict';
const fs = require('fs');
const path = require('path');
const esbuild = require('esbuild');

const PS = path.dirname(require.resolve('pokemon-showdown/package.json'));
const DIST = path.join(PS, 'dist');
const MODS = ['gen3', 'gen4', 'gen5', 'gen6', 'gen7', 'gen8'];  // gen3's inherit chain
const SKIP = new Set(['learnsets', 'pokemongo']);
const OUT = path.resolve(__dirname, '../web/dist/showdown.js');

const files = [];
const add = (dir) => {
  for (const f of fs.readdirSync(path.join(DIST, dir))) {
    const id = f.replace(/\.js$/, '');
    if (f.endsWith('.js') && !SKIP.has(id)) files.push(`${dir}/${id}`);
  }
};
add('data');
for (const mod of MODS) add(`data/mods/${mod}`);
add('data/random-battles/gen3');

const entry = `
const MAP = {
${files.map((f) => `  ${JSON.stringify('/ps/' + f)}: () => require(${JSON.stringify(`pokemon-showdown/dist/${f}`)}),`).join('\n')}
  // Descriptions only; the simulator never reads them.
${['pokedex', 'moves', 'abilities', 'items', 'default'].map((t) => `  "/ps/data/text/${t}": () => new Proxy({}, { get: () => ({}) }),`).join('\n')}
  // Formats for other generations name mods that are not bundled.
  "/ps/config/formats": () => ({ Formats: require("pokemon-showdown/dist/config/formats").Formats
    .filter((f) => f.section || ${JSON.stringify(MODS)}.includes(f.mod)) }),
};
globalThis.require = (p) => {
  const out = [];
  for (const seg of p.replace(/\\.js$/, '').split('/')) { if (seg === '..') out.pop(); else if (seg && seg !== '.') out.push(seg); }
  const key = '/' + out.join('/');
  if (MAP[key]) return MAP[key]();
  const e = new Error('not bundled: ' + p); e.code = 'MODULE_NOT_FOUND'; throw e;
};
const { Battle } = require('pokemon-showdown/dist/sim/battle');
const { PRNG } = require('pokemon-showdown/dist/sim/prng');
const { Dex } = require('pokemon-showdown/dist/sim/dex');
const { Teams } = require('pokemon-showdown/dist/sim/teams');
const { State } = require('pokemon-showdown/dist/sim/state');
// The page's Emerald-style opponent: the sweep's own policy code, on this Dex.
const { chooseReplacement } = require('./lib/policy.js');
const { maxDamageMove } = require('./lib/estimate.js');
globalThis.Showdown = { Battle, PRNG, Dex, Teams, State, chooseReplacement, maxDamageMove };
`;

// A path module for the posix paths dex.js builds, and an fs that lists the mods.
const shims = {
  path: `const norm = (p) => { const out = []; for (const s of p.split('/')) { if (s === '..') out.pop(); else if (s && s !== '.') out.push(s); } return '/' + out.join('/'); };
    module.exports = { resolve: (...a) => norm(a.reduce((acc, s) => s.startsWith('/') ? s : acc + '/' + s, '')), join: (...a) => norm(a.join('/')), sep: '/' };`,
  ps: `module.exports = { Dex: require('pokemon-showdown/dist/sim/dex').Dex };`,
  lib: `module.exports = { Utils: require('./lib/utils').Utils };`,
  // dex-species compares a forme with its base to inherit the tier.
  util: `const eq = (a, b) => {
    if (Object.is(a, b)) return true;
    if (typeof a !== 'object' || typeof b !== 'object' || !a || !b || Object.getPrototypeOf(a) !== Object.getPrototypeOf(b)) return false;
    const ka = Object.keys(a), kb = Object.keys(b);
    return ka.length === kb.length && ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && eq(a[k], b[k]));
  };
  module.exports = { isDeepStrictEqual: eq };`,
  fs: `module.exports = { readdirSync: () => ${JSON.stringify(MODS)}, existsSync: () => false };`,
};

// The simulator bundle carries Showdown's and ts-chacha20's MIT notices, as their licenses require.
const LEGAL = ['pokemon-showdown', 'ts-chacha20']
  .map((p) => fs.readFileSync(path.join(__dirname, 'node_modules', p, 'LICENSE'), 'utf8')).join('\n');

esbuild.build({
  stdin: { contents: entry, resolveDir: __dirname },
  banner: { js: `/*!\n${LEGAL.replaceAll('*/', '* /')}*/` },
  // keepNames: State's save/restore names objects by class ("[Pokemon:p1a]"), which minifying renames.
  bundle: true, format: 'iife', platform: 'browser', outfile: OUT, minify: true, keepNames: true, legalComments: 'none',
  define: { __dirname: '"/ps/sim"', 'process.env.NODE_ENV': '"production"' },
  plugins: [{
    name: 'node-shims',
    setup(b) {
      b.onResolve({ filter: /^(node:)?(fs|path|util)$/ }, (a) => ({ path: a.path.replace('node:', ''), namespace: 'shim' }));
      b.onResolve({ filter: /^(node:)?[a-z_]+$/ }, (a) => (
        ['child_process', 'cluster', 'net', 'http', 'https', 'repl', 'crypto', 'os', 'zlib'].includes(a.path.replace('node:', ''))
          ? { path: a.path, namespace: 'empty' } : undefined));
      // lib/policy.js asks for the package; it gets the bundle's own Dex.
      b.onResolve({ filter: /^pokemon-showdown$/ }, () => ({ path: 'ps', namespace: 'shim' }));
      b.onResolve({ filter: /^\/ps\// }, (a) => ({ path: a.path, external: true }));  // the runtime map's
      // lib/index pulls in the server; the team generators want only Utils.
      b.onResolve({ filter: /^(\.\.\/)+lib$/ }, () => ({ path: 'lib', namespace: 'shim' }));
      // teams.js requires `random-battles/${mod}/teams`, which would glob every generation.
      b.onLoad({ filter: /dist[\\/]sim[\\/]teams\.js$/ }, (a) => ({
        contents: fs.readFileSync(a.path, 'utf8').replace(/require\(`\.\.\/data\/random-battles\/\$\{mod\}\/teams`\)/,
          'require("../data/random-battles/gen3/teams")'),
        loader: 'js', resolveDir: path.dirname(a.path),
      }));
      // gen3's generator subclasses every later one, whose modules load their own sets and
      // factory data (over half the bundle); gen3 never reads them.
      b.onLoad({ filter: /dist[\\/]data[\\/]random-battles[\\/].*\.json$/ }, (a) => (
        /random-battles[\\/]gen3[\\/]/.test(a.path) ? undefined : { contents: '{}', loader: 'json' }));
      b.onLoad({ filter: /.*/, namespace: 'shim' }, (a) => ({ contents: shims[a.path], loader: 'js', resolveDir: a.path === 'ps' ? __dirname : DIST }));
      b.onLoad({ filter: /.*/, namespace: 'empty' }, () => ({ contents: 'module.exports = {};', loader: 'js' }));
    },
  }],
  logLevel: 'warning',
}).then(() => console.log(`${OUT}: ${(fs.statSync(OUT).size / 1e6).toFixed(1)} MB, ${files.length} data files`));

// The page itself: web/js/app.js and its modules as one minified file, one request.
const APP = path.resolve(__dirname, '../web/dist/app.js');
esbuild.build({
  entryPoints: [path.resolve(__dirname, '../web/js/app.js')], bundle: true, format: 'esm', minify: true,
  outfile: APP, logLevel: 'warning',
}).then(() => console.log(`${APP}: ${(fs.statSync(APP).size / 1e3).toFixed(0)} KB`));

// The stylesheet: web/css/app.css and its imports as one file.
const CSS = path.resolve(__dirname, '../web/dist/app.css');
esbuild.build({
  entryPoints: [path.resolve(__dirname, '../web/css/app.css')], bundle: true, minify: true,
  outfile: CSS, logLevel: 'warning',
}).then(() => console.log(`${CSS}: ${(fs.statSync(CSS).size / 1e3).toFixed(0)} KB`));

// The MCTS worker: a classic script (it importScripts the simulator), web/js/mcts/worker.js bundled.
const WORKER = path.resolve(__dirname, '../web/dist/worker.js');
esbuild.build({
  entryPoints: [path.resolve(__dirname, '../web/js/mcts/worker.js')], bundle: true, format: 'iife', minify: true,
  outfile: WORKER, logLevel: 'warning',
}).then(() => console.log(`${WORKER}: ${(fs.statSync(WORKER).size / 1e3).toFixed(0)} KB`));
