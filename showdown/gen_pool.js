// Seeded, sharded Gen 3 Random Battle team pool with Showdown-computed stats.
// Usage: node showdown/gen_pool.js --n 1000000 --seed 1 [--shards 32] [--out artifacts/pool.jsonl]
// The teams depend on --seed and --shards (each shard is its own stream), not on the machine:
// the committed manifest's pool is --n 1000000 --seed 1 --shards 32, the defaults.
// One team per line: {"s":shard,"i":index,"mons":[{species,level,gender,ability,item,hpType,maxhp,stats,moves,maxpp}]}
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { Dex, Teams } = require('pokemon-showdown');
const { Battle } = require('pokemon-showdown/dist/sim/battle');

const FORMAT = 'gen3randombattle';
const seedFor = (seed, shard, tag) => 'sodium,' + crypto.createHash('sha256').update(`${seed}:${shard}:${tag}`).digest('hex');

// Stats, gender, hpType and max PP as Showdown resolves them; a p1-only Battle is enough.
function monRow(p) {
  return {
    species: p.species.id, level: p.level, gender: p.gender, ability: p.ability, item: p.item, hpType: p.hpType,
    maxhp: p.maxhp, stats: [p.storedStats.atk, p.storedStats.def, p.storedStats.spa, p.storedStats.spd, p.storedStats.spe],
    moves: p.moveSlots.map((m) => m.id), maxpp: p.moveSlots.map((m) => m.maxpp),
  };
}

function* teams(seed, shard, count) {
  const gen = Teams.getGenerator(FORMAT, seedFor(seed, shard, 'team'));
  for (let i = 0; i < count; i++) {
    const b = new Battle({ formatid: FORMAT, seed: seedFor(seed, shard, 'battle') });
    b.setPlayer('p1', { team: gen.getTeam() });
    yield JSON.stringify({ s: shard, i, mons: b.p1.pokemon.map(monRow) });
  }
}

function worker() {
  const { seed, shard, count, part } = workerData;
  const out = fs.openSync(part, 'w');
  let buf = [], done = 0;
  for (const line of teams(seed, shard, count)) {
    buf.push(line);
    if (buf.length === 1000) { fs.writeSync(out, buf.join('\n') + '\n'); done += buf.length; buf = []; parentPort.postMessage(done); }
  }
  if (buf.length) fs.writeSync(out, buf.join('\n') + '\n');
  fs.closeSync(out);
  parentPort.postMessage(count);
}

function arg(name, dflt) { const i = process.argv.indexOf('--' + name); return i < 0 ? dflt : process.argv[i + 1]; }

async function main() {
  const n = +arg('n', 1000000), seed = +arg('seed', 1), shards = +arg('shards', 32);
  const out = path.resolve(arg('out', path.join(__dirname, '..', 'artifacts', 'pool.jsonl')));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  const per = Math.ceil(n / shards), t0 = Date.now(), progress = new Array(shards).fill(0);
  let lastPrint = 0;
  const parts = [];
  await Promise.all(Array.from({ length: shards }, (_, k) => new Promise((resolve, reject) => {
    const count = Math.max(0, Math.min(per, n - k * per));
    const part = `${out}.${k}.part`;
    parts.push(part);
    const w = new Worker(__filename, { workerData: { seed, shard: k, count, part } });
    w.on('message', (d) => {
      progress[k] = d;
      const tot = progress.reduce((a, b) => a + b, 0);
      if (Date.now() - lastPrint > 5000) { lastPrint = Date.now(); console.error(`${tot}/${n} teams, ${Math.round(tot / ((Date.now() - t0) / 1000))}/s`); }
    });
    w.on('error', reject);
    w.on('exit', (c) => (c ? reject(new Error(`shard ${k} exit ${c}`)) : resolve()));
  })));
  const fd = fs.openSync(out, 'w');
  for (const p of parts) { fs.writeSync(fd, fs.readFileSync(p)); fs.unlinkSync(p); }
  fs.closeSync(fd);

  // Self-check: shard 0 regenerated in-process must match the file head.
  // Partial read: the full pool exceeds Node's max string length.
  const fh = fs.openSync(out, 'r'), hb = Buffer.alloc(1 << 16);
  const head = hb.subarray(0, fs.readSync(fh, hb, 0, hb.length, 0)).toString('utf8').split('\n', 3);
  fs.closeSync(fh);
  let k = 0;
  for (const line of teams(seed, 0, 3)) if (line !== head[k++]) throw new Error(`pool not reproducible at line ${k}`);
  console.log(JSON.stringify({ out: path.relative(process.cwd(), out), teams: n, shards, seed, secs: +((Date.now() - t0) / 1000).toFixed(1), bytes: fs.statSync(out).size }));
}

if (isMainThread) main().catch((e) => { console.error(e); process.exit(1); }); else worker();
