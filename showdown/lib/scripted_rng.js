// Showdown's PRNG, patched: every raw u32 is logged, and a named call site can
// be told what to return. L2 scenarios steer outcomes with it (forced crit,
// forced miss, a chosen speed tie) and the engine replays the same raw values,
// so forcing can only change which code path a scenario reaches, never what
// either side computes from it.
//
// The site machinery lives here because trace.js and the scenario runner must
// name a call site the same way: `force` keys are the catalog's names.
'use strict';

const U32 = 2 ** 32;
const WRAPPERS = new Set(['Battle.random', 'Battle.randomChance', 'Battle.sample', 'Battle.shuffle']);

// Scenario `force` names for the sites that matter; unaliased site keys stay raw.
const ALIASES = [
  [/^Battle\.endTurn</, 'quickclaw'], [/^BattleActions\.tryMoveHit</, 'accuracy'], [/^BattleActions\.getDamage</, 'crit'],
  [/^Battle\.randomizer</, 'damage_roll'], [/^BattleActions\.secondaries</, 'secondary'], [/^BattleActions\.selfDrops</, 'self_drop'],
  [/^Battle\.speedSort</, 'speedtie'], [/^new Pokemon</, 'gender'], [/^Battle\.getRandomSwitchable</, 'drag_in'],
  [/^par:Battle\.onBeforeMove</, 'par_check'], [/^frz:Battle\.onBeforeMove</, 'thaw'], [/^slp:Battle\.onStart</, 'sleep_turns'],
  [/^confusion:Battle\.onBeforeMove</, 'confusion_check'], [/^sleeptalk:Battle\.onHit</, 'sleeptalk'], [/^Battle\.durationCallback</, 'volatile_duration'],
  [/^Side\.randomFoe</, 'trace_target'], [/^BattleQueue\.insertChoice</, 'switchin_tie'],
];

// Unaliased effect-handler sites get `<effect>_<handler>`, e.g. static:Battle.onDamagingHit -> static_damaginghit.
function siteName(key) {
  const alias = ALIASES.find(([re]) => re.test(key));
  if (alias) return alias[1];
  const m = /^(\w+):\w+\.on([A-Z]\w*)</.exec(key);
  return m ? `${m[1]}_${m[2].toLowerCase()}` : null;
}

// Site key: [effect id ':' if the top frame is a handler] + two innermost non-RNG frames.
function siteKey(battle, self) {
  const frames = new Error().stack.split('\n').slice(1)
    .filter((l) => !l.includes('prng.js') && !l.includes(self) && !l.includes(__filename))
    .map((l) => l.trim().slice(3).split(' (')[0].replace(/^async /, ''))
    .filter((f) => !WRAPPERS.has(f));
  const eff = /\b(on[A-Z]\w*|\w+Callback)$/.test(frames[0]) && battle.effect && battle.effect.id ? battle.effect.id + ':' : '';
  return eff + frames.slice(0, 2).join('<');
}

// Patch the PRNG: log every raw u32 (or feed one from `script`), the site key
// and the outermost PRNG call with its arguments.
function instrument(battle, script, sink, self = __filename) {
  const prng = battle.prng, rng = prng.rng, next = rng.next.bind(rng);
  let cur = null, depth = 0, di = 0;
  rng.next = () => {
    let v;
    if (script) { if (di >= script.length) throw new Error('script exhausted'); v = script[di++]; } else v = next();
    if (sink) {
      const forced = sink(v, siteKey(battle, self), cur);
      if (forced !== undefined) v = forced;  // a sink that only logs returns nothing
    }
    return v;
  };
  const argsOf = { random: (a) => a, randomChance: (a) => a, sample: ([items]) => [items.length], shuffle: ([items, s, e]) => [items.length, s, e] };
  for (const fn of Object.keys(argsOf)) {
    const orig = prng[fn].bind(prng);
    prng[fn] = (...a) => { if (depth++ === 0) cur = [fn, ...argsOf[fn](a)]; try { return orig(...a); } finally { if (--depth === 0) cur = null; } };
  }
}

// The lowest raw u32 that makes `random(n)` return k: ceil(k * 2^32 / n), which
// is 0 for k = 0. Showdown maps a draw with floor(raw * n / 2^32).
function rawForOutcome(k, n) {
  if (!(k >= 0 && k < n)) throw new Error(`outcome ${k} is not in [0, ${n})`);
  const raw = Math.min(Math.ceil((k * U32) / n), U32 - 1);
  if (Math.floor((raw * n) / U32) !== k) throw new Error(`no u32 gives outcome ${k} of ${n}`);
  return raw;
}

// What a forced value means depends on the call the site made.
//   randomChance(num, den)  true or false
//   random(n) / random(m,n) the number the call should return
//   sample(len)             the index into the list
//   shuffle(len, s, e)      the index this swap should pick
function rawFor(call, want) {
  const [fn, a, b, c] = call;
  if (fn === 'randomChance') return want ? 0 : rawForOutcome(a, b);
  if (fn === 'random') return b === undefined ? rawForOutcome(want, a) : rawForOutcome(want - a, b - a);
  if (fn === 'sample') return rawForOutcome(want, a);
  if (fn === 'shuffle') return rawForOutcome(want - b, c - b);
  throw new Error(`cannot force ${fn}`);
}

// `force` maps a site name to a standing value or to a list consumed in order:
// `{"crit": false}` means no crit all battle, `{"damage_roll": [15, 0]}` means
// max then min and then whatever the battle's own PRNG says. A site with
// nothing forced always draws for itself. Returns the log the engine replays:
// one entry per draw, in order.
function scriptedRNG(battle, force = {}) {
  const standing = new Map(Object.entries(force).filter(([, v]) => !Array.isArray(v)));
  const queues = new Map(Object.entries(force).filter(([, v]) => Array.isArray(v)).map(([k, v]) => [k, [...v]]));
  const used = new Set();
  const draws = [];
  instrument(battle, null, (value, key, call) => {
    const name = siteName(key);
    const queue = queues.get(name);
    let raw = value, want;
    if (queue && queue.length) want = queue.shift();
    else if (standing.has(name)) want = standing.get(name);
    if (want !== undefined) {
      if (!call) throw new Error(`${name} drew outside a PRNG wrapper, so there is nothing to force`);
      raw = rawFor(call, want);
      used.add(name);
    }
    draws.push({ raw, site: key, name, call });
    return raw;
  });
  return {
    draws,
    // A force that never fired is a scenario that did not reach the site it
    // meant to steer, which is a broken test rather than a passing one.
    unused: () => [...new Set([...standing.keys(), ...queues.keys()])]
      .filter((k) => !used.has(k) || (queues.get(k) || []).length),
  };
}

// A PRNG seed string from an integer, the form Showdown takes.
const seedString = (n) => `sodium,${n.toString(16).padStart(64, '0')}`;

// ADVSIM_SORT_DEBUG=1 logs every sort of more than one entry to stderr, with
// each handler's effect, side, order, sub-order and speed: how a draw is traced
// back to the handler list that made it.
function debugSorts(battle, tag) {
  if (!process.env.ADVSIM_SORT_DEBUG) return;
  const base = battle.speedSort.bind(battle);
  battle.speedSort = (list, cmp) => {
    if (list.length > 1) {
      console.error(`SORT ${tag} turn=${battle.turn} ` + list.map((h) =>
        `${h.effect?.id ?? h.choice ?? '?'}/${(h.effectHolder ?? h.pokemon)?.side?.id ?? '?'}` +
        `#${h.order ?? '-'}.${h.subOrder ?? '-'}@${h.speed ?? '-'}`).join(' '));
    }
    return base(list, cmp);
  };
}

module.exports = { scriptedRNG, instrument, siteName, seedString, debugSorts };
