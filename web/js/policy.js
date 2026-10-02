// The page's network: PolicyTF's forward pass (advsim/net_tf.py) on one observation: pre-RMSNorm
// transformer blocks with a feed-forward per token type, `react` (the opponent's predicted action
// entering the upper layers as one more token), and the compact layout's 17 tokens: the 12
// Pokemon (each active's state added into its token, the foe active's revealed moves too), the
// own active's 4 moves, and the summary (the field added in). The full 24-token layout is not
// here, nor PolicyV2; tools/export_web.py refuses them. With `damage`, each own Pokemon token also
// reads the most it could deal to the foe active and take from its revealed moves, and each move
// token its damage range (damage.js). `model`: export_web.py's JSON (weights, net.py's feature
// columns, config [d, layers, heads, react, compact, damage, ...], the damage tables), `L`:
// vocab.json's layout. Returns 12 logits (illegal ones -1e9) and the value.
//
// A token that attention masks out (an absent Pokemon, an empty move slot) is never computed:
// nothing reads it, since it is no one's key and the heads read only legal actions' tokens.

import { damageFeatures } from './damage.js';
import { linears } from './matmul.js';

// Where the summary and the prediction sit, each token's learned type embedding and the
// feed-forward it takes (net_tf.py's COMPACT_KINDS and COMPACT_FF_KIND).
const SUMMARY = 16, PRED = 17, TOKENS = 18;
const KIND = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 7, 8];
const FF = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3];
const EPS = 1.1920928955078125e-7;  // torch.finfo(float32).eps, RMSNorm's

// An IEEE half as a number: sign, 5-bit exponent, 10-bit fraction.
function half(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

// One array: base64 of little-endian float16 (tools/export_web.py), decoded to float64, the JS
// arithmetic's own type.
function decode({ shape, data }) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  return { shape, w: Float64Array.from(new Uint16Array(bytes.buffer), half) };
}

// erf to ~1e-7 (Abramowitz and Stegun 7.1.26), for the exact GELU torch uses.
function erf(x) {
  const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export function loadPolicy(model, L) {
  const P = Object.fromEntries(Object.entries(model.weights).map(([k, v]) => [k, decode(v)]));
  const F = model.features, D = model.damage;
  const [d, layers, heads, react, compact, damage] = model.config;
  if (model.arch !== 3 || !compact) throw new Error('policy.js: the page runs the compact PolicyTF only');
  const dh = d / heads, M = L.MON.length, A = L.ACTIVE.length;
  const col = (n) => L.MON.indexOf(n);
  const C = { present: col('present'), active: col('active'), species: col('species'), ability: col('ability'),
    item: col('item'), status: col('status'), move0: col('move0'), pp0: col('pp0') };
  const AC = { type0: L.ACTIVE.indexOf('type0'), type1: L.ACTIVE.indexOf('type1'), choice: L.ACTIVE.indexOf('choice_move') };
  const weather = L.FIELD.indexOf('weather');
  const nMoves = P['move.weight'].shape[0], nSpecies = P['species.weight'].shape[0];
  const lower = react ? Math.floor(layers / 2) : layers;
  const blocks = [...Array(layers).keys()].map((i) => (i < lower ? `lower.${i}` : `upper.${i - lower}`));

  const { layer, matmul } = linears(P, TOKENS);
  // y[yOff..] = W x[xOff..] + b for one vector, by name.
  const linear = (name, x, xOff, y, yOff, act = null) => matmul(layer(name), x, xOff, 1, y, yOff, act);
  const embed = (name, idx, y, off) => {
    const { shape: [, n], w } = P[`${name}.weight`];
    for (let i = 0; i < n; i++) y[off + i] = w[idx * n + i];
    return n;
  };
  // RMSNorm of x[off..] into y[yOff..], with weight `w` (a name, or the array).
  const rms = (w, x, off, y, yOff = 0) => {
    if (typeof w === 'string') w = P[`${w}.weight`].w;
    let s = 0;
    for (let i = 0; i < d; i++) s += x[off + i] * x[off + i];
    const r = 1 / Math.sqrt(s / d + EPS);
    for (let i = 0; i < d; i++) y[yOff + i] = x[off + i] * r * w[i];
  };

  const x = new Float64Array(TOKENS * d), norm = new Float64Array(d), qkv = new Float64Array(TOKENS * 3 * d);
  const nIn = P['mon_in.weight'].shape[1], monIn = new Float64Array(12 * nIn), moveIn = new Float64Array(P['move_in.weight'].shape[1]);
  const actIn = new Float64Array(P['act_in.weight'].shape[1]), fieldIn = new Float64Array(P['field_in.weight'].shape[1]);
  const moveMean = new Float64Array(32), score = new Float64Array(TOKENS), pred = new Float64Array(nMoves + nSpecies);
  const expect = new Float64Array(65), headIn = new Float64Array(2 * d), headHid = new Float64Array(d), one = new Float64Array(2);
  const kindW = P['kind.weight'].w;
  // What is folded into other tokens: each active's state, the foe's moves, the field.
  const acts = new Float64Array(2 * d), foeMoves = new Float64Array(d), field = new Float64Array(d), tmp = new Float64Array(d);

  // Each block's layers, looked up once; the feed-forward per token type (net_tf.py's GROUPS).
  const B = blocks.map((name) => ({ n1: P[`${name}.n1.weight`].w, n2: P[`${name}.n2.weight`].w,
    qkv: layer(`${name}.qkv`), out: layer(`${name}.out`),
    ff: [0, 1, 2, 3].map((g) => (P[`${name}.ff.${g}.0.weight`] ? [layer(`${name}.ff.${g}.0`), layer(`${name}.ff.${g}.2`)] : null)) }));
  const Xn = new Float64Array(TOKENS * d), Hd = new Float64Array(TOKENS * 2 * d), Yo = new Float64Array(TOKENS * d);
  const scale = 1 / Math.sqrt(dh);

  // One block over the kept tokens (rows r of the batch): attention, then each token type's
  // feed-forward, each as one matrix product over its rows.
  function block(b, keep) {
    const n = keep.length;
    keep.forEach((t, r) => rms(b.n1, x, t * d, Xn, r * d));
    matmul(b.qkv, Xn, 0, n, qkv, 0);
    for (let r = 0; r < n; r++) {
      for (let h = 0; h < heads; h++) {
        const q = r * 3 * d + h * dh;
        let top = -Infinity;
        for (let u = 0; u < n; u++) {
          const k = u * 3 * d + d + h * dh;
          let s = 0;
          for (let e = 0; e < dh; e++) s += qkv[q + e] * qkv[k + e];
          score[u] = s * scale;
          if (score[u] > top) top = score[u];
        }
        let z = 0;
        for (let u = 0; u < n; u++) z += (score[u] = Math.exp(score[u] - top));
        const a = r * d + h * dh;
        for (let e = 0; e < dh; e++) Xn[a + e] = 0;
        for (let u = 0; u < n; u++) {
          const v = u * 3 * d + 2 * d + h * dh, p = score[u] / z;
          for (let e = 0; e < dh; e++) Xn[a + e] += p * qkv[v + e];
        }
      }
    }
    matmul(b.out, Xn, 0, n, Yo, 0);
    keep.forEach((t, r) => { for (let i = 0; i < d; i++) x[t * d + i] += Yo[r * d + i]; });
    for (let g = 0; g < 4; g++) {
      const rows = keep.filter((t) => FF[t] === g);
      if (!rows.length) continue;
      rows.forEach((t, r) => rms(b.n2, x, t * d, Xn, r * d));
      matmul(b.ff[g][0], Xn, 0, rows.length, Hd, 0, gelu);
      matmul(b.ff[g][1], Hd, 0, rows.length, Yo, 0);
      rows.forEach((t, r) => { for (let i = 0; i < d; i++) x[t * d + i] += Yo[r * d + i]; });
    }
  }

  return function forward(obs) {
    const mon = (t, c) => obs[L.MON_BASE + t * M + c];
    const legal = (a) => obs[L.MASK_BASE + a] !== 0;
    const keep = [];
    const dmg = damage ? damageFeatures(obs, L, D) : null;
    // The 12 Pokemon: each present one's input row, then one product for them all.
    for (let t = 0; t < 12; t++) {
      if (!mon(t, C.present)) continue;
      const row = keep.length * nIn;
      keep.push(t);
      let o = row;
      o += embed('species', mon(t, C.species), monIn, o);
      o += embed('ability', mon(t, C.ability), monIn, o);
      o += embed('item', mon(t, C.item), monIn, o);
      o += embed('status', mon(t, C.status), monIn, o);
      moveMean.fill(0);
      for (let j = 0; j < 4; j++) {  // the mean of the four slots' embeddings, empty slots' included
        const w = P['move.weight'].w, id = mon(t, C.move0 + j);
        for (let i = 0; i < 32; i++) moveMean[i] += w[id * 32 + i] / 4;
      }
      monIn.set(moveMean, o);
      o += 32;
      for (const [c, s] of F.mon_numeric) monIn[o++] = mon(t, c) / s;
      monIn[o++] = t < 6 ? 0 : 1;
      if (dmg) { monIn[o++] = t < 6 ? dmg.party[t][0] : 0; monIn[o] = t < 6 ? dmg.party[t][1] : 0; }  // foe tokens: none
    }
    matmul(layer('mon_in'), monIn, 0, keep.length, Yo, 0);
    keep.forEach((t, r) => x.set(Yo.subarray(r * d, (r + 1) * d), t * d));
    // Each active's four moves (own, then foe's revealed), with their matchup: the own moves are
    // tokens, the foe's are summed into its active.
    const actives = [-1, -1];
    foeMoves.fill(0);
    for (let side = 0; side < 2; side++) {
      for (let t = side * 6; t < side * 6 + 6; t++) if (mon(t, C.active) && mon(t, C.present)) actives[side] = t;
      const act = actives[side];
      for (let j = 0; j < 4; j++) {
        const id = act < 0 ? 0 : mon(act, C.move0 + j);
        if (!id) continue;
        embed('move', id, moveIn, 0);
        moveIn[32] = act < 0 ? 0 : mon(act, C.pp0 + j) / 64;
        moveIn[33] = obs[L.MATCHUP_BASE + side * 8 + 2 * j] / 4;
        moveIn[34] = obs[L.MATCHUP_BASE + side * 8 + 2 * j + 1];
        if (dmg) [moveIn[35], moveIn[36]] = (side === 0 ? dmg.ownMoves : dmg.foeMoves)[j];
        if (side === 0) {
          keep.push(12 + j);
          linear('move_in', moveIn, 0, x, (12 + j) * d);
        } else {
          linear('move_in', moveIn, 0, tmp, 0);
          for (let i = 0; i < d; i++) foeMoves[i] += tmp[i];
        }
      }
    }
    // The actives' state and the field, added into the actives' Pokemon and the summary.
    for (let side = 0; side < 2; side++) {
      const row = (i) => obs[L.ACTIVE_BASE + side * A + i];
      let k = 0;
      for (const i of F.active_boosts) actIn[k++] = row(i) / 6;
      k += embed('types', row(AC.type0), actIn, k);
      k += embed('types', row(AC.type1), actIn, k);
      for (const i of F.active_flags) actIn[k++] = row(i);
      embed('move', row(AC.choice), actIn, k);
      linear('act_in', actIn, 0, acts, side * d);
    }
    let o = embed('weather', obs[L.FIELD_BASE + weather], fieldIn, 0);
    for (const [c, s] of F.field_numeric) fieldIn[o++] = obs[L.FIELD_BASE + c] / s;
    o += embed('move', obs[L.HISTORY_BASE], fieldIn, o);
    embed('move', obs[L.HISTORY_BASE + 1], fieldIn, o);
    linear('field_in', fieldIn, 0, field, 0);
    for (let i = 0; i < d; i++) x[SUMMARY * d + i] = P.summary.w[i] + field[i];
    if (actives[0] >= 0) for (let i = 0; i < d; i++) x[actives[0] * d + i] += acts[i];
    if (actives[1] >= 0) for (let i = 0; i < d; i++) x[actives[1] * d + i] += acts[d + i] + foeMoves[i];
    keep.push(SUMMARY);
    for (const t of keep) for (let i = 0; i < d; i++) x[t * d + i] += kindW[KIND[t] * d + i];

    blocks.forEach((name, i) => {
      if (react && i === lower) {  // the opponent's predicted action, as one more token
        rms('opp_action.0', x, SUMMARY * d, norm);
        linear('opp_action.1', norm, 0, pred, 0);
        let top = -Infinity, z = 0;
        for (const v of pred) top = Math.max(top, v);
        for (let a = 0; a < pred.length; a++) z += (pred[a] = Math.exp(pred[a] - top));
        expect.fill(0);
        const mw = P['move.weight'].w, sw = P['species.weight'].w;
        for (let a = 0; a < nMoves; a++) { const q = pred[a] / z; for (let e = 0; e < 32; e++) expect[e] += q * mw[a * 32 + e]; }
        for (let a = 0; a < nSpecies; a++) {
          const q = pred[nMoves + a] / z;
          for (let e = 0; e < 32; e++) expect[32 + e] += q * sw[a * 32 + e];
          expect[64] += q;
        }
        linear('pred_in', expect, 0, x, PRED * d);
        for (let e = 0; e < d; e++) x[PRED * d + e] += kindW[8 * d + e];
        keep.push(PRED);
      }
      block(B[i], keep);
    });

    const s = new Float64Array(d);
    rms('norm', x, SUMMARY * d, s);
    const head = (name, t) => {
      rms('norm', x, t * d, headIn);
      headIn.set(s, d);
      linear(`${name}.0`, headIn, 0, headHid, 0, (v) => (v < 0 ? 0 : v));
      linear(`${name}.2`, headHid, 0, one, 0);
      return one[0];
    };
    const logits = new Array(12).fill(-1e9);
    for (let j = 0; j < 4; j++) if (legal(j)) logits[j] = head('move_head', 12 + j);
    for (let i = 0; i < 6; i++) if (legal(4 + i)) logits[4 + i] = head('switch_head', i);
    linear('other', s, 0, one, 0);
    if (legal(10)) logits[10] = one[0];
    if (legal(11)) logits[11] = one[1];
    linear('v.0', s, 0, headHid, 0, (v) => (v < 0 ? 0 : v));
    linear('v.2', headHid, 0, one, 0);
    return { logits, value: one[0] };
  };
}

// An action code sampled at temperature `t` (0: the top action).
export function sample(logits, t, random = Math.random) {
  if (t <= 0) return logits.indexOf(Math.max(...logits));
  const top = Math.max(...logits), p = logits.map((v) => Math.exp((v - top) / t));
  let r = random() * p.reduce((a, b) => a + b);
  for (let i = 0; i < p.length; i++) if ((r -= p[i]) <= 0) return i;
  return logits.indexOf(top);
}
