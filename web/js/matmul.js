// The network's matrix product (policy.js's hot loop): WebAssembly SIMD where the browser has it
// (web/wasm/matmul.c, built into matmul_wasm.js by showdown/build_wasm.js), a JS loop otherwise.
import { MATMUL_WASM } from './matmul_wasm.js';

// The kernel, compiled once per page or worker; null without WebAssembly SIMD.
const KERNEL = (() => {
  try {
    const bytes = Uint8Array.from(atob(MATMUL_WASM), (c) => c.charCodeAt(0));
    return WebAssembly.validate(bytes) ? new WebAssembly.Module(bytes) : null;
  } catch { return null; }
})();

// `P`'s Linear layers (decoded weights, by name) and the product over them. `rows`: the most rows
// one product takes. -> { layer(name), matmul(L, X, xOff, n, Y, yOff, act) }.
export function linears(P, rows) {
  // Every Linear layer's weights, looked up once: { w [out * inp], b [out], out, inp }. With the
  // kernel, each also sits in WebAssembly memory as float32, rows zero-padded to k (a multiple of
  // 4), at `at` (weights) and `atB` (bias), next to scratch rows for the input and the output.
  const names = Object.keys(P).filter((k) => k.endsWith('.weight') && P[k].shape.length === 2 && P[k.replace(/weight$/, 'bias')]);
  const pad = (n) => (n + 3) & ~3;
  const layers_ = new Map(names.map((k) => {
    const { shape: [out, inp], w } = P[k], name = k.slice(0, -'.weight'.length);
    return [name, { w, b: P[`${name}.bias`].w, out, inp, k: pad(inp) }];
  }));
  let simd = null;
  if (KERNEL) {
    let words = 0;
    for (const L of layers_.values()) { L.at = words; L.atB = words + L.out * L.k; words += L.out * L.k + L.out; }
    const width = Math.max(...[...layers_.values()].map((L) => Math.max(L.k, L.out)));
    const xAt = words, yAt = words + rows * width;
    words += 2 * rows * width;
    const memory = new WebAssembly.Memory({ initial: Math.ceil((4 * words) / 65536) });
    const F = new Float32Array(memory.buffer);
    for (const L of layers_.values()) {
      for (let o = 0; o < L.out; o++) for (let i = 0; i < L.inp; i++) F[L.at + o * L.k + i] = L.w[o * L.inp + i];
      F.set(L.b, L.atB);
    }
    simd = { F, xAt, yAt, run: new WebAssembly.Instance(KERNEL, { env: { __linear_memory: memory } }).exports.matmul };
  }
  // Y[r * out + o] = act(b[o] + W[o] . X[xOff + r * inp ..]) for the n rows of X: the forward's
  // hot loop (two thirds of an MCTS descent). With the kernel, the rows go into WebAssembly memory
  // (their padding zeroed: an earlier, wider layer left values there), the SIMD product runs, and
  // the outputs come back through `act`. Without it, the JS loop below.
  function matmul(L, X, xOff, n, Y, yOff, act = null) {
    if (!simd) return matmulJS(L, X, xOff, n, Y, yOff, act);
    const { F, xAt, yAt } = simd, { inp, k, out } = L;
    for (let r = 0; r < n; r++) {
      const src = xOff + r * inp, dst = xAt + r * k;
      for (let i = 0; i < inp; i++) F[dst + i] = X[src + i];
      for (let i = inp; i < k; i++) F[dst + i] = 0;
    }
    simd.run(4 * L.at, 4 * L.atB, out, k, 4 * xAt, n, 4 * yAt);
    const m = n * out;
    if (act) for (let j = 0; j < m; j++) Y[yOff + j] = act(F[yAt + j]);
    else for (let j = 0; j < m; j++) Y[yOff + j] = F[yAt + j];
  }
  return { layer: (name) => layers_.get(name), matmul };
}

// The same in JS: rows four at a time, so each weight read serves four dot products; a lone row
// runs on four accumulators, which V8 overlaps where one stalls.
function matmulJS(L, X, xOff, n, Y, yOff, act = null) {
  const { w, b, out, inp } = L;
  for (let o = 0; o < out; o++) {
    const ro = o * inp, bo = b[o];
    let r = 0;
    for (; r + 3 < n; r += 4) {
      const x0 = xOff + r * inp, x1 = x0 + inp, x2 = x1 + inp, x3 = x2 + inp;
      let s0 = bo, s1 = bo, s2 = bo, s3 = bo;
      for (let i = 0; i < inp; i++) {
        const wi = w[ro + i];
        s0 += wi * X[x0 + i]; s1 += wi * X[x1 + i]; s2 += wi * X[x2 + i]; s3 += wi * X[x3 + i];
      }
      const y = yOff + r * out + o;
      Y[y] = act ? act(s0) : s0; Y[y + out] = act ? act(s1) : s1;
      Y[y + 2 * out] = act ? act(s2) : s2; Y[y + 3 * out] = act ? act(s3) : s3;
    }
    for (; r < n; r++) {
      const xr = xOff + r * inp;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, i = 0;
      for (; i + 3 < inp; i += 4) {
        s0 += w[ro + i] * X[xr + i];
        s1 += w[ro + i + 1] * X[xr + i + 1];
        s2 += w[ro + i + 2] * X[xr + i + 2];
        s3 += w[ro + i + 3] * X[xr + i + 3];
      }
      for (; i < inp; i++) s0 += w[ro + i] * X[xr + i];
      const v = bo + (s0 + s1) + (s2 + s3);
      Y[yOff + r * out + o] = act ? act(v) : v;
    }
  }
}
