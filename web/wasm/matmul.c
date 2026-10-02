// The page network's matrix product in WebAssembly SIMD (web/js/policy_tf.js's hot loop):
//   y[r * out + o] = b[o] + sum_i w[o * k + i] * x[r * k + i]   for rows r < n, outputs o < out,
// with k a multiple of 4 (weight rows and input rows zero-padded to it), float32 throughout as
// torch computes. Rows go four at a time, so each four-wide weight load serves four dot products.
// Built by showdown/build_wasm.js into web/js/matmul_wasm.js; no libc, no stack.
#include <wasm_simd128.h>

static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) +
         wasm_f32x4_extract_lane(v, 3);
}

void matmul(const float *w, const float *b, int out, int k, const float *x, int n, float *y) {
  for (int o = 0; o < out; o++) {
    const float *wo = w + o * k;
    int r = 0;
    for (; r + 3 < n; r += 4) {
      const float *x0 = x + r * k, *x1 = x0 + k, *x2 = x1 + k, *x3 = x2 + k;
      v128_t s0 = wasm_f32x4_splat(0), s1 = s0, s2 = s0, s3 = s0;
      for (int i = 0; i < k; i += 4) {
        v128_t wi = wasm_v128_load(wo + i);
        s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(wi, wasm_v128_load(x0 + i)));
        s1 = wasm_f32x4_add(s1, wasm_f32x4_mul(wi, wasm_v128_load(x1 + i)));
        s2 = wasm_f32x4_add(s2, wasm_f32x4_mul(wi, wasm_v128_load(x2 + i)));
        s3 = wasm_f32x4_add(s3, wasm_f32x4_mul(wi, wasm_v128_load(x3 + i)));
      }
      y[r * out + o] = b[o] + hsum(s0);
      y[(r + 1) * out + o] = b[o] + hsum(s1);
      y[(r + 2) * out + o] = b[o] + hsum(s2);
      y[(r + 3) * out + o] = b[o] + hsum(s3);
    }
    for (; r < n; r++) {
      const float *xr = x + r * k;
      v128_t s = wasm_f32x4_splat(0);
      for (int i = 0; i < k; i += 4) s = wasm_f32x4_add(s, wasm_f32x4_mul(wasm_v128_load(wo + i), wasm_v128_load(xr + i)));
      y[r * out + o] = b[o] + hsum(s);
    }
  }
}
