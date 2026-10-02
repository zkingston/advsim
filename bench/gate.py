"""Gate microbenchmark: Warp against JAX and raw CUDA on the same mini-step.

Runs three workload shapes at four batch sizes, checks that all three
implementations reach a bit-identical state, and reports throughput plus the
launch overhead that GPU-resident search would pay.

    uv run --group bench python bench/gate.py [--quick]
"""
import argparse
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'microbench'))

import numpy as np  # noqa: E402

import tables as T  # noqa: E402

BATCHES = (1024, 4096, 16384, 65536)
MODES = ('dense', 'search', 'search_graph')
REPEATS = 3
NVCC = os.environ.get('NVCC') or shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'


def index_maps(batch):
    """Slot maps shared by all three implementations; CUDA rebuilds them identically."""
    half = batch // 2
    idx = (np.arange(batch, dtype=np.uint32) * np.uint32(2654435761)) & np.uint32(batch - 1)
    src = (np.arange(half, dtype=np.int32) * 5 + 3) & (half - 1)
    dst = half + np.arange(half, dtype=np.int32)
    return idx.astype(np.int32), src, dst


def run_warp(batch, iters, mode):
    import warp as wp
    import warp_bench as w
    dev = 'cuda:0'
    idx_np, src_np, dst_np = index_maps(batch)
    idx = wp.array(idx_np, dtype=wp.int32, device=dev)
    src = wp.array(src_np, dtype=wp.int32, device=dev)
    dst = wp.array(dst_np, dtype=wp.int32, device=dev)
    s = w.make_state(batch, dev)
    half = batch // 2

    def iteration():
        if mode == 'dense':
            wp.launch(w.k_step, dim=batch, inputs=[s], device=dev)
        else:
            wp.launch(w.k_fork, dim=half, inputs=[s, src, dst], device=dev)
            wp.launch(w.k_step_idx, dim=batch, inputs=[s, idx], device=dev)

    for _ in range(20):
        iteration()
    wp.synchronize()
    graph = None
    if mode == 'search_graph':
        with wp.ScopedCapture() as cap:
            iteration()
        graph = cap.graph
    times = []
    for _ in range(REPEATS):
        wp.launch(w.k_init, dim=batch, inputs=[s], device=dev)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            wp.capture_launch(graph) if graph else iteration()
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), w.checksum(s, batch, dev)


def run_jax(batch, iters, mode):
    import jax
    import jax.numpy as jnp
    import jax_bench as j
    idx_np, src_np, dst_np = index_maps(batch)
    idx, src, dst = jnp.asarray(idx_np), jnp.asarray(src_np), jnp.asarray(dst_np)
    state, lanes = j.make_state(batch)

    def iteration(st):
        if mode == 'dense':
            return j.step_batch(st, lanes)
        return j.step_batch_idx(j.fork_batch(st, src, dst), lanes, idx)

    loop = None
    if mode == 'search_graph':  # JAX's equivalent of graph capture: the loop inside one jit
        @jax.jit
        def loop(st):
            def body(_, s_):
                return j.step_batch_idx(j.fork_batch(s_, src, dst), lanes, idx)
            return jax.lax.fori_loop(0, iters, body, st)

    st = state
    for _ in range(3):
        st = loop(st) if loop else iteration(st)
    jax.block_until_ready(st)
    times = []
    for _ in range(REPEATS):
        st = state
        jax.block_until_ready(st)
        t0 = time.perf_counter()
        if loop:
            st = loop(st)
        else:
            for _ in range(iters):
                st = iteration(st)
        jax.block_until_ready(st)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), int(j.checksum(st))


def run_cuda(binary, batch, iters, mode):
    out = subprocess.run([str(binary), str(batch), str(iters), mode], capture_output=True, text=True, check=True)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    return r['ms'] / 1000.0, int(r['checksum'], 16)


def build_cuda(out_dir):
    (HERE / 'microbench' / 'tables.h').write_text(T.c_header())
    binary = out_dir / 'gate_cuda'
    subprocess.run([NVCC, '-O3', '-arch=native', '-o', str(binary), str(HERE / 'microbench' / 'gate.cu')], check=True)
    return binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='one batch size, fewer iterations')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--out', type=pathlib.Path, default=HERE / 'results')
    ap.add_argument('--report', action='store_true', help='re-print the report from a previous run')
    args = ap.parse_args()
    if args.report:
        rows = json.loads((args.out / 'gate.json').read_text())
        return report(rows, [], sorted({r['batch'] for r in rows}))
    batches = (4096,) if args.quick else BATCHES
    iters = 50 if args.quick else args.iters
    args.out.mkdir(parents=True, exist_ok=True)
    binary = build_cuda(args.out)

    rows, mismatches = [], []
    for mode in MODES:
        cuda_mode = {'dense': 'dense', 'search': 'search', 'search_graph': 'graph'}[mode]
        for batch in batches:
            r = {}
            r['warp'] = run_warp(batch, iters, mode)
            r['jax'] = run_jax(batch, iters, mode)
            r['cuda'] = run_cuda(binary, batch, iters, cuda_mode)
            sums = {k: v[1] for k, v in r.items()}
            if len(set(sums.values())) != 1:
                mismatches.append((mode, batch, {k: f'{v:016x}' for k, v in sums.items()}))
            steps = batch * iters
            rows.append(dict(mode=mode, batch=batch, checksum=f'{sums["warp"]:016x}',
                             **{k: steps / v[0] for k, v in r.items()},
                             **{f'{k}_iter_us': v[0] / iters * 1e6 for k, v in r.items()}))
            print(f'{mode:13s} batch {batch:6d}  ' + '  '.join(f'{k} {rows[-1][k] / 1e6:8.1f}M/s' for k in ('warp', 'jax', 'cuda')))

    (args.out / 'gate.json').write_text(json.dumps(rows, indent=2) + '\n')
    report(rows, mismatches, batches)


def report(rows, mismatches, batches):
    get = lambda mode, batch: next(r for r in rows if r['mode'] == mode and r['batch'] == batch)
    print('\n== steps/s (millions), and Warp relative to the others ==')
    print(f'{"mode":13s} {"batch":>6s} {"warp":>9s} {"jax":>9s} {"cuda":>9s} {"warp/jax":>9s} {"warp/cuda":>10s}')
    for mode in MODES:
        for b in batches:
            r = get(mode, b)
            print(f'{mode:13s} {b:6d} {r["warp"]/1e6:9.1f} {r["jax"]/1e6:9.1f} {r["cuda"]/1e6:9.1f} '
                  f'{r["warp"]/r["jax"]:9.2f} {r["warp"]/r["cuda"]:10.2f}')

    big = [b for b in batches if b >= 16384]
    ratios = [get('dense', b)['warp'] / get('dense', b)['jax'] for b in big]
    cuda_ratio = [get('dense', b)['warp'] / get('dense', b)['cuda'] for b in big]
    # Dispatch overhead is what graph capture can remove: the gap between launching
    # the iteration's kernels from the host and replaying one captured graph. The
    # search shape's extra cost over a dense step is fork and gather traffic, real
    # work that no capture removes, so it is reported separately.
    print('\n== per-iteration cost of the search-shaped workload (Warp) ==')
    print(f'{"batch":>6s} {"no graph us":>12s} {"graph us":>10s} {"dispatch":>9s} {"vs dense":>9s}')
    overhead = {}
    for b in batches:
        ng, g = get('search', b)['warp_iter_us'], get('search_graph', b)['warp_iter_us']
        overhead[b] = max(0.0, (ng - g) / ng)
        print(f'{b:6d} {ng:12.1f} {g:10.1f} {overhead[b]:8.1%} '
              f'{get("dense", b)["warp"] / get("search_graph", b)["warp"]:8.2f}x')

    checks = [('Warp >= 2x JAX on dense at batch >= 16K', all(x >= 2.0 for x in ratios) if ratios else False),
              ('Warp within 1.5x of raw CUDA at batch >= 16K', all(x >= 1 / 1.5 for x in cuda_ratio) if cuda_ratio else False),
              ('graph capture leaves < 10% dispatch overhead at 16K', overhead.get(16384, 1.0) < 0.10),
              ('all three implementations agree', not mismatches)]
    print()
    for name, ok in checks:
        print(f'[{"PASS" if ok else "FAIL"}] {name}')
    for m in mismatches:
        print(f'  checksum mismatch: {m}')
    return all(ok for _, ok in checks)


if __name__ == '__main__':
    sys.exit(0 if main() is not False else 1)
