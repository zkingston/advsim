# Contributing

Setup is in the [README](README.md). The design is in [docs/SPEC.md](docs/SPEC.md),
the source of truth: read the sections a change touches before making it.

## Workflow

1. Small commits, one concern each. Run the tests before committing
   (`uv run --group dev --group rl pytest -q`).
2. If the code has to contradict the SPEC, change the SPEC in the same commit.
3. After any change to the engine, sweep a seed range nobody has swept
   (`tools/parity.py sweep --policy mix --seed <new> --turns 100`). A clean
   range says nothing about the next one.
4. After any change to `engine/obs.py` or `advsim/live/`, run the converter
   parity (`tests/test_obs_parity.py`) on a new seed range too.

## Rules: parity

- Showdown 0.11.11 is the oracle, bug for bug, not the cartridge. The npm
  version is pinned exactly.
- Never claim parity without a Showdown comparison (hash streams or logs).
  Compare logs with `|t:|` lines removed.
- Every random draw goes through `rng_u32`, at Showdown's call sites, in
  Showdown's order: the Gen 3 Quick Claw roll in every `endTurn`, `sample()`
  on one-element lists, `speedSort` tie shuffles. No pre-drawing.
- Pool teams carry explicit gender so Showdown makes no setup draws; replay
  starts at the first in-battle draw.
- Stats come from Showdown (`storedStats`, `maxhp`), not from a stat formula.
  The one exception is Transform's sorting Speed, which Showdown itself
  recomputes from the copied base stats (`move_effects.py`).

## Rules: engine

- `advsim/engine/` is pure Warp: no NumPy, no I/O. Kernels live only in
  `engine/kernels.py` (the step and everything that calls it) and
  `engine/kernels_search.py` (search, which never calls the step); other
  engine files export `@wp.func`s. Both set `enable_backward=False`.
- Battle logic is integer-only, in Showdown's rounding order. Floats appear
  only in the search tree and the reward.
- `wp.constant` is for scalars, never for tables: tables are arrays in the
  `Dex` struct.
- State fields are defined once, in `engine/layout.py`; everything else
  derives from it. A field owned by an off flag must be zero.
- `engine/_generated/` is written only by `advsim build` (`build/codegen.py`).
- Behaviour is data: effect families plus parameters. No effect VM.

## Rules: files and structure

- JSON for anything a person edits, Node reads or the browser sees; npz for
  numeric arrays only Python reads; JSONL for append-only streams. No YAML.
- A u64 is never a JSON number.
- ID 0 means none. Arrays are fixed-width and 0-padded.
- Python file I/O only through `advsim/fileio.py`; Python talks to Node only
  through `advsim/oracle.py`. JavaScript lives in `showdown/` (Node) and
  `web/` (browser).
- About 300 lines per file, one concept per file. The known exception is
  `engine/execute.py`: `execute_move` follows Showdown's `tryMoveHit` as one
  flow.

## What the parity work taught

- Compare draw counts before state. A wrong count names a missing call site;
  equal counts with a wrong state mean a value was read as the wrong roll.
- `tools/parity.py sweep` counts mismatches at every decision point; once a
  battle diverges every later point counts again, so treat totals as a signal
  and use `first` to find bugs. Long battles are where PP bugs live: sweep
  with `--turns 100`.
- `ADVSIM_SORT_DEBUG=1` logs Showdown's residual sorts. Reading `speedSort`
  is not enough: it is a selection sort whose swaps scatter the tail.
- An ability with no callbacks can still matter, because another entry reads
  it by name (Early Bird, Truant, Guts). The build fails on one left unmapped.
- An L2 scenario may only use the built vocabulary, and every `force` it names
  must fire: one that never fires means the position never reached the site.

## What Warp taught

- `numpy()` on a CPU Warp array is a view: copy before comparing, or a test
  compares an array with itself. Perturb the input and check a test fails
  before believing it.
- Build time is per module and set by what nvcc inlines: every kernel that
  calls the step gets its own copy, so `step_idx` is the only one.
- Warp unrolls any `range(k)` with a constant k <= 16. In copied or nested
  code that explodes (one function reached 10,000 lines of CUDA): bound such
  loops by an array's shape. Time the build; generated line counts do not show
  the inlining.
- A `while` whose condition mutates an `int` assigned from `wp.where` raised a
  floating-point exception on the CPU; a bounded `for` works.
- Writing past a Warp vector corrupts the heap rather than failing.
- Share one device copy of the 240 MB pool per device (`load_pool`).

## What the web page taught

- After a checkpoint or observation-layout change, rerun
  `tools/export_web.py <tag>` and `tests/test_web.py`, which holds the JS
  converter and network to the Python ones. The page plays the first tag in
  `web/models/index.json`.
- The forward pass is the page's search cost. Time it alone, before and after
  any change. `web/wasm/matmul.c` is compiled by `node showdown/build_wasm.js`
  (clang, no wasm linker) into `web/js/matmul_wasm.js`, which is committed so
  CI needs no clang; without SIMD the page falls back to a JS loop.
- Showdown's `State` names objects by class, so the bundle needs esbuild's
  `keepNames`; and `State.deserializeBattle` hands the copy the root's own log
  array, so every search world needs `world.log = []`.
- Page action codes index the own party in the first request's order: build
  an InfoState by replaying every view, never from one request, which lists
  the party as switches left it.
- `node showdown/check_bundle.js N` and `node showdown/check_mcts.js time|leak|match`
  check the bundle against the package and the page's search.
