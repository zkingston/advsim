# showdown/: the oracle and the page's build

Node scripts around `pokemon-showdown` 0.11.11 (pinned in `package.json`; Node 22.12 or newer). The browser page's own JavaScript is in `web/js/`.

    cd showdown && npm ci --omit=optional

Run from the repo root. `advsim build` must have run first for anything that loads engine state (`oracle.js`, through `lib/export_state.js`, reads `artifacts/ids.json` and `layout.json`).

**The data**

| Command | Output | Notes |
| --- | --- | --- |
| `node showdown/dump_dex.js` | `artifacts/dex_raw.json` (committed) | Resolved `Dex.mod('gen3')` data plus `callbacks: {event: source}`, including `condition`, `secondaries`, `self` and `fling`, plus the gen3 `Scripts` overrides and the `gen3randombattle` rule table. |
| `node showdown/gen_pool.js` | `artifacts/pool.jsonl` | 1M teams from seeded `Teams.getGenerator`s, one per line, with Showdown's `storedStats`, `maxhp`, resolved gender, `hpType` and max PP. The defaults (`--n 1000000 --seed 1 --shards 32`) give the pool the manifest records. |
| `node showdown/trace.js --games 1000 --policy random\|greedy --seed 1 [--verify 100]` | `traces/<policy>-<seed>.jsonl`, `artifacts/catalog.json` (committed) | Raw `rng.next()` draws with call-site keys, choices, and the protocol log without `\|t:\|` lines. |

**The oracle.** `oracle.js` is a JSONL server that `advsim/oracle.py` runs: one command per stdin line, results one per stdout line, each command closed by `{"done": true}`. Commands: `damage` (L1), `turn` (whole battles for the sweep), `prng`, `scenario` (one L2 scenario), `perft`, `emerald`, and `observe`/`forward` (the page's converter and network, for `tests/test_web.py`). Its parts are in `lib/`: `export_state.js` (a battle as engine state), `scripted_rng.js` (Showdown's PRNG, logged and steerable by call site), `scenario.js`, `perft.js`, `play.js` (the sweep's policies), `policy.js` and `emerald.json` (the Emerald replacement chooser), `estimate.js` (the max-damage move choice) and `web.js` (the page's modules under Node).

**The web page**

| Command | Output |
| --- | --- |
| `node showdown/bundle.js` | `web/dist/`: the simulator, the page's modules, its stylesheet and the search worker |
| `node showdown/fetch_sprites.js` | `web/sprites/` (not committed: game images) |
| `node showdown/web_text.js` | `web/data/text.json`, the tooltips' gen 3 descriptions |
| `node showdown/build_wasm.js` | `web/js/matmul_wasm.js` from `web/wasm/matmul.c` (needs clang) |
| `node showdown/check_bundle.js [battles] [seed]` | the bundle against the package, log for log |
| `node showdown/check_mcts.js time\|leak\|match` | the page's search: cost, hidden-information leaks, strength |

**Determinism.** Every seed is `sodium,sha256("<seed>:<shard-or-game>:<tag>")`, so output depends only on the arguments (for the pool, `--seed` and `--shards`: each shard is its own stream). Teams, battle and policy draws use separate seeds; the policy never perturbs battle draws. `dump_dex.js` stamps the time it ran; otherwise the same arguments reproduce a file byte for byte.

**Self-checks** (each script fails loudly rather than writing a bad artifact):

- `dump_dex.js` throws if a callback appears under an unexpected key, or if any dumped callback source no longer parses.
- `gen_pool.js` regenerates shard 0 in-process and compares it against the file head.
- `trace.js` replays the first `--verify` games from their logged draws and requires an identical log; it also fails if any setup gender draw happens, or if a game lacks the per-turn Quick Claw roll.

**Call-site catalog.** `artifacts/catalog.json` maps a site key (`<effect>:<frame><<frame>`) to `{name, fn, args, count}`. `name` is the scenario `force` name: a short alias for the common sites (`quickclaw`, `crit`, `accuracy`, `damage_roll`, `secondary`, `speedtie`, …) and `<effect>_<handler>` otherwise. Counts are per policy and are replaced on each run of that policy.
