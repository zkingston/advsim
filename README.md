# advsim

A GPU battle engine for Pokémon Showdown's **[Gen 3] Random Battle**, written in
[NVIDIA Warp](https://github.com/NVIDIA/warp) and bit-exact with
[`pokemon-showdown`](https://github.com/smogon/pokemon-showdown) 0.11.11: same
damage, same random draws at the same call sites, same state at every decision
point. It runs thousands of battles at once for reinforcement learning and
test-time search.

**Play it:** [zkingston.com/advsim](https://zkingston.com/advsim/). The page runs
Showdown's simulator, the trained network and its search in your browser.

What is here:

- **The engine** (`advsim/engine/`): every move, ability and item in the
  format, one battle per GPU thread, integer arithmetic in Showdown's rounding
  order. Checked against Showdown on 100,000+ whole battles and 6.8 million
  decision points with zero divergences, 10 million damage cases, and perft
  over 2 million leaves ([docs/RESULTS.md](docs/RESULTS.md#parity-with-showdown)).
- **An RL environment** (`advsim/env.py`): batched `observe`/`step` with
  automatic resets, zero-copy PyTorch views and CUDA graph capture. A smoke
  PPO run reaches about 745K decisions/s on an RTX 4070.
- **Training** (`tools/train_league.py`): PPO in a league of self-play,
  past snapshots and five scripted baselines, with a transformer policy over
  the observation's tokens.
- **Search** (`advsim/search/`, `engine/tree.py`): simultaneous-move MCTS on
  the GPU over determinized worlds, with the network as prior and value.
- **An Elo bracket** (`tools/elo.py`) that rates any of these against each other.

The best player, search over the strongest network, rates 2275 where the
max-damage heuristic rates 1595 and random play 1000
([docs/RESULTS.md](docs/RESULTS.md#the-elo-bracket)).

## Requirements

- Linux with an NVIDIA GPU for training and search. The CPU runs everything
  else, including the test suite.
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.13).
- Node.js 22.12 or newer, for Showdown: building the data, the parity tests
  and the web page.

## Setup

```sh
git clone https://github.com/zkingston/advsim && cd advsim
uv sync --group dev --group rl                      # the engine, PyTorch, pytest
(cd showdown && npm ci --omit=optional)             # pokemon-showdown 0.11.11
node showdown/gen_pool.js                           # 1M teams -> artifacts/pool.jsonl (about 1 min on 32 cores)
uv run advsim build                                 # dex tables, the pool, generated engine code
uv run --group dev --group rl pytest -q             # about 155 tests
```

`gen_pool.js` with its defaults reproduces the pool the manifest records
(`artifacts/manifest.json`), so a build matches across machines. Without Node
or the build, the tests that need them skip.

Trained networks are downloads, not part of the repository. Put them in
`artifacts/`:

```sh
gh release download v0.1.0 -R zkingston/advsim -D artifacts
```

`ppo_dmg46.npz` is the strongest network (the one the page plays),
`ppo_long.npz` the reference `train_league.py` evaluates against, and
`ppo_baseline.npz` the first PPO player in the bracket.

## Using the engine

```python
import torch
from advsim.env import Gen3Env
from advsim.net import legal_bits

env = Gen3Env(batch=4096, device='cuda:0')
env.reset(seed=0)
for _ in range(200):
    obs, mask = env.observe()          # int16 [B, 2, OBS_DIM], int32 [B, 2] legal-action bitmasks
    legal = legal_bits(mask).float()   # [B, 2, 12]
    actions = torch.multinomial(legal.view(-1, 12), 1).view(-1, 2)  # random legal play
    env.step(actions.int())            # finished battles are dealt again
    print(env.done.sum().item(), 'battles ended; p1 rewards', env.reward[env.done.bool()])
```

Actions are 12 codes per side: four moves, six switches, pass and the forced
action (Struggle, a recharge, a locked move). The observation is only what the
protocol shows that player, laid out in `advsim/engine/obs_layout.py`.

## Training and rating

```sh
uv run --group rl python examples/ppo_smoke.py --iters 40                  # PPO against random play, 1 min
uv run --group rl python tools/train_league.py --anneal --minutes 120 --save artifacts/ppo_mine.npz
uv run --group rl python tools/elo.py --games 2048                         # the bracket of baselines
uv run --group rl python tools/elo.py --add ppo-t25:mine                   # rate a new player against it
```

`train_league.py` with no options trains the recipe that works (a compact
transformer, width 96, 3 layers). Warm-starting from a checkpoint (`--init`)
wants `--lr 1e-4`. Every tool explains its options under `--help`.

## Checking parity

```sh
uv run python tools/parity.py sweep --policy mix --seed 123456 --seeds 20 --battles 100 --turns 100 --jobs 20
uv run python tools/parity.py first --seed 123456          # the earliest divergence of each bad battle
```

Showdown is the oracle: `showdown/oracle.js` plays battles, logs every random
draw and exports its state, and the engine replays them and is judged by a
canonical hash at every decision point.

## The web page

```sh
node showdown/fetch_sprites.js     # the sprites, from Showdown's sprite server
node showdown/bundle.js            # web/dist/
python -m http.server -d web       # then open http://localhost:8000
```

`tools/export_web.py <tag>` writes a network for the page.

## Layout

| Path | What |
|---|---|
| `advsim/engine/` | The battle step, observation, search tree: pure Warp |
| `advsim/build/` | Showdown's dump to dex tables and generated engine code |
| `advsim/env.py` | The batched environment |
| `advsim/net.py`, `net_tf.py`, `damage.py` | Policy networks |
| `advsim/search/` | The search's host side |
| `advsim/live/` | The observation from a Showdown protocol stream, for live play |
| `showdown/` | Node scripts around `pokemon-showdown`: the oracle, the pool, the bundle |
| `tools/` | Training, the bracket, parity sweeps |
| `web/` | The browser page |
| `docs/SPEC.md` | The design |
| `docs/RESULTS.md` | What has been measured |

## License

MIT; see [LICENSE](LICENSE). Showdown's data and the logic ported from it are
also MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Pokémon and all related names are © Nintendo, Game Freak and Creatures Inc.
This is a non-commercial fan project, not affiliated with or endorsed by them
or by Pokémon Showdown.
