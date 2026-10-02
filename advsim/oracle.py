"""The only bridge between Python and Showdown.

Runs `showdown/oracle.js` as a subprocess and speaks one JSON object per line.
Nothing else in the package may spawn Node, so there is one place to look when
the oracle's protocol changes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Iterator

from advsim import fileio

SCRIPT = fileio.ROOT / 'showdown' / 'oracle.js'
NODE_PATH = fileio.ROOT / 'showdown' / 'node_modules'


class Oracle:
    """A running Showdown process. Use as a context manager."""

    def __init__(self) -> None:
        if not NODE_PATH.exists():
            raise FileNotFoundError(f'{NODE_PATH} missing; run `npm ci` in showdown/')
        self.node = os.environ.get('ADVSIM_NODE') or shutil.which('node')
        if not self.node:
            raise FileNotFoundError('node not found on PATH; install Node >= 20 or set ADVSIM_NODE')
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> 'Oracle':
        # The environment is fixed so a run cannot pick up anything local,
        # except the ADVSIM_ debug switches, which oracle.js reads. stderr is
        # inherited: Showdown writes nothing there unless something breaks.
        env = {'NODE_PATH': str(NODE_PATH), 'PATH': '/usr/bin:/bin'}
        env.update({k: v for k, v in os.environ.items() if k.startswith('ADVSIM_')})
        self.proc = subprocess.Popen(
            [self.node, str(SCRIPT)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)

    def request(self, **command: Any) -> Iterator[dict]:
        """Send one command; yield result objects until the terminating `done`."""
        assert self.proc, 'use Oracle as a context manager'
        self.proc.stdin.write(json.dumps(command) + '\n')
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            obj = json.loads(line)
            if 'error' in obj:
                raise RuntimeError(f'oracle: {obj["error"]} {obj.get("stack", "")}')
            if obj.get('done'):
                return
            yield obj
        raise RuntimeError(f'oracle: node exited ({self.proc.wait()}) during {command.get("cmd")}; its stderr is above')


def _run(**command: Any) -> list[dict]:
    with Oracle() as oracle:
        return list(oracle.request(**command))


def damage_cases(n: int, seed: int = 1) -> list[dict]:
    return _run(cmd='damage', n=n, seed=seed)


def turn_cases(n: int, seed: int = 1, turns: int = 1, policy: str = 'random',
               protocol: bool = False) -> list[dict]:
    """Whole battles: the state before, then per decision point the choices,
    every draw, the state after and the legal masks. `policy` names how both
    sides play: random, emerald, maxdamage, status, switchaverse, or mix.
    `protocol` adds the full log and, per decision point, the log position
    and both request JSONs: what each player's client had."""
    return _run(cmd='turn', n=n, seed=seed, turns=turns, policy=policy, protocol=protocol)


def emerald(cases: list[dict]) -> tuple[list[dict], list[str]]:
    """The Emerald chooser's Section 1 and Section 2 picks on hand-built positions,
    and where its type table disagrees with Showdown's gen 3 chart."""
    *picks, last = _run(cmd='emerald', cases=cases)
    return picks, last['mismatches']


def perft(seed: int, turns: int = 10, depth: int = 1, budget: int = 100000, stride: int = 1,
          chunk: int = 5000) -> Iterator[list[dict] | int]:
    """Every leaf of one perft position as replay cases, in chunks as Showdown
    finds them, so a position with 10^5 leaves never sits in memory at once;
    last, the depth-1 leaf count (0 when the battle ends before the position)."""
    with Oracle() as oracle:
        batch = []
        for obj in oracle.request(cmd='perft', seed=seed, turns=turns, depth=depth, budget=budget, stride=stride):
            if 'before' not in obj:
                depth1 = obj['depth1']
                continue
            batch.append(obj)
            if len(batch) == chunk:
                yield batch
                batch = []
        if batch:
            yield batch
        yield depth1


def prng_cases(n: int, seed: int = 1) -> list[dict]:
    """Raw draws with the value each of Showdown's mappings produced from them."""
    return _run(cmd='prng', n=n, seed=seed)


def web_observe(cases: list[dict]) -> list[list]:
    """The page's converter (web/js/) on protocol cases: per case, per view
    (the start, then every segment), both players' obs vectors."""
    with Oracle() as oracle:
        return [list(oracle.request(cmd='observe', case=c))[0]['obs'] for c in cases]


def web_forward(model: str, obs: list[list[int]]) -> dict:
    """The page's network (web/models/<model>.json) on obs rows: logits and values."""
    return _run(cmd='forward', model=model, obs=obs)[0]
