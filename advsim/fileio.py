"""The only module in the package that touches the filesystem.

Keeping reads and writes here means format rules (sorted JSON keys, compact
JSONL, uncompressed npz) are enforced once rather than at every call site.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Iterator

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'


def read_json(path: pathlib.Path) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: pathlib.Path, obj: Any) -> None:
    """Committed JSON: sorted keys, two-space indent, trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write('\n')


def iter_jsonl(path: pathlib.Path) -> Iterator[Any]:
    """Stream a JSONL file; pool.jsonl is over a gigabyte, so never read it whole."""
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_npz(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    """Uncompressed: 1M pool rows load in 0.18 s this way, 10x faster than compressed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        np.savez(f, **arrays)


def read_npz(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    """Hash array bytes, not the npz file: zip metadata must never change an ID."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def hash_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
