"""Manifest of built artifacts, with content hashes loaders refuse to ignore."""
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from advsim import fileio

MANIFEST = fileio.ARTIFACTS / 'manifest.json'
SCHEMA = 1


def write_manifest(entries: dict[str, Any], showdown: str) -> dict:
    """Merge `entries` into the manifest: a dex rebuild must not drop the pool's hash."""
    old = fileio.read_json(MANIFEST) if MANIFEST.exists() else {}
    if old.get('schema') != SCHEMA:
        old = {}
    manifest = {**old, 'schema': SCHEMA, 'showdown': showdown, **entries}
    fileio.write_json(MANIFEST, manifest)
    return manifest


def read_manifest() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError(f'{MANIFEST} missing; run `advsim build`')
    m = fileio.read_json(MANIFEST)
    if m.get('schema') != SCHEMA:
        raise ValueError(f'manifest schema {m.get("schema")} != {SCHEMA}; rebuild')
    return m


def load_dex(path: pathlib.Path | None = None) -> dict[str, np.ndarray]:
    """Load dex.npz and refuse it if the array bytes do not match the manifest."""
    path = path or fileio.ARTIFACTS / 'dex.npz'
    arrays = fileio.read_npz(path)
    want = read_manifest().get('dex', {}).get('sha256')
    got = fileio.hash_arrays(arrays)
    if want and want != got:
        raise ValueError(f'{path.name} hash {got[:12]} != manifest {want[:12]}; rebuild')
    return arrays


def load_ids(path: pathlib.Path | None = None) -> dict[str, list[str]]:
    path = path or fileio.ARTIFACTS / 'ids.json'
    ids = fileio.read_json(path)
    want = read_manifest().get('ids', {}).get('sha256')
    got = fileio.hash_json(ids)
    if want and want != got:
        raise ValueError(f'{path.name} hash {got[:12]} != manifest {want[:12]}; rebuild')
    return ids
