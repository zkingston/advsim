"""State layout invariants, and the derived artifacts that must follow it."""

import pytest
from advsim import fileio
from advsim.engine import layout
from advsim.engine.dex import Dex


def test_every_field_is_unique_and_shaped():
    assert len(layout.BY_NAME) == len(layout.FIELDS)
    for f in layout.FIELDS:
        assert f.scope in layout.SHAPES, f.name
        assert f.width >= 1
        assert f.shape(7)[0] == 7


@pytest.mark.built
def test_layout_json_matches_the_module():
    path = fileio.ARTIFACTS / 'layout.json'
    assert fileio.read_json(path) == layout.as_json(), 'layout.json is stale; run `advsim build`'


@pytest.mark.built
def test_dex_struct_matches_the_built_arrays():
    arrays = fileio.read_npz(fileio.ARTIFACTS / 'dex.npz')
    assert set(Dex.vars) == set(arrays), 'engine/dex.py and dex.npz disagree on columns'
    for name, var in Dex.vars.items():
        assert arrays[name].ndim == var.type.ndim, f'{name}: npz is {arrays[name].ndim}D, struct is {var.type.ndim}D'
