"""The engine imports only Warp, the standard library bits `layout.py` needs,
and itself. This is the import rule SPEC §Project layout states."""
import ast

from advsim import fileio

ALLOWED = ('warp', 'advsim.engine', '__future__', 'dataclasses')


def test_the_engine_imports_only_warp_and_itself():
    bad = []
    for path in (fileio.ROOT / 'advsim' / 'engine').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            bad += [f'{path.name}: {n}' for n in names if not n.startswith(ALLOWED)]
    assert not bad, bad
