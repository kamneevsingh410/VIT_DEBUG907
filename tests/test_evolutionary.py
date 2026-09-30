from __future__ import annotations

import ast

from tools.evolutionary import rename_variables, reorder_statements

SRC = """import sys
from math import gcd

def solve(numbers, limit):
    total = 0  # running sum
    for value in numbers:
        if value <= limit:
            total += value
    print(total, sep=" ", file=sys.stdout)
    return gcd(total, limit)

count = int(input())
items = list(map(int, input().split()))
solve(items, count)
"""


def test_rename_keeps_structure_and_semantics():
    out = rename_variables(SRC)
    assert out and "total" not in out and "value" not in out
    assert "# running sum" in out
    assert "def solve(" in out and "gcd(" in out
    assert "sep=" in out and "file=sys.stdout" in out
    assert "print(" in out and "int(input())" in out
    ast.parse(out)


def test_rename_skips_unparseable_code():
    assert rename_variables("print 'python 2'\n") is None


def test_reorder_swaps_only_a_safe_pair():
    out = reorder_statements(SRC)
    assert out is not None and out != SRC
    assert out.index("from math import gcd") < out.index("import sys")
    ast.parse(out)


def test_reorder_refuses_dependent_or_side_effecting_statements():
    src = "a = int(input())\nb = int(input())\nc = a + 1\nd = c * 2\n"
    assert reorder_statements(src) is None


def test_style_preserving_rename_uses_short_unused_names():
    src = "n = int(input())\na = list(map(int, input().split()))\nprint(max(a) - n)\n"
    out = rename_variables(src, style="short")
    assert out and "var" not in out
    assert "n =" not in out and "a =" not in out
    ast.parse(out)
    names = {node.id for node in ast.walk(ast.parse(out)) if isinstance(node, ast.Name)}
    assert all(len(x) <= 2 for x in names - {"int", "input", "list", "map", "print", "max"})
