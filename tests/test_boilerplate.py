from __future__ import annotations

import pytest

from pipeline.query_proc import (
    QueryConfig,
    process,
    strip_boilerplate,
)

REAL = (
    "Anton has the integer x. He is interested what positive integer, which "
    "doesn't exceed x, has the maximum sum of digits.\n\n"
    "-----Input-----\n\nThe first line contains the positive integer x.\n\n"
    "-----Output-----\n\nPrint the positive integer.\n"
)


def test_cuts_at_first_marker_exactly():
    got = strip_boilerplate(REAL)
    assert got.endswith("maximum sum of digits.")
    assert "-----Input-----" not in got
    assert "first line" not in got


@pytest.mark.parametrize("marker", [
    "-----Input-----", "-----Output-----", "-----Example-----",
    "-----Examples-----", "-----Note-----", "-----Constraints-----",
    "---Input---", "--------Input--------", "-----input-----",
    "-----INPUT-----",
])
def test_all_marker_spellings(marker):
    prose = "Find the shortest path between two nodes in a weighted graph."
    assert strip_boilerplate(f"{prose}\n\n{marker}\n\nblah") == prose


def test_only_the_first_marker_matters():
    prose = "Compute the greatest common divisor of two integers quickly."
    text = f"{prose}\n-----Input-----\na\n-----Output-----\nb\n-----Note-----\nc"
    assert strip_boilerplate(text) == prose


def test_no_marker_is_returned_unchanged():
    text = "Reverse a singly linked list in place without extra memory."
    assert strip_boilerplate(text) == text


def test_marker_at_the_very_start_is_not_stripped():
    text = "-----Input-----\n\nThe first line contains n, the array length."
    assert strip_boilerplate(text) == text


def test_empty_and_whitespace():
    assert strip_boilerplate("") == ""
    assert strip_boilerplate("   ") == "   "


def test_a_dashed_rule_is_not_a_marker():
    text = "Sort the array ----- then print it, keeping duplicates in order."
    assert strip_boilerplate(text) == text


def test_hyphenated_words_survive():
    text = "Find the breadth-first ordering of a directed acyclic graph now."
    assert strip_boilerplate(text) == text


def test_default_is_off_because_it_measured_harmful():
    assert QueryConfig().strip_boilerplate is False
    assert "-----Input-----" in process(REAL).embedding_text


def test_switch_on_strips_prose():
    p = process(REAL, QueryConfig(strip_boilerplate=True))
    assert p.embedding_text.endswith("maximum sum of digits.")
    assert "-----Input-----" not in p.embedding_text


def test_baseline_config_disables_stripping():
    assert QueryConfig.baseline().strip_boilerplate is False
    assert "-----Input-----" in process(REAL, QueryConfig.baseline()).embedding_text


def test_stripping_shrinks_the_query_substantially():
    p = process(REAL, QueryConfig(strip_boilerplate=True))
    assert len(p.embedding_text) < len(REAL) * 0.75
    assert len(p.embedding_text) > 20


def test_tokens_also_exclude_boilerplate_when_enabled():
    joined = " ".join(process(REAL, QueryConfig(strip_boilerplate=True)).tokens)
    assert "first" not in joined.split()
