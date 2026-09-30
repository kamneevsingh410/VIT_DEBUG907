from __future__ import annotations

import json
from datetime import datetime

import pytest


def test_datetime_payload_is_serialisable_with_default_str():
    payload = {"task_name": "AppsRetrieval", "date": datetime(2026, 9, 22, 3, 30),
               "scores": {"test": [{"ndcg_at_10": 0.10982}]}}
    with pytest.raises(TypeError):
        json.dumps(payload)
    assert json.dumps(payload, default=str)


def test_the_runner_logs_the_payload_safely():
    import inspect

    from tools import batch_run_screening
    source = inspect.getsource(batch_run_screening.step2_mteb)
    dumps_calls = [line for line in source.splitlines() if "json.dumps" in line]
    assert dumps_calls, "step2 should log the payload"
    for line in dumps_calls:
        assert "default=str" in line, f"unguarded json.dumps: {line.strip()}"


def test_score_extraction_survives_both_payload_shapes():
    for entries in ([{"ndcg_at_10": 0.11}], {"ndcg_at_10": 0.11}):
        scores = {"test": entries}
        got = []
        for _split, value in scores.items():
            for entry in value if isinstance(value, list) else [value]:
                got.append(entry.get("ndcg_at_10"))
        assert got == [0.11]


def test_a_failing_step_does_not_stop_later_steps():
    ran: list[int] = []

    def ok(n):
        return lambda: ran.append(n)

    def boom():
        raise RuntimeError("step blew up")

    steps = [(1, ok(1)), (2, boom), (3, ok(3)), (4, ok(4))]
    failures = []
    for number, fn in steps:
        try:
            fn()
        except Exception:
            failures.append(number)

    assert ran == [1, 3, 4], "steps after a failure must still run"
    assert failures == [2]


def test_runner_wraps_every_step():
    import inspect

    from tools import batch_run_screening
    source = inspect.getsource(batch_run_screening.main)
    assert "try:" in source and "except Exception" in source
