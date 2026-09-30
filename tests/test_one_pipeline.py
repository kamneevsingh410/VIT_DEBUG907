from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import cli
from bench import dataset

ROOT = Path(__file__).resolve().parent.parent


def test_load_never_falls_back_to_the_fixture(monkeypatch):
    def broken(limit=None):
        raise ConnectionError("offline")

    monkeypatch.setattr(dataset, "load_appsretrieval", broken)
    with pytest.raises(ConnectionError):
        dataset.load()


def test_fixture_announces_itself(capsys):
    corpus, _q, _r, source = dataset.load(fixture=True)
    assert "FIXTURE" in source
    assert "FIXTURE" in capsys.readouterr().err
    assert len(corpus) == 30


def test_only_selftest_touches_the_fixture():
    for name, fn in vars(cli).items():
        if not name.startswith("cmd_") or name == "cmd_selftest":
            continue
        src = inspect.getsource(fn)
        assert "load_sample" not in src and "fixture=True" not in src, name
        assert "prefer_real" not in src, name


def test_default_db_is_not_the_fixture():
    import app
    assert cli.DEFAULT_DB.resolve() not in {p.resolve() for p in app.FIXTURE_DBS}


def test_selftest_prints_fixture(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "check_environment", lambda: (False, "stubbed"))
    monkeypatch.setattr(cli, "SELFTEST_DB", tmp_path / "selftest.db")
    cli.main(["selftest"])
    assert "FIXTURE" in capsys.readouterr().out


def test_fixture_era_duplicates_are_gone():
    assert not (ROOT / "bench" / "ablation.py").exists()
    assert not (ROOT / "run_eval.py").exists()


def test_ablate_runs_the_real_table(monkeypatch):
    import tools.real_ablation as real_ablation
    seen = {}
    monkeypatch.setattr(real_ablation, "main", lambda argv=None: seen.setdefault("argv", argv) and 0)
    cli.main(["ablate"])
    assert "argv" in seen


def test_eval_and_reproduce_share_the_scoring_path():
    for fn in (cli.cmd_eval, cli.cmd_reproduce):
        assert "workflows.evaluate_shipped(" in inspect.getsource(fn), fn.__name__


def test_reindex_builds_the_shipped_view_only(monkeypatch, tmp_path):
    from retrieval import versions
    views = []
    real_build = versions.build

    def spy(*a, **k):
        views.append(k.get("views"))
        return real_build(*a, **k, use_embeddings=False)

    monkeypatch.setattr(versions, "build", spy)
    corpus = {"a": "def f():\n    return 1\n", "b": "def g():\n    return 2\n"}
    versions.reindex(corpus, dict(corpus, b="def g():\n    return 3\n"),
                     tmp_path / "v.db")
    assert views == [("code",), ("code",)]
