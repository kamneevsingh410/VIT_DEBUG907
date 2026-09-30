from __future__ import annotations

from pathlib import Path

import pytest

from cli import DEFAULT_DB, build_parser, main


@pytest.fixture(scope="module")
def parser():
    return build_parser()


@pytest.mark.parametrize("command", ["interactive", "index"])
def test_db_accepted_after_subcommand(parser, command):
    assert parser.parse_args([command, "--db", "b.db"]).db == Path("b.db")


def test_db_accepted_after_search_subcommand(parser):
    args = parser.parse_args(["search", "some query", "--db", "b.db"])
    assert args.db == Path("b.db") and args.query == "some query"


def test_db_accepted_before_subcommand(parser):
    assert parser.parse_args(["--db", "a.db", "interactive"]).db == Path("a.db")


def test_db_defaults_when_absent(parser):
    assert parser.parse_args(["interactive"]).db == DEFAULT_DB


def test_index_repo_defaults_to_eight_threads(parser):
    assert parser.parse_args(["index-repo", "some/path"]).threads == 8


def test_search_defaults_to_the_shipped_track(parser):
    assert parser.parse_args(["search", "q"]).track == "a"


def test_bare_invocation_launches_the_tool(monkeypatch):
    import app
    seen = {}
    monkeypatch.setattr(app, "main", lambda db=None, debug=False, versions=None: seen.update(db=db) or 0)
    assert main([]) == 0
    assert seen == {"db": None}, "the fixture DEFAULT_DB must not be forwarded"


def test_help_prints_banner_and_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "By Aadvik Chawla" in out and "Kamneev Singh" in out
    assert "index-repo" in out and "reproduce" in out


def test_reproduce_is_registered_and_takes_db(parser):
    args = parser.parse_args(["reproduce", "--db", "out/x.db"])
    assert args.command == "reproduce" and args.db == Path("out/x.db")
    assert args.threads == 8


def test_index_has_thread_limit(parser):
    assert parser.parse_args(["--real", "index"]).threads == 8


def test_reproduce_target_is_the_recorded_sample_number():
    import cli
    assert cli.EXPECTED_SAMPLE_NDCG == 0.6206
    assert cli.REPRODUCE_TOLERANCE <= 0.005


def test_eval_limit_accepted_after_subcommand(parser):
    assert parser.parse_args(["eval", "--limit", "5"]).limit == 5
    assert parser.parse_args(["--limit", "7", "eval"]).limit == 7
    assert parser.parse_args(["eval"]).limit is None


def test_search_and_tool_take_version_flags(parser):
    a = parser.parse_args(["search", "q", "--version", "v1"])
    assert a.version == "v1" and a.all_versions is False
    assert parser.parse_args(["search", "q", "--all-versions"]).all_versions is True
    b = parser.parse_args(["interactive", "--all-versions"])
    assert b.all_versions is True and b.version is None


def test_reproduce_full_is_registered(parser):
    assert parser.parse_args(["reproduce", "--full"]).full is True
    assert parser.parse_args(["reproduce"]).full is False


def test_full_reproduce_targets_are_the_shipped_artifact():
    import cli
    targets = cli.full_split_targets()
    assert targets == {"ndcg_at_10": 0.60952, "mrr_at_10": 0.5632}


def test_full_reproduce_runs_the_official_mteb_path():
    import inspect
    import cli
    assert "reproduce_full_mteb(" in inspect.getsource(cli.cmd_reproduce)
    helper = inspect.getsource(cli.reproduce_full_mteb)
    assert "run_mteb.main(" in helper and "compare_mteb_result(" in helper
    assert 'Path("out/' in helper, "the MTEB run must write into out/, not the artifact"


def test_compare_mteb_result(tmp_path):
    import json
    import cli
    ok = {"scores": {"test": [{"ndcg_at_10": 0.60952, "mrr_at_10": 0.5632031}]}}
    bad = {"scores": {"test": [{"ndcg_at_10": 0.53926, "mrr_at_10": 0.5632031}]}}
    (tmp_path / "ok.json").write_text(json.dumps(ok), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    assert cli.compare_mteb_result(tmp_path / "ok.json")[0] is True
    assert cli.compare_mteb_result(tmp_path / "bad.json")[0] is False


def test_search_include_tests_flag(parser):
    assert parser.parse_args(["search", "q"]).include_tests is False
    assert parser.parse_args(["search", "q", "--include-tests"]).include_tests is True


def test_explicit_db_always_wins_even_when_it_is_the_default(monkeypatch):
    import app
    seen = {}
    monkeypatch.setattr(app, "main", lambda db=None, debug=False, versions=None: seen.update(db=db) or 0)
    for argv in (["--db", str(DEFAULT_DB)], ["interactive", "--db", str(DEFAULT_DB)]):
        seen.clear()
        assert main(argv) == 0
        assert seen["db"] == DEFAULT_DB, argv


def test_search_can_switch_off_the_hubness_correction(parser):
    assert parser.parse_args(["search", "q", "--no-hubness"]).no_hubness is True
    assert parser.parse_args(["search", "q"]).no_hubness is False


def test_run_mteb_other_task_never_uses_the_apps_hub_table(monkeypatch):
    import run_mteb
    seen = {}
    monkeypatch.setattr(run_mteb, "run_mteb", lambda args: seen.update(vars(args)) or 0)
    assert run_mteb.main(["--task", "CosQA", "--out", "out/x.json"]) == 0
    assert seen["task"] == "CosQA" and seen["no_hubness"] is True
    run_mteb.main(["--out", "out/x.json"])
    assert seen["task"] == "AppsRetrieval" and seen["no_hubness"] is False


def test_run_mteb_ships_raw_documents_and_can_reproduce_the_code_view(monkeypatch):
    import run_mteb
    seen = {}
    monkeypatch.setattr(run_mteb, "run_mteb", lambda args: seen.update(vars(args)) or 0)
    run_mteb.main(["--out", "out/x.json"])
    assert seen["raw_docs"] is True and seen["hubness"].name == "apps_hubness.json"
    run_mteb.main(["--code-view", "--out", "out/x.json"])
    assert seen["raw_docs"] is False and seen["hubness"].name == "apps_hubness_codeview.json"


def test_the_shipped_hub_table_is_for_raw_documents():
    import json
    from pathlib import Path
    table = json.loads(Path("data/apps_hubness.json").read_text(encoding="utf-8"))
    assert table["document_text"] == "raw" and len(table["by_id"]) == 8765
    old = json.loads(Path("data/apps_hubness_codeview.json").read_text(encoding="utf-8"))
    assert old["document_text"] == "code"
