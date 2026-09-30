from __future__ import annotations

import hashlib
import inspect
import io
import math
import re
from pathlib import Path

import pytest

import app
import banner
from retrieval import workflows


class FakeEmbedder:

    name = "fake/test-encoder"
    max_seq_length = 1024
    quantize = False
    prefix = ""

    @property
    def signature(self) -> str:
        return "fake|seq1024|int8=False"

    def encode(self, texts, show_progress=False):
        return [self._vec(t) for t in texts]

    def encode_one(self, text):
        return self._vec(text)

    def warm(self):
        pass

    @staticmethod
    def _vec(text: str) -> list[float]:
        v = [0.0] * 32
        for tok in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 32] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    code = tmp_path / "project"
    code.mkdir()
    (code / "auth.js").write_text(
        "function checkPassword(user, password) {\n"
        "  // validate the password hash for this user\n"
        "  return hash(password) === user.passwordHash;\n}\n", encoding="utf-8")
    (code / "config.py").write_text(
        "def load_config(path):\n"
        "    \"\"\"read the yaml config file from disk\"\"\"\n"
        "    with open(path) as handle:\n"
        "        return yaml.safe_load(handle)\n", encoding="utf-8")
    (code / "math_utils.py").write_text(
        "def gcd(a, b):\n"
        "    \"\"\"greatest common divisor\"\"\"\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a\n", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(app, "OUT_DIR", out)
    monkeypatch.setattr(app, "DEFAULT_DB", out / "real.db")
    monkeypatch.setattr(app, "LAST_INDEX_FILE", out / "last_index.txt")
    return code, out


def run_tool(lines: list[str], db=None, cwd=None, embedder=None) -> tuple[int, str]:
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    stdout = io.StringIO()
    tool = app.Tool(db=db, stdin=stdin, stdout=stdout,
                    embedder=embedder or FakeEmbedder(), cwd=cwd)
    code = tool.run()
    return code, stdout.getvalue()


def indexed_db(code: Path, out: Path) -> Path:
    db = out / "project.db"
    workflows.index_folder(code, db, embedder=FakeEmbedder())
    return db


def test_banner_is_lowercase_ascii_and_at_most_80_columns():
    text = banner.render()
    assert all(len(line) <= 80 for line in text.splitlines())
    assert text.isascii()
    assert "natural-language code retrieval" in text
    assert "By Aadvik Chawla" in text and "Kamneev Singh" in text
    art = "\n".join(banner.ART)
    assert not re.search(r"[A-Z]", art), "letterforms must be lowercase"


def test_credits_sit_beneath_the_art_with_the_names_aligned():
    lines = banner.render().splitlines()
    last_art = max(i for i, ln in enumerate(lines) if ln.strip() and "|" in ln)
    first = next(i for i, ln in enumerate(lines) if "By Aadvik Chawla" in ln)
    assert first > last_art
    second = lines[first + 1]
    assert second.strip() == "Kamneev Singh"
    assert second.index("Kamneev") == lines[first].index("Aadvik")


def test_launch_reaches_the_prompt_and_quits(sandbox):
    code, out = sandbox
    rc, text = run_tool([":quit"], db=indexed_db(code, out))
    assert rc == 0
    assert "debug907>" in text
    assert "ready" in text and "3 snippets" in text
    assert text.rstrip().endswith("bye.")


def test_ctrl_d_end_of_input_exits_cleanly(sandbox):
    code, out = sandbox
    rc, text = run_tool([], db=indexed_db(code, out))
    assert rc == 0 and "bye." in text
    assert "Traceback" not in text


def test_empty_input_is_ignored_and_unknown_command_shows_help(sandbox):
    code, out = sandbox
    rc, text = run_tool(["", "   ", ":frobnicate", ":quit"], db=indexed_db(code, out))
    assert rc == 0
    assert "unknown command :frobnicate" in text
    assert ":open N" in text


def test_missing_index_asks_what_to_index_then_indexes_the_folder(sandbox):
    code, out = sandbox
    rc, text = run_tool(["1", "", ":quit"], cwd=code)
    assert rc == 0
    assert "What should I index?" in text
    assert f"(default: {code})" in text
    assert "indexed 3 snippets" in text
    assert "debug907>" in text


def test_quitting_at_the_index_question_exits_cleanly(sandbox):
    rc, text = run_tool(["q"], cwd=sandbox[0])
    assert rc == 0 and "bye." in text


def test_query_renders_rank_score_path_lines_confidence_and_latency(sandbox):
    code, out = sandbox
    rc, text = run_tool(["read the yaml config file", ":quit"], db=indexed_db(code, out))
    assert rc == 0
    assert re.search(r"\d+ results\s+\|\s+[\d.]+ ms\s+\|\s+confidence (high|medium|low)", text)
    assert re.search(r"^\s+1\.\s+\d\.\d{4}\s+config\.py:1-4\s+load_config", text, re.M)
    assert "safe_load" in text


def test_open_shows_the_full_snippet_and_more_pages(sandbox):
    code, out = sandbox
    rc, text = run_tool(["greatest common divisor", ":open 1", ":more", ":quit"],
                        db=indexed_db(code, out))
    assert rc == 0
    assert "return a" in text


def test_open_without_a_query_is_a_friendly_message(sandbox):
    code, out = sandbox
    rc, text = run_tool([":open 1", ":quit"], db=indexed_db(code, out))
    assert "run a query first" in text and "Traceback" not in text


def test_stats_reports_latency(sandbox):
    code, out = sandbox
    rc, text = run_tool(["gcd", "config", ":stats", ":quit"], db=indexed_db(code, out))
    assert "queries    2" in text and "p50" in text


def test_refuses_to_open_the_fixture(sandbox, monkeypatch):
    monkeypatch.setattr(app, "FIXTURE_DBS", {sandbox[1] / "index.db"})
    fixture = sandbox[1] / "index.db"
    workflows.index_folder(sandbox[0], fixture, embedder=FakeEmbedder())
    rc, text = run_tool([":quit"], db=fixture)
    assert rc == 1
    assert "FIXTURE" in text and "Traceback" not in text


def test_errors_are_one_line_not_a_traceback(sandbox):
    code, out = sandbox
    rc, text = run_tool([":index /definitely/not/a/folder", ":quit"],
                        db=indexed_db(code, out))
    assert "error:" in text and "try:" in text
    assert "Traceback" not in text


def test_tool_and_reproduce_share_one_search_path(sandbox, monkeypatch):
    import cli

    calls = []
    real = workflows.shipped_search

    def spy(*a, **k):
        calls.append(a[1])
        return real(*a, **k)

    monkeypatch.setattr(workflows, "shipped_search", spy)
    code, out = sandbox
    run_tool(["greatest common divisor", ":quit"], db=indexed_db(code, out))
    assert calls == ["greatest common divisor"]

    assert "workflows.shipped_search(" in inspect.getsource(cli.cmd_search)
    for fn in (cli.cmd_reproduce, cli.cmd_eval):
        assert "workflows.evaluate_shipped(" in inspect.getsource(fn)
    assert "shipped_search(" in inspect.getsource(workflows.evaluate_shipped)
    cfg = workflows.shipped_config()
    assert cfg.use_sparse is False and cfg.rerank.enabled is False
    assert cfg.view_weights == {"code": 1.0, "nl": 0.0, "raw": 1.0}


def test_confidence_thresholds_match_the_calibration():
    assert workflows.confidence([("a", 0.80), ("b", 0.75)])[0] == "high"
    assert workflows.confidence([("a", 0.80), ("b", 0.78)])[0] == "medium"
    assert workflows.confidence([("a", 0.80), ("b", 0.795)])[0] == "low"


class UnreadableStdin:

    def __init__(self):
        self.reads = 0

    def isatty(self):
        return False

    def readline(self):
        self.reads += 1
        if self.reads > 5:
            raise KeyboardInterrupt
        raise OSError("reading from stdin while output is captured")


def test_unreadable_stdin_exits_instead_of_spinning(sandbox):
    code, out = sandbox
    stdin = UnreadableStdin()
    tool = app.Tool(db=indexed_db(code, out), stdin=stdin, stdout=io.StringIO(),
                    embedder=FakeEmbedder())
    assert tool.run() == 1
    assert stdin.reads == 1


class SlowToLoadEmbedder(FakeEmbedder):

    def __init__(self, *a, **k):
        self.loaded = False

    def warm(self):
        if not self.loaded:
            import time
            time.sleep(0.6)
            self.loaded = True

    def encode(self, texts, show_progress=False):
        self.warm()
        return super().encode(texts)

    def encode_one(self, text):
        self.warm()
        return super().encode_one(text)


def test_search_latency_excludes_model_load(sandbox, monkeypatch, capsys):
    import cli
    code, out = sandbox
    db = indexed_db(code, out)
    monkeypatch.setattr(cli, "Embedder", SlowToLoadEmbedder)
    for track in ("a", "b"):
        assert cli.main(["search", "greatest common divisor", "--db", str(db),
                         "--track", track]) == 0
        text = capsys.readouterr().out
        total = float(re.search(r"latency\s+([\d.]+) ms total", text).group(1))
        assert total < 400, (track, total)
        assert "model load" in text


def two_version_db(tmp_path, out):
    v1 = tmp_path / "v1"; v2 = tmp_path / "v2"
    for folder in (v1, v2):
        folder.mkdir()
        (folder / "cfg.py").write_text(
            "def load_config(path):\n    \"\"\"read the yaml config file\"\"\"\n"
            "    return yaml.safe_load(open(path))\n", encoding="utf-8")
    (v1 / "auth.py").write_text(
        "def check_password(user, pw):\n    return md5(pw) == user.hash\n", encoding="utf-8")
    (v2 / "auth.py").write_text(
        "def check_password(user, pw):\n    return bcrypt.verify(pw, user.hash)\n", encoding="utf-8")
    db = out / "proj.db"
    workflows.index_folder(v1, db, version="v1", embedder=FakeEmbedder())
    workflows.index_folder(v2, db, version="v2", embedder=FakeEmbedder())
    return db


def test_version_command_lists_and_switches(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":version", ":version v1", "check password", ":quit"], db=db)
    assert rc == 0
    assert "versions   v1, v2" in text and "searching  v2" in text
    assert "searching  v1" in text
    assert "md5" in text and "bcrypt" not in text


def test_all_versions_collapses_and_shows_where_each_result_exists(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":all-versions", "read the yaml config file", ":quit"], db=db)
    assert rc == 0
    assert "searching  v1, v2" in text
    headings = re.findall(r"^\s+\d+\.\s+\d\.\d{4}\s+cfg\.py:\S+\s+load_config", text, re.M)
    assert len(headings) == 1, "one row per snippet across versions"
    assert "in v1 = v2" in text


def test_changed_function_is_marked_and_history_shows_the_diff(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":all-versions", "check the user password", ":history 1", ":quit"], db=db)
    assert rc == 0
    assert "in v1 != v2" in text
    assert "history of check_password: in 2 of 2 searched versions" in text
    assert "v1 -> v2" in text
    assert "-    return md5(pw) == user.hash" in text
    assert "+    return bcrypt.verify(pw, user.hash)" in text


def test_history_of_unchanged_code_says_identical(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":all-versions", "read the yaml config file", ":history 1", ":quit"], db=db)
    assert rc == 0 and "identical in every version" in text and " -> " not in text


def test_history_needs_a_query_and_a_valid_number(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":history 1", "read the yaml config file", ":history 99", ":quit"], db=db)
    assert "run a query first" in text and ":history takes a result number" in text


def test_newest_on_shows_the_newest_identical_version(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":all-versions", ":newest on", "read the yaml config file", ":quit"], db=db)
    assert rc == 0 and "newest     on" in text
    assert re.search(r"load_config.*\(v2; in v1 = v2\)", text)


def test_unknown_version_is_a_friendly_error(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":version v9", ":quit"], db=db)
    assert rc == 0 and "v9 is not in" in text and "Traceback" not in text


def test_old_or_damaged_index_is_a_friendly_error_not_an_exit(sandbox):
    import sqlite3
    code, out = sandbox
    db = indexed_db(code, out)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM meta WHERE key = 'schema'")
    conn.commit(); conn.close()
    rc, text = run_tool([":quit"], db=db)
    assert rc == 1
    assert "error:" in text and "try:" in text and "rebuild" in text


def test_collapse_can_be_switched_off_and_on(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":all-versions", ":collapse off", "read the yaml config file",
                         ":collapse on", "read the yaml config file", ":quit"], db=db)
    assert rc == 0
    off, on = text.split("collapse   on")
    heads = lambda t: re.findall(r"^\s+\d+\.\s+\d\.\d{4}\s+cfg\.py:\S+\s+load_config", t, re.M)
    assert len(heads(off)) == 2 and len(heads(on)) == 1
    assert re.search(r"distinct functions 2 of 4", off)
    assert re.search(r"distinct functions 2 of 2", on)


def test_version_accepts_a_comma_list(sandbox, tmp_path):
    db = two_version_db(tmp_path, sandbox[1])
    rc, text = run_tool([":version v1,v2", ":quit"], db=db)
    assert rc == 0 and "searching  v1, v2" in text


def test_repo_results_hide_tests_by_default_and_tests_on_shows_them(sandbox, tmp_path):
    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    body = "def parse_config(path):\n    return yaml.safe_load(open(path)) or {}\n"
    (root / "config.py").write_text(body, encoding="utf-8")
    (root / "tests" / "test_config.py").write_text(
        "def test_parse_config(tmp_path):\n    assert parse_config(tmp_path) == {}\n",
        encoding="utf-8")
    db = sandbox[1] / "p.db"
    workflows.index_folder(root, db, embedder=FakeEmbedder())
    rc, text = run_tool(["parse the config file", ":tests on", "parse the config file", ":quit"],
                        db=db)
    hidden, shown = text.split("tests      shown")
    assert "tests/test_config.py" not in hidden and "tests hidden" in hidden
    assert "tests/test_config.py" in shown


def test_is_test_path():
    for p in ["tests/test_a.py", "a/tests/b.py", "test_x.py", "x_test.py", "src/a.test.ts",
              "src/__tests__/b.js", "c.spec.tsx"]:
        assert workflows.is_test_path(p), p
    for p in ["src/testing_utils.py", "contest.py", "latest.js", "src/app.ts"]:
        assert not workflows.is_test_path(p), p


def test_confidence_ignores_copies_of_the_same_function_in_other_versions():
    hits = [("f@v1", 0.80), ("f@v2", 0.80), ("g@v1", 0.75)]
    identity = {"f@v1": "f", "f@v2": "f", "g@v1": "g"}
    label, margin = workflows.confidence(hits, identity)
    assert label == "high" and abs(margin - 0.05) < 1e-9
    assert workflows.confidence(hits)[0] == "low"


def test_status_line_shows_the_index_relative_not_the_users_home(sandbox, monkeypatch):
    code, out = sandbox
    db = indexed_db(code, out)
    monkeypatch.chdir(out.parent)
    rc, text = run_tool([":quit"], db=db.resolve())
    status = next(line for line in text.splitlines() if "ready" in line)
    assert str(out.parent) not in status
    assert "index out" in status


def test_hubness_command_switches_the_correction(sandbox, monkeypatch):
    code, out = sandbox
    seen = []
    real = workflows.shipped_search

    def spy(*a, **k):
        seen.append(k.get("hubness"))
        return real(*a, **k)

    monkeypatch.setattr(workflows, "shipped_search", spy)
    rc, text = run_tool(["check the password", ":hubness off", "check the password",
                         ":hubness on", "check the password", ":hubness auto", "check the password",
                         ":hubness maybe", ":quit"], db=indexed_db(code, out))
    assert rc == 0
    assert seen == [None, False, True, None]
    assert "hubness    off" in text and "usage: :hubness on | off | auto" in text
