from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVIL = ("def evil():\n"
        "    return '\x1b[2J\x1b[31mSYSTEM COMPROMISED\x1b[0m [red]fake[/red] "
        "\x1b]8;;http://attacker.example\x07click\x1b]8;;\x07 \x07\x08\x9b31m'\n")


def test_indexed_code_cannot_send_control_sequences_to_the_terminal(tmp_path, monkeypatch):
    from tests.test_tool import indexed_db, run_tool
    code = tmp_path / "project"
    code.mkdir()
    (code / "evil.py").write_text(EVIL, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.chdir(tmp_path)
    rc, text = run_tool(["return a fake message", "open 1", ":uses COMPROMISED",
                         "where is evil defined", "quit"], db=indexed_db(code, out))
    assert rc == 0
    assert "\x1b" not in text and "\x07" not in text and "\x9b" not in text and "\x08" not in text
    assert "SYSTEM COMPROMISED" in text
    assert "[red]fake[/red]" in text


def test_the_search_command_also_strips_control_sequences(tmp_path, capsys, monkeypatch):
    import cli
    from retrieval import workflows
    from tests.test_tool import FakeEmbedder
    code = tmp_path / "project"
    code.mkdir()
    (code / "evil.py").write_text(EVIL, encoding="utf-8")
    db = tmp_path / "evil.db"
    workflows.index_folder(code, db, embedder=FakeEmbedder())
    monkeypatch.setattr(cli, "Embedder", FakeEmbedder)
    cli.main(["search", "fake message", "--db", str(db)])
    printed = capsys.readouterr().out
    assert "\x1b" not in printed and "\x07" not in printed


def test_a_symlinked_file_pointing_outside_the_repo_is_not_indexed(tmp_path):
    from pipeline.repo import extract_repo
    outside = tmp_path / "helpers_elsewhere.py"
    outside.write_text("def private_key_loader(path):\n    with open(path) as handle:\n"
                       "        return handle.read().splitlines()\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ok.py").write_text("def fine(values):\n    total = 0\n    for v in values:\n"
                                "        total += v * 2\n    return total\n", encoding="utf-8")
    try:
        os.symlink(outside, repo / "link.py")
        os.symlink(repo, repo / "loop", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    corpus = extract_repo(repo)
    assert not any("private_key_loader" in t for t in corpus.values())
    assert any("fine" in sid for sid in corpus)


def test_rank_refuses_to_overwrite_the_shipped_artifact(tmp_path, monkeypatch, capsys):
    import cli
    before = (ROOT / "appsretrieval_results.json").read_bytes()
    q = tmp_path / "q.jsonl"
    q.write_text('{"id": "q1", "text": "x"}\n', encoding="utf-8")
    code = cli.main(["rank", "--queries", str(q), "--db", str(tmp_path / "none.db"),
                     "--out", str(ROOT / "appsretrieval_results.json")])
    assert code != 0
    assert (ROOT / "appsretrieval_results.json").read_bytes() == before
    assert "refusing" in capsys.readouterr().out


def test_outputs_never_overwrite_an_unrelated_existing_file(tmp_path, capsys):
    import cli
    precious = tmp_path / "notes.txt"
    precious.write_text("my notes\n", encoding="utf-8")
    q = tmp_path / "q.jsonl"
    q.write_text('{"id": "q1", "text": "x"}\n', encoding="utf-8")
    assert cli.main(["rank", "--queries", str(q), "--db", str(tmp_path / "none.db"),
                     "--out", str(precious)]) != 0
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"id": "a", "text": "def f(): pass"}\n', encoding="utf-8")
    assert cli.main(["index-corpus", str(corpus), "--out", str(precious)]) != 0
    assert precious.read_text(encoding="utf-8") == "my notes\n"


def test_sync_rejects_a_revision_that_looks_like_an_option(tmp_path):
    from retrieval import sync
    with pytest.raises(sync.SyncError, match="option"):
        sync.resolve(tmp_path, "--output=/tmp/pwned")


def test_the_docker_image_runs_as_a_non_root_user():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    users = re.findall(r"^USER\s+(\S+)", text, re.M)
    assert users and users[-1] not in {"root", "0"}


def test_symlinked_files_and_folders_are_skipped_by_the_walk(tmp_path, monkeypatch):
    from collections import Counter
    from pipeline import repo
    (tmp_path / "ok.py").write_text("def fine():\n    return 1\n", encoding="utf-8")
    (tmp_path / "link.py").write_text("def outside():\n    return 2\n", encoding="utf-8")
    (tmp_path / "linkdir").mkdir()
    (tmp_path / "linkdir" / "x.py").write_text("def inner():\n    return 3\n", encoding="utf-8")
    real = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name in {"link.py", "linkdir"} or real(self))
    skipped_files, skipped_dirs = Counter(), Counter()
    found = [p.name for p in repo.iter_source_files(tmp_path, skipped_files=skipped_files,
                                                     skipped_dirs=skipped_dirs)]
    assert found == ["ok.py"]
    assert skipped_files["symbolic link"] == 1 and skipped_dirs["symbolic link"] == 1
