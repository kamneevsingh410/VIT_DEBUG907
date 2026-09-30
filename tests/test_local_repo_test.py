from __future__ import annotations

import json
import shutil
import subprocess

import pytest

import app
from tests.test_tool import FakeEmbedder, sandbox
from tools import local_repo_test as lrt

SENTINEL_PATH = "zebrapath"
SENTINEL_IDENT = "quokkaCompute"
SENTINEL_DOC = "calibrate the narwhal gauge before sunrise every morning"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture()
def target_repo(tmp_path):
    root = tmp_path / "knode"
    (root / SENTINEL_PATH).mkdir(parents=True)
    src = root / SENTINEL_PATH / "core.py"
    src.write_text(
        f"def {SENTINEL_IDENT}(x):\n    \"\"\"{SENTINEL_DOC}\"\"\"\n    return x * 2 + 1\n\n"
        f"def run(items):\n    return [{SENTINEL_IDENT}(i) for i in items] + [{SENTINEL_IDENT}(0)]\n",
        encoding="utf-8")
    (root / "app.js").write_text(
        "// load the settings file for the dashboard page\n"
        "function loadSettings (path) {\n  const raw = read(path)\n  return parse(raw)\n}\n",
        encoding="utf-8")
    (root / ".env").write_text("TOKEN=supersecretvalue", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one")
    src.write_text(src.read_text(encoding="utf-8") + "\ndef extra(y):\n    return y - 1 + y * 3\n",
                   encoding="utf-8")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "two")
    return root


@pytest.mark.skipif(shutil.which("git") is None,
                    reason="needs git (the Docker image has none; the harness is a dev tool)")
def test_local_repo_test_is_read_only_and_aggregate_only(target_repo, sandbox, tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(lrt, "OUT", tmp_path / "out")
    monkeypatch.setattr(lrt, "SCRATCH", tmp_path / "out" / "scratch")
    (tmp_path / "out").mkdir()
    before = sorted(p.relative_to(target_repo).as_posix() for p in target_repo.rglob("*")
                    if ".git" not in p.parts)
    manual = [{"query": "how is the gauge value doubled", "file": f"{SENTINEL_PATH}/core.py"}]
    report, details = lrt.run_repo("knode", target_repo, manual, FakeEmbedder())

    after = sorted(p.relative_to(target_repo).as_posix() for p in target_repo.rglob("*")
                   if ".git" not in p.parts)
    assert after == before
    assert git(target_repo, "status", "--porcelain") == ""
    assert report["versions"]["scratch_deleted"] is True

    committed = lrt.render([report], idle=False) + json.dumps(report)
    for secret in (SENTINEL_PATH, SENTINEL_IDENT, SENTINEL_DOC, "narwhal", "supersecret",
                   "gauge", "loadSettings", "dashboard"):
        assert secret not in committed, secret
    assert SENTINEL_IDENT in json.dumps(details)

    assert report["scan"]["skipped_files"].get("secret file") == 1
    assert report["auto"]["python_docstrings"]["kept"] == 1
    assert report["versions"]["tool"]["errors"] == 0
