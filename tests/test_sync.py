from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from retrieval import sync
from retrieval.index import load
from tests.test_vector_keys import TextEmbedder

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def run(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", message)
    return run(repo, "rev-parse", "--short", "HEAD")


def snapshot(repo: Path) -> dict[str, str]:
    return {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(repo.rglob("*")) if p.is_file()}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "WORK_DIR", tmp_path / "work")
    r = tmp_path / "proj"
    r.mkdir()
    run(r, "init", "-q")
    commit(r, {"cfg.py": "def load_config(path):\n    return yaml.safe_load(open(path))\n",
               "auth.py": "def check_password(user, pw):\n    return md5(pw) == user.hash\n",
               "node_modules/dep/index.js": "function dep() { return 1 }\n"}, "one")
    return r


def test_sync_indexes_the_commit_as_its_short_hash_and_never_writes_to_the_repo(repo, tmp_path):
    (repo / "auth.py").write_text("def check_password(user, pw):\n    return True  # WIP\n",
                                  encoding="utf-8")
    before = snapshot(repo)
    db = tmp_path / "c.db"
    report = sync.sync(repo, db, embedder=TextEmbedder())
    assert snapshot(repo) == before, "sync must not write anything into the repository"
    assert not report.skipped and report.label == run(repo, "rev-parse", "--short", "HEAD")
    index = load(db)
    try:
        assert index.loaded_versions == [report.label]
        text = "\n".join(index.content(u) for u in index.ids)
        assert "md5" in text and "WIP" not in text
        assert not any("node_modules" in u for u in index.ids)
    finally:
        index.close()
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())


def test_next_commit_is_incremental_and_resyncing_is_a_no_op(repo, tmp_path):
    db = tmp_path / "c.db"
    emb = TextEmbedder()
    first = sync.sync(repo, db, embedder=emb)
    second_label = commit(repo, {"auth.py": "def check_password(user, pw):\n"
                                            "    return bcrypt.verify(pw, user.hash)\n"}, "two")
    second = sync.sync(repo, db, embedder=emb)
    assert second.label == second_label != first.label
    assert second.encoded_vectors == 1 and second.reused_vectors == 1
    assert sync.synced_versions(db) == [first.label, second.label]
    again = sync.sync(repo, db, embedder=emb)
    assert again.skipped and "already synced" in sync.describe(again)


def test_sync_an_older_revision_by_name(repo, tmp_path):
    first = run(repo, "rev-parse", "--short", "HEAD")
    commit(repo, {"net.py": "def fetch(url):\n    return get(url)\n"}, "two")
    report = sync.sync(repo, tmp_path / "c.db", rev="HEAD~1", embedder=TextEmbedder())
    assert report.label == first and report.snippets == 2


def test_watch_syncs_each_new_head_by_polling(repo, tmp_path):
    db = tmp_path / "c.db"
    said: list[str] = []
    labels = []

    def fake_sleep(_seconds):
        labels.append(commit(repo, {"net.py": f"def fetch_{len(labels)}(url):\n"
                                              "    return get(url)\n"}, "more"))

    sync.watch(repo, db, interval=0, say=said.append, max_polls=3, sleep=fake_sleep,
               embedder=TextEmbedder())
    assert len(sync.synced_versions(db)) == 3
    assert sync.synced_versions(db)[1:] == labels
    assert not (repo / ".git" / "hooks" / "post-commit").exists()


def test_not_a_git_repository_is_a_clear_error(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(sync.SyncError, match="not a git repository"):
        sync.sync(plain, tmp_path / "c.db", embedder=TextEmbedder())


def test_a_folder_inside_another_repository_is_refused(repo, tmp_path):
    sub = repo / "pkg"
    sub.mkdir()
    with pytest.raises(sync.SyncError, match="it is inside"):
        sync.sync(sub, tmp_path / "c.db", embedder=TextEmbedder())


def test_only_indexable_files_are_exported_and_secrets_never_touch_disk(repo, tmp_path):
    commit(repo, {".env": "API_KEY=abc\n", "keys/server.pem": "-----BEGIN\n",
                  "test/fixtures/big.json": "{}\n", "dist/app.js": "function built() {}\n",
                  "src/util.js": "function util() { return 1 }\n"}, "more")
    dest = tmp_path / "export"
    head = run(repo, "rev-parse", "HEAD")
    sync.export_commit(repo, head, dest)
    written = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert written == ["auth.py", "cfg.py", "src/util.js"]
    sync.export_commit(repo, head, dest, include_vendored=True)
    assert (dest / "node_modules/dep/index.js").exists()


def test_export_of_many_files_does_not_deadlock(repo, tmp_path):
    commit(repo, {f"src/m{i}.js": f"function f{i}() {{ return '{'x' * 400}' }}\n"
                  for i in range(2000)}, "many")
    head = run(repo, "rev-parse", "HEAD")
    assert sync.export_commit(repo, head, tmp_path / "export") == 2002
