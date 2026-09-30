from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from retrieval.index import IndexUnavailable

WORK_DIR = Path("out") / "sync_work"


class SyncError(RuntimeError):
    pass


@dataclass
class SyncReport:
    label: str
    commit: str
    skipped: bool
    snippets: int = 0
    encoded_vectors: int = 0
    reused_vectors: int = 0
    model_encoded: int = 0
    export_s: float = 0.0
    index_s: float = 0.0
    total_s: float = 0.0


def git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True)
    except FileNotFoundError:
        raise SyncError("git is not installed or not on PATH") from None
    except subprocess.CalledProcessError as exc:
        raise SyncError(exc.stderr.strip() or f"git {args[0]} failed") from None
    return done.stdout.strip()


def resolve(repo: Path, rev: str = "HEAD") -> tuple[str, str]:
    if not rev or rev.startswith("-"):
        raise SyncError(f"not a revision: {rev!r} (a revision cannot start with '-', which git "
                        "would read as an option)")
    full = git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")
    return full, git(repo, "rev-parse", "--short", full)


def check_top_level(repo: Path) -> None:
    try:
        top = git(repo, "rev-parse", "--show-toplevel")
    except SyncError:
        raise SyncError(f"not a git repository: {repo}") from None
    if Path(top).resolve() != Path(repo).resolve():
        raise SyncError(f"not a git repository: {repo} (it is inside {top}; "
                        "pass that folder to sync the whole repository)")


def synced_versions(db: Path) -> list[str]:
    import sqlite3
    if not Path(db).exists():
        return []
    conn = sqlite3.connect(f"file:{Path(db).resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute("SELECT version FROM versions ORDER BY built_at")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def wanted(path: str, include_vendored: bool = False, all_languages: bool = False) -> bool:
    from pipeline.repo import SKIP_DIR_REASONS, SOURCE_SUFFIXES, WINDOW_LANGUAGES, is_secret_file
    *dirs, name = path.split("/")
    for d in dirs:
        if d.startswith("."):
            return False
        for reason, names in SKIP_DIR_REASONS.items():
            if d in names and not (include_vendored and reason == "vendored dependencies"):
                return False
    if name == "pyvenv.cfg":
        return True
    if is_secret_file(name):
        return False
    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    return suffix in SOURCE_SUFFIXES or (all_languages and suffix in WINDOW_LANGUAGES)


def export_commit(repo: Path, commit: str, dest: Path, include_vendored: bool = False,
                  all_languages: bool = False) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    listing = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit],
                             capture_output=True)
    if listing.returncode != 0:
        raise SyncError(listing.stderr.decode("utf-8", "replace").strip() or "git ls-tree failed")
    blobs: list[tuple[str, str]] = []
    for entry in listing.stdout.split(b"\0"):
        if not entry:
            continue
        meta, raw_path = entry.split(b"\t", 1)
        mode, kind, sha = meta.split()
        path = raw_path.decode("utf-8", "surrogateescape")
        if kind == b"blob" and mode != b"120000" and wanted(path, include_vendored, all_languages):
            blobs.append((sha.decode(), path))
    if not blobs:
        return 0
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    root = dest.resolve()

    def feed() -> None:
        try:
            proc.stdin.write("".join(f"{sha}\n" for sha, _ in blobs).encode())
            proc.stdin.close()
        except OSError:
            pass

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        for sha, path in blobs:
            header = proc.stdout.readline().split()
            if len(header) != 3 or header[0].decode() != sha:
                raise SyncError(f"git cat-file: unexpected reply for {path}")
            data = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)
            target = (dest / path).resolve()
            if not target.is_relative_to(root):
                raise SyncError(f"unsafe path in commit: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        if proc.wait() != 0:
            raise SyncError(err or "git cat-file failed")
    return len(blobs)


def sync(repo: Path, db: Path, *, rev: str = "HEAD", embedder=None,
         include_vendored: bool = False, all_languages: bool = False,
         show_progress: bool = False) -> SyncReport:
    from pipeline.repo import extract_repo
    from retrieval.workflows import _build

    started = time.perf_counter()
    repo = Path(repo)
    check_top_level(repo)
    full, short = resolve(repo, rev)
    if short in synced_versions(db):
        return SyncReport(label=short, commit=full, skipped=True,
                          total_s=time.perf_counter() - started)
    work = WORK_DIR / short
    try:
        t = time.perf_counter()
        export_commit(repo, full, work, include_vendored=include_vendored,
                      all_languages=all_languages)
        export_s = time.perf_counter() - t
        t = time.perf_counter()
        corpus = extract_repo(work, include_vendored=include_vendored,
                              all_languages=all_languages)
        if not corpus:
            raise SyncError(f"no functions found in commit {short}")
        try:
            report = _build(corpus, db, short, embedder, show_progress,
                            f"git:{repo.resolve().name}@{full}", corpus_kind="repo")
        except IndexUnavailable as exc:
            raise SyncError(str(exc)) from None
        index_s = time.perf_counter() - t
    finally:
        shutil.rmtree(work, ignore_errors=True)
    st = report.stats
    return SyncReport(label=short, commit=full, skipped=False, snippets=report.snippets,
                      encoded_vectors=st.encoded_vectors, reused_vectors=st.reused_vectors,
                      model_encoded=st.model_encoded, export_s=export_s, index_s=index_s,
                      total_s=time.perf_counter() - started)


def watch(repo: Path, db: Path, *, interval: float = 30.0, say=print, max_polls: int | None = None,
          sleep=time.sleep, **kwargs) -> int:
    check_top_level(Path(repo))
    last = None
    polls = 0
    try:
        while max_polls is None or polls < max_polls:
            polls += 1
            try:
                head, _ = resolve(repo)
            except SyncError as exc:
                say(f"  cannot read HEAD: {exc}")
                head = last
            if head and head != last:
                say(describe(sync(repo, db, rev=head, **kwargs)))
                last = head
            if max_polls is None or polls < max_polls:
                sleep(interval)
    except KeyboardInterrupt:
        say("  watch stopped")
    return 0


def describe(r: SyncReport) -> str:
    if r.skipped:
        return f"  {r.label}  already synced - nothing to do ({r.total_s:.1f}s)"
    return (f"  {r.label}  {r.snippets:,} functions: {r.encoded_vectors:,} new or changed "
            f"({r.model_encoded:,} through the model), {r.reused_vectors:,} reused  |  "
            f"export {r.export_s:.1f}s + index {r.index_s:.1f}s = {r.total_s:.1f}s")
