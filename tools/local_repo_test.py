from __future__ import annotations

import argparse
import ast
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.repo_eval import pseudo_queries
from pipeline.repo import scan_repo
from retrieval import workflows
from retrieval.index import build, load

OUT = Path("out")
SCRATCH = OUT / "scratch"
RESULTS = Path("bench/results")
MIN_WORDS = 5


def python_docstring_queries(corpus: dict[str, str]):
    stripped = dict(corpus)
    found: dict[str, list[str]] = {}
    stats = Counter()
    for sid, text in corpus.items():
        if not sid.split("::")[0].endswith(".py"):
            continue
        try:
            node = ast.parse(text.strip() + "\n").body[0]
        except (SyntaxError, IndexError, ValueError):
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        stats["with_docstring"] += 1
        query = " ".join(doc.strip().split("\n\n")[0].split())
        if len(query.split()) < MIN_WORDS:
            stats["too_short"] += 1
            continue
        expr = node.body[0]
        lines = text.strip().splitlines()
        body = lines[:expr.lineno - 1] + lines[expr.end_lineno:]
        if len(body) <= 1:
            stats["only_a_docstring"] += 1
            continue
        stripped[sid] = "\n".join(body)
        found.setdefault(query, []).append(sid)
    queries = []
    for query, sids in found.items():
        if len(sids) > 1:
            stats["duplicate_text"] += len(sids)
            continue
        queries.append({"text": query, "answer": sids[0]})
    norm = lambda s: " ".join(s.split()).lower()
    blob = norm("\n".join(stripped.values()))
    kept = [q for q in queries if norm(q["text"]) not in blob]
    stats["leaked"] = len(queries) - len(kept)
    stats["kept"] = len(kept)
    return kept, stripped, dict(stats)


def path_of(sid: str) -> str:
    return sid.split("::", 1)[0]


def ranks(ids: list[str], answer_sid: str | None, answer_file: str) -> tuple:
    fn = ids.index(answer_sid) + 1 if answer_sid in ids else None
    files: list[str] = []
    for sid in ids:
        p = path_of(sid)
        if p not in files:
            files.append(p)
    fl = files.index(answer_file) + 1 if answer_file in files else None
    return fn, fl


def summarise(rows: list[dict], key: str) -> dict:
    n = len(rows)
    got = [r[key] for r in rows]
    top = lambda k: sum(1 for x in got if x is not None and x <= k) / max(1, n)
    mrr = sum(1 / x for x in got if x) / max(1, n)
    return {"n": n, "top1": top(1), "top3": top(3), "top10": top(10), "mrr": mrr}


def git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def archive(root: Path, commit: str, dest: Path) -> None:
    data = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit],
                          capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def most_called(corpus: dict[str, str]) -> tuple[str | None, tuple | None]:
    defined = {sid.split("::")[1].split("#")[0] for sid in corpus}
    defined = {d for d in defined if len(d) > 3 and d not in {"window", "main", "__init__"}}
    calls = Counter()
    pairs = Counter()
    for text in corpus.values():
        seen = [m.group(1) for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", text)
                if m.group(1) in defined]
        calls.update(set(seen))
        order = list(dict.fromkeys(seen))
        pairs.update((a, b) for i, a in enumerate(order) for b in order[i + 1:])
    name = calls.most_common(1)[0][0] if calls else None
    pair = pairs.most_common(1)[0][0] if pairs else None
    return name, pair


def run_repo(name: str, root: Path, manual: list[dict], embedder) -> tuple[dict, dict]:
    report: dict = {"repo": name}
    details: dict = {"repo": name}

    started = time.perf_counter()
    scan = scan_repo(root)
    report["scan"] = {"seconds": round(time.perf_counter() - started, 1),
                      "files_indexed": scan.files_indexed, "snippets": len(scan.corpus),
                      "languages": scan.languages,
                      "skipped_dirs": dict(scan.skipped_dirs),
                      "skipped_files": dict(scan.skipped_files),
                      "skipped_snippets": dict(scan.skipped_snippets)}

    db = OUT / f"local_{name}.db"
    started = time.perf_counter()
    rep = workflows.index_folder(root, db, version="head", embedder=embedder)
    st = rep.stats
    report["index"] = {"seconds": round(time.perf_counter() - started, 1),
                       "snippets": rep.snippets, "vectors_new": st.encoded_vectors,
                       "texts_through_model": st.model_encoded,
                       "texts_from_cache": st.text_cache_hits,
                       "db_mb": round(db.stat().st_size / 1e6, 1)}

    py_q, stripped, py_stats = python_docstring_queries(scan.corpus)
    js_q, js_stats = pseudo_queries(root, scan.corpus)
    auto = py_q + [{"text": q.text, "answer": q.answer} for q in js_q]
    eval_db = OUT / f"local_{name}_eval.db"
    build(stripped, eval_db, version="eval", views=("code",), embedder=embedder,
          corpus_kind="repo")
    index = load(eval_db)
    try:
        embedder.encode([q["text"] for q in auto])
        latencies, auto_rows = [], []
        for q in auto:
            t = time.perf_counter()
            ids = workflows.shipped_search(index, q["text"], embedder, top_k=100).ids
            latencies.append((time.perf_counter() - t) * 1000)
            fn, fl = ranks(ids, q["answer"], path_of(q["answer"]))
            auto_rows.append({"fn": fn, "file": fl})
    finally:
        index.close()
    report["auto"] = {"python_docstrings": py_stats, "js_comments": js_stats,
                      "function_level": summarise(auto_rows, "fn"),
                      "file_level": summarise(auto_rows, "file"),
                      "query_ms_p50": sorted(latencies)[len(latencies) // 2] if latencies else None}
    details["auto"] = [dict(q, **r) for q, r in zip(auto, auto_rows)]

    index = load(db)
    try:
        man_rows = []
        for q in manual:
            ids = workflows.shipped_search(index, q["query"], embedder, top_k=100).ids
            _, fl = ranks(ids, None, q["file"])
            man_rows.append({"file": fl})
        report["manual"] = summarise(man_rows, "file") if manual else None
        report["manual_ranks"] = [r["file"] for r in man_rows]
        details["manual"] = [dict(q, **r) for q, r in zip(manual, man_rows)]

        ident, pair = most_called(scan.corpus)
        t = time.perf_counter()
        uses = workflows.uses(index, ident) if ident else []
        uses_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        calls = workflows.calls_before(index, *pair) if pair else []
        calls_ms = (time.perf_counter() - t) * 1000
        report["uses"] = {"lines": len(uses), "files": len({h.path for h in uses}),
                          "ms": round(uses_ms)}
        report["calls"] = {"functions": len(calls), "ms": round(calls_ms)}
        details["identifiers"] = {"uses": ident, "calls": pair}
    finally:
        index.close()

    commits = int(git(root, "rev-list", "--count", "HEAD"))
    back = min(20, commits - 1)
    new_sha, old_sha = git(root, "rev-parse", "HEAD"), git(root, "rev-parse", f"HEAD~{back}")
    scratch = SCRATCH / name
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        archive(root, old_sha, scratch / "old")
        archive(root, new_sha, scratch / "new")
        vdb = OUT / f"local_{name}_versions.db"
        if vdb.exists():
            vdb.unlink()
        t = time.perf_counter()
        first = workflows.index_folder(scratch / "old", vdb, version="old", embedder=embedder)
        first_s = time.perf_counter() - t
        t = time.perf_counter()
        second = workflows.index_folder(scratch / "new", vdb, version="new", embedder=embedder)
        second_s = time.perf_counter() - t
        report["versions"] = {
            "commits_apart": back,
            "old": {"snippets": first.snippets, "seconds": round(first_s, 1),
                    "vectors_new": first.stats.encoded_vectors},
            "new": {"snippets": second.snippets, "seconds": round(second_s, 1),
                    "vectors_new": second.stats.encoded_vectors,
                    "vectors_reused": second.stats.reused_vectors,
                    "texts_through_model": second.stats.model_encoded}}
        report["versions"]["tool"] = tool_version_check(vdb, embedder,
                                                        manual[0]["query"] if manual else
                                                        (auto[0]["text"] if auto else "main entry"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    report["versions"]["scratch_deleted"] = not scratch.exists()
    report["versions"]["worktree_vs_head"] = worktree_changes(root)
    report["tests_hidden"] = tests_hidden_effect(name, root, manual, embedder)
    return report, details


def worktree_changes(root: Path) -> dict:
    codes = Counter(line[:2].strip() or "?" for line in
                    git(root, "status", "--porcelain").splitlines() if line.strip())
    return {"deleted": codes.get("D", 0), "modified": codes.get("M", 0),
            "untracked": codes.get("??", 0),
            "other": sum(v for k, v in codes.items() if k not in {"D", "M", "??"})}


def tests_hidden_effect(name: str, root: Path, manual: list[dict], embedder) -> dict:
    scan = scan_repo(root)
    py_q, _, _ = python_docstring_queries(scan.corpus)
    js_q, _ = pseudo_queries(root, scan.corpus)
    auto = [q for q in py_q + [{"text": q.text, "answer": q.answer} for q in js_q]
            if not workflows.is_test_path(path_of(q["answer"]))]
    out = {}
    for label, db, questions in (
            ("manual", OUT / f"local_{name}.db",
             [{"text": q["query"], "answer": None, "file": q["file"]} for q in manual]),
            ("auto_non_test", OUT / f"local_{name}_eval.db",
             [{"text": q["text"], "answer": q["answer"], "file": path_of(q["answer"])}
              for q in auto])):
        if not questions or not db.exists():
            continue
        index = load(db)
        try:
            embedder.encode([q["text"] for q in questions])
            rows = {"shown": [], "hidden": []}
            for q in questions:
                ids = workflows.shipped_search(index, q["text"], embedder, top_k=200).ids
                for mode, kept in (("shown", ids),
                                   ("hidden", [i for i in ids
                                               if not workflows.is_test_path(path_of(i))])):
                    fn, fl = ranks(kept, q["answer"], q["file"])
                    rows[mode].append({"file": fl})
        finally:
            index.close()
        out[label] = {mode: summarise(r, "file") for mode, r in rows.items()}
    return out


def tool_version_check(db: Path, embedder, query: str) -> dict:
    import app
    script = [":version", ":version old", query, ":all-versions", ":collapse off", query,
              ":collapse on", query, ":quit"]
    out = io.StringIO()
    rc = app.Tool(db=db, stdin=io.StringIO("\n".join(script) + "\n"), stdout=out,
                  embedder=embedder, page_size=10).run()
    text = out.getvalue()
    distinct = re.findall(r"distinct functions (\d+) of (\d+)", text)
    return {"exit": rc,
            "listed_versions": "versions   old, new" in text,
            "switched_to_old": "searching  old" in text,
            "all_versions": "searching  old, new" in text,
            "collapse_off": list(map(int, distinct[0])) if len(distinct) > 0 else None,
            "collapse_on": list(map(int, distinct[1])) if len(distinct) > 1 else None,
            "errors": text.count("error:")}


def pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def render(reports: list[dict], idle: bool) -> str:
    lines = ["# Real local repo test: debug907 on the team's own project (KNode)", "",
             "Two real repositories of the team's KNode project, indexed read-only (nothing written "
             "inside them; git used only through `rev-list` and `archive`). **Aggregate numbers "
             "only**: no code, snippet text, file paths, identifiers or questions are in this "
             "file. `tools/local_repo_test.py`. Timings "
             + ("measured idle." if idle else "**measured under load** (other benchmarks were "
                                              "running on the same machine)."), ""]
    for r in reports:
        s, i, a = r["scan"], r["index"], r["auto"]
        lines += [f"## {r['repo']}", "",
                  "### Indexing", "",
                  "| measure | value |", "|---|---:|",
                  f"| scan time | {s['seconds']} s |",
                  f"| source files indexed | {s['files_indexed']:,} |",
                  f"| snippets (functions, classes, windows) | {s['snippets']:,} |",
                  f"| languages (files) | " + ", ".join(f"{k} {v}" for k, v in sorted(s["languages"].items())) + " |",
                  f"| index build | {i['seconds']} s ({i['texts_through_model']:,} texts through the model, "
                  f"{i['texts_from_cache']:,} from the text cache) |",
                  f"| index on disk | {i['db_mb']} MB |", "",
                  "Skipped, and why:", "",
                  "| what | reason | count |", "|---|---|---:|"]
        for reason, n in sorted(s["skipped_dirs"].items()):
            lines.append(f"| directories | {reason} | {n} |")
        for reason, n in sorted(s["skipped_files"].items()):
            lines.append(f"| files | {reason} | {n} |")
        for reason, n in sorted(s["skipped_snippets"].items()):
            lines.append(f"| snippets | {reason} | {n} |")
        py, js = a["python_docstrings"], a["js_comments"]
        fl, fn = a["file_level"], a["function_level"]
        lines += ["", "### Automatic questions (the code's own documentation)", "",
                  f"Python docstrings: {py.get('kept', 0)} questions kept of "
                  f"{py.get('with_docstring', 0)} documented functions (dropped: "
                  f"{py.get('too_short', 0)} too short, {py.get('leaked', 0)} leaked, "
                  f"{py.get('duplicate_text', 0)} duplicate, {py.get('only_a_docstring', 0)} "
                  f"nothing but a docstring). JS/TS comments: {js.get('kept', 0)} kept of "
                  f"{js.get('with_comment', 0)} (dropped {js.get('leaked', 0)} leaked). "
                  "A Python docstring is removed from its indexed answer (JS/TS comments sit "
                  "above the function and are never part of it).", "",
                  "| level | questions | top-1 | top-3 | top-10 | MRR |", "|---|---:|---:|---:|---:|---:|",
                  f"| right FILE | {fl['n']} | {pct(fl['top1'])} | {pct(fl['top3'])} | {pct(fl['top10'])} | {fl['mrr']:.3f} |",
                  f"| right function | {fn['n']} | {pct(fn['top1'])} | {pct(fn['top3'])} | {pct(fn['top10'])} | {fn['mrr']:.3f} |",
                  "", f"Query latency p50: {a['query_ms_p50']:.0f} ms." if a["query_ms_p50"] else ""]
        m = r.get("manual")
        if m:
            ranks_text = ", ".join("-" if x is None else str(x) for x in r["manual_ranks"])
            lines += ["", "### Hand-written questions", "",
                      f"{m['n']} plain-English questions written for this test; each expected file "
                      "was chosen from the list of function names BEFORE any search was run.", "",
                      f"Rank of the expected file, per question: {ranks_text}  ",
                      f"top-1 {pct(m['top1'])}, top-3 {pct(m['top3'])}, top-10 {pct(m['top10'])}, MRR {m['mrr']:.3f}."]
        u, c = r["uses"], r["calls"]
        lines += ["", "### `:uses` and `:calls` on real identifiers", "",
                  f"`:uses <the most-called function>`: {u['lines']} lines in {u['files']} files, {u['ms']} ms.  ",
                  f"`:calls <A> before <B>` (the most frequent ordered pair): {c['functions']} function{'' if c['functions'] == 1 else 's'}, {c['ms']} ms."]
        v = r.get("versions")
        if v:
            t = v["tool"]
            lines += ["", "### Two real commits as versions", "",
                      f"Commits {v['commits_apart']} apart, extracted read-only with `git archive` into "
                      f"`out/scratch/` (deleted afterwards: {'yes' if v['scratch_deleted'] else 'NO'}).", "",
                      "| build | snippets | new vectors | reused | time |", "|---|---:|---:|---:|---:|",
                      f"| older commit (first) | {v['old']['snippets']:,} | {v['old']['vectors_new']:,} | - | {v['old']['seconds']} s |",
                      f"| newer commit (incremental) | {v['new']['snippets']:,} | {v['new']['vectors_new']:,} | "
                      f"{v['new']['vectors_reused']:,} | {v['new']['seconds']} s |", "",
                      f"Through the tool: `:version` listed both versions: {'yes' if t['listed_versions'] else 'NO'}; "
                      f"`:version old`: {'yes' if t['switched_to_old'] else 'NO'}; "
                      f"`:all-versions`: {'yes' if t['all_versions'] else 'NO'}; "
                      f"distinct functions in the top 10: collapse off {t['collapse_off'][0] if t['collapse_off'] else '-'}"
                      f"/{t['collapse_off'][1] if t['collapse_off'] else '-'}, collapse on "
                      f"{t['collapse_on'][0] if t['collapse_on'] else '-'}/{t['collapse_on'][1] if t['collapse_on'] else '-'}; "
                      f"errors: {t['errors']}."]
            w = v.get("worktree_vs_head")
            if w and sum(w.values()):
                lines += ["", f"The checkout on disk differs from its own HEAD ({w['deleted']} files "
                          f"deleted, {w['modified']} modified, {w['untracked']} untracked, "
                          f"{w['other']} other), so the live index above and the committed "
                          "versions here cover slightly different files."]
        th = r.get("tests_hidden")
        if th:
            lines += ["", "### Hiding test files (the tool's default for your own code)", "",
                      "Test files crowded implementation out of the results, so repo indexes hide "
                      "them by default (`:tests on` shows them). Same questions, same index:", "",
                      "| questions | tests | n | top-1 | top-3 | top-10 | MRR |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            names = {"manual": "hand-written", "auto_non_test": "automatic, answer not a test"}
            for key, modes in th.items():
                for mode in ("shown", "hidden"):
                    m = modes[mode]
                    lines.append(f"| {names[key]} | {mode} | {m['n']} | {pct(m['top1'])} | "
                                 f"{pct(m['top3'])} | {pct(m['top10'])} | {m['mrr']:.3f} |")
            lines += ["", "For questions whose answer is not a test, hiding tests can only keep or "
                      "improve its rank; the cost is that tests themselves are not shown until "
                      "`:tests on`."]
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", required=True, help="NAME=PATH (read-only)")
    parser.add_argument("--questions", type=Path, default=None,
                        help="JSON {NAME: [{query, file}]} - kept out of git")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--idle", action="store_true")
    parser.add_argument("--augment", action="store_true",
                        help="add worktree + tests-hidden numbers to the existing run and re-render")
    args = parser.parse_args(argv)
    from retrieval.embed import Embedder, set_threads
    set_threads(args.threads)
    manual = json.loads(args.questions.read_text(encoding="utf-8")) if args.questions else {}
    embedder = Embedder()
    if args.augment:
        reports = json.loads((RESULTS / "local_repo_test.json").read_text(encoding="utf-8"))
        roots = dict(spec.partition("=")[::2] for spec in args.repo)
        for r in reports:
            root = Path(roots[r["repo"]])
            r["versions"]["worktree_vs_head"] = worktree_changes(root)
            r["tests_hidden"] = tests_hidden_effect(r["repo"], root, manual.get(r["repo"], []),
                                                    embedder)
        (RESULTS / "local_repo_test.md").write_text(render(reports, args.idle), encoding="utf-8")
        (RESULTS / "local_repo_test.json").write_text(json.dumps(reports, indent=1),
                                                      encoding="utf-8")
        print(json.dumps([r["tests_hidden"] for r in reports], indent=1))
        return 0
    reports, details = [], []
    for spec in args.repo:
        name, _, path = spec.partition("=")
        r, d = run_repo(name, Path(path), manual.get(name, []), embedder)
        reports.append(r)
        details.append(d)
        print(json.dumps(r, indent=1), flush=True)
    (OUT / "local_repo_test_details.json").write_text(json.dumps(details, indent=1),
                                                      encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "local_repo_test.md").write_text(render(reports, args.idle), encoding="utf-8")
    (RESULTS / "local_repo_test.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    print(f"wrote {RESULTS / 'local_repo_test.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
