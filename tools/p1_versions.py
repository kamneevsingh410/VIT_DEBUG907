from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.repo import extract_repo
from retrieval import workflows
from retrieval.embed import Embedder, set_threads
from retrieval.index import load

NPM = Path("external/npm")
RELEASES = ["v10.8.0", "v10.8.1", "v10.8.2", "v10.8.3", "v10.9.0"]
RESULTS = Path("bench/results")
QUERY = "check whether the registry auth token is valid before publishing"


def checkout(tag: str) -> None:
    subprocess.run(["git", "-C", str(NPM), "checkout", "-q", "--force", tag], check=True)


def diff(prev: dict[str, str], cur: dict[str, str]) -> dict[str, int]:
    prev_texts = set(prev.values())
    return {
        "added_ids": len(cur.keys() - prev.keys()),
        "removed_ids": len(prev.keys() - cur.keys()),
        "same_id_changed_text": sum(1 for k in cur.keys() & prev.keys() if cur[k] != prev[k]),
        "texts_not_in_previous": sum(1 for t in cur.values() if t not in prev_texts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1: full vs incremental on npm releases")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--db", type=Path, default=Path("out/npm_versions.db"))
    parser.add_argument("--releases", nargs="+", default=RELEASES)
    parser.add_argument("--idle", action="store_true", help="the machine is otherwise idle")
    parser.add_argument("--render", action="store_true",
                        help="only re-render the report from bench/results/p1_versions.json")
    args = parser.parse_args(argv)
    if args.render:
        write_md(json.loads((RESULTS / "p1_versions.json").read_text(encoding="utf-8")))
        return 0
    if not NPM.is_dir():
        print(f"{NPM} not found: git clone https://github.com/npm/cli.git {NPM}")
        return 1
    if args.db.exists():
        print(f"{args.db} exists; P1 needs a cold first build. Move or delete it first.")
        return 1

    print(f"threads {set_threads(args.threads)}   releases {' '.join(args.releases)}")
    emb = Embedder(cache="write")
    emb.warm()
    rows, prev = [], {}
    for tag in args.releases:
        checkout(tag)
        corpus = extract_repo(NPM)
        changes = diff(prev, corpus) if prev else None
        started = time.perf_counter()
        report = workflows.index_folder(NPM, args.db, version=tag, embedder=emb)
        seconds = time.perf_counter() - started
        st = report.stats
        row = {"release": tag, "functions": report.snippets, "chars": sum(map(len, corpus.values())),
               "new_vectors": st.encoded_vectors, "reused_vectors": st.reused_vectors,
               "texts_through_model": st.model_encoded, "encode_s": round(st.encode_ms / 1000, 1),
               "total_s": round(seconds, 1), "db_mb": round(args.db.stat().st_size / 1e6, 1),
               "changes": changes}
        rows.append(row)
        prev = corpus
        print(json.dumps(row), flush=True)

    cold = rows[0]
    per_text = cold["encode_s"] / max(1, cold["texts_through_model"])
    for row in rows:
        row["full_rebuild_est_s"] = round(row["functions"] * per_text, 1)

    index = load(args.db, versions="all")
    try:
        per_version = {v: sum(1 for u in index.ids if index.version_of[u] == v)
                       for v in index.loaded_versions}
        result = workflows.shipped_search(index, QUERY, emb, top_k=5)
        top = [{"id": index.identity_of[u], "shown": index.version_of[u],
                "in": result.versions.get(u, [])} for u in result.ids]
    finally:
        index.close()

    out = {"releases": rows, "per_version_rows": per_version, "query": QUERY, "top5": top,
           "threads": args.threads, "text_cache": "write-only (never read)",
           "idle": args.idle}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p1_versions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(out)
    print(f"wrote {RESULTS / 'p1_versions.md'}")
    return 0


def write_md(out: dict) -> None:
    rows = out["releases"]
    lines = [
        "# P1: incremental indexing on real code (npm releases)",
        "",
        f"Corpus: first-party npm/cli source, releases {rows[0]['release']} to "
        f"{rows[-1]['release']}, function level (index-repo extractor), shipped config "
        f"(gte-modernbert-base, fp32, seq 1024, code view). One index, versions coexist. "
        f"The encoder never READS the text cache (write-only mode), so the first build "
        f"pays the true model cost and later builds reuse only the index's own text-keyed "
        f"vectors. {out['threads']} threads; timings "
        + ("measured idle." if out.get("idle") else "**measured under load** (other "
           "benchmarks ran on the machine at the same time).") + "",
        "",
        "| release | functions | texts through the model | new vectors | reused | encode s "
        "| total s | full rebuild would cost (s) | index MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['release']} | {r['functions']:,} | {r['texts_through_model']:,} | "
                     f"{r['new_vectors']:,} | {r['reused_vectors']:,} | {r['encode_s']} | "
                     f"{r['total_s']} | {r['full_rebuild_est_s']} | {r['db_mb']} |")
    lines += ["", "## What changed between releases", "",
              "| release | ids added | ids removed | same id, new text | texts not in previous |",
              "|---|---:|---:|---:|---:|"]
    for r in rows[1:]:
        c = r["changes"]
        lines.append(f"| {r['release']} | {c['added_ids']} | {c['removed_ids']} | "
                     f"{c['same_id_changed_text']} | {c['texts_not_in_previous']} |")
    lines += ["", "Ids are `path::name#line`, so a function that only MOVED gets a new id; its "
              "text, and therefore its vector, is unchanged and reused.", "",
              "## All versions searchable together", "",
              f"Rows per version: {out['per_version_rows']}", "",
              f"Query: \"{out['query']}\" (all versions, collapsed):", ""]
    for i, t in enumerate(out["top5"], 1):
        lines.append(f"{i}. `{t['id']}`: shown from {t['shown']}, present in {', '.join(t['in'])}")
    lines += sync_section()
    (RESULTS / "p1_versions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_section() -> list[str]:
    path = RESULTS / "sync.json"
    if not path.exists():
        return []
    s = json.loads(path.read_text(encoding="utf-8"))
    lines = ["", "## Commit sync (`debug907 sync`, 2.5)", "",
             "Each commit indexed as a version labelled with its short hash, read-only on the "
             "repository (`git rev-parse`, `ls-tree`, `cat-file`; nothing checked out, no hooks). "
             f"Text cache write-only, {s['threads']} threads, timings "
             + ("measured idle." if s.get("idle") else "**measured under load**.") + " `export` "
             "writes the commit's indexable files to a scratch folder; `index` is extraction plus "
             "the incremental build.", ""]
    for name, title in (("releases", "release tags"), ("commits", "consecutive commits up to v10.9.0")):
        rows, summ = s[name], s["summary"][name]
        lines += [f"### {len(rows)} {title}", "",
                  "| commit | functions | new or changed | reused | export s | index s | total s |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {r['label']} ({r['rev'][:12]}) | {r['functions']:,} | {r['new_or_changed']:,} | "
                  f"{r['reused']:,} | {r['export_s']} | {r['index_s']} | {r['total_s']} |" for r in rows]
        if summ["median_later_total_s"] is not None:
            lines += ["", f"First sync (cold): {summ['first_total_s']} s. Every later one: median "
                      f"{summ['median_later_total_s']} s, re-encoding a median of "
                      f"{summ['median_later_new']} functions.", ""]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
