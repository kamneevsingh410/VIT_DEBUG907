from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import sync
from retrieval.embed import Embedder, set_threads

NPM = Path("external/npm")
RELEASES = ["v10.8.0", "v10.8.1", "v10.8.2", "v10.8.3", "v10.9.0"]
RESULTS = Path("bench/results")


def run(revs: list[str], db: Path, emb) -> list[dict]:
    if db.exists():
        db.unlink()
    rows = []
    for rev in revs:
        r = sync.sync(NPM, db, rev=rev, embedder=emb)
        row = {"rev": rev, "label": r.label, "functions": r.snippets,
               "new_or_changed": r.encoded_vectors, "reused": r.reused_vectors,
               "through_model": r.model_encoded, "export_s": round(r.export_s, 2),
               "index_s": round(r.index_s, 2), "total_s": round(r.total_s, 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def summary(rows: list[dict]) -> dict:
    later = rows[1:]
    return {"first_total_s": rows[0]["total_s"],
            "median_later_total_s": round(statistics.median(r["total_s"] for r in later), 2)
            if later else None,
            "median_later_export_s": round(statistics.median(r["export_s"] for r in later), 2)
            if later else None,
            "median_later_new": statistics.median(r["new_or_changed"] for r in later)
            if later else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--commits", type=int, default=20)
    parser.add_argument("--idle", action="store_true", help="the machine is otherwise idle")
    args = parser.parse_args(argv)
    if not NPM.is_dir():
        print(f"{NPM} not found")
        return 1
    print(f"threads {set_threads(args.threads)}")
    emb = Embedder(cache="write")
    emb.warm()
    commits = sync.git(NPM, "rev-list", "--first-parent", "--reverse",
                       f"-n{args.commits}", "v10.9.0").split()
    out = {"threads": args.threads, "idle": args.idle, "text_cache": "write-only (never read)",
           "releases": run(RELEASES, Path("out/npm_sync_releases.db"), emb),
           "commits": run(commits, Path("out/npm_sync_commits.db"), emb)}
    out["summary"] = {"releases": summary(out["releases"]), "commits": summary(out["commits"])}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sync.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
