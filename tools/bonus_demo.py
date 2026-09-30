from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app

DB = Path("out/npm_versions.db")
VERSIONS = "v10.8.0,v10.8.3,v10.9.0"
QUERIES = [
    "check that the user is logged in to the registry before publishing a package",
    "read the npmrc config file and merge it with the environment",
    "print the audit report as a table of vulnerabilities",
]
RESULTS = Path("bench/results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--versions", default=VERSIONS)
    parser.add_argument("--query", action="append", help="replace the default queries")
    args = parser.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found: run tools/p1_versions.py first")
        return 1
    queries = args.query or QUERIES

    script = [f":version {args.versions}"]
    for q in queries:
        script += [":collapse off", q, ":collapse on", q]
    script.append(":quit")
    stdout = io.StringIO()
    tool = app.Tool(db=args.db, stdin=io.StringIO("\n".join(script) + "\n"), stdout=stdout,
                    page_size=10)
    rc = tool.run()
    transcript = stdout.getvalue()

    counts = re.findall(r"distinct functions (\d+) of (\d+)", transcript)
    rows = []
    for i, q in enumerate(queries):
        off, on = counts[2 * i], counts[2 * i + 1]
        rows.append((q, off, on))
    lines = ["# Bonus demo: evolutionary retrieval across npm releases", "",
             f"One index, releases {args.versions.replace(',', ', ')} loaded together; the shipped "
             "search, top 10, near-duplicate collapse OFF vs ON. Recorded through the terminal "
             "tool (`tools/bonus_demo.py` scripts the keystrokes; the transcript below is its "
             "unedited output).", "",
             "| query | collapse OFF: distinct functions in top 10 | collapse ON |",
             "|---|---:|---:|"]
    for q, off, on in rows:
        lines.append(f"| {q} | {off[0]} of {off[1]} | {on[0]} of {on[1]} |")
    lines += ["", "With collapse OFF the same function appears once per release, crowding "
              "out different functions; with it ON each function appears once, labelled with "
              "every release that contains it.", "", "## Transcript", "", "```",
              transcript.strip(), "```"]
    lines += evolution_section(args.db)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bonus_demo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for q, off, on in rows:
        print(f"{off[0]}/{off[1]} -> {on[0]}/{on[1]}  {q}")
    print(f"wrote {RESULTS / 'bonus_demo.md'}")
    return rc


EVOLUTION_QUERIES = [
    "run a command from a local or remote npm package",
    "list the webhooks registered for a package on the registry",
    "print the funding urls of installed dependencies",
    "decide whether an installed package is outdated compared to the registry",
]


def evolution_section(db: Path, candidates: list[str] = EVOLUTION_QUERIES) -> list[str]:
    from retrieval import workflows
    from retrieval.versions import evolution
    stdout = io.StringIO()
    tool = app.Tool(db=db, stdin=io.StringIO(""), stdout=stdout, page_size=5)
    tool.load_model()
    tool.open_index(db, versions="all")
    query = candidates[-1]
    for q in candidates:
        hits = workflows.shipped_search(tool.index, q, tool.embedder, top_k=5,
                                        include_tests=False).hits
        if any(any(a.changed for a in evolution(tool.index, sid)) for sid, _ in hits):
            query = q
            break
    tool.open_index(db)
    typed = [":all-versions", query]
    for line in typed:
        tool.say(f"debug907> {line}")
        tool.handle(line)
    changed = next((n for n, (sid, _) in enumerate(tool.hits[:tool.page_size], 1)
                    if any(a.changed for a in evolution(tool.index, sid))), None)
    more = ([f":history {changed}"] if changed else []) + [":newest on", query]
    for line in more:
        tool.say(f"debug907> {line}")
        tool.handle(line)
    typed += more
    releases = len(tool.index.available_versions)
    tool.index.close()
    return ["", "## Evolution view (2.3)", "",
            f"All {releases} releases loaded. Each collapsed row lists "
            "the releases that contain the function, `=` when its code is identical to the "
            "previous release and `!=` when it changed (content hash). `:history N` prints the "
            "releases and a unified diff per change; `:newest on` (off by default) shows the "
            "newest of the identical versions, so paths and line numbers are today's. "
            "Versions are linked by `path::name` when that is unique in every release, so a "
            "function that only moved down its file keeps its history. The query is the first "
            f"of {len(candidates)} fixed candidates (`EVOLUTION_QUERIES`) whose top 5 contains "
            "a function that changed; most code is identical across patch releases.", "",
            "Typed: " + " → ".join(f"`{t}`" for t in typed), "", "```",
            stdout.getvalue().strip(), "```"]


if __name__ == "__main__":
    raise SystemExit(main())
