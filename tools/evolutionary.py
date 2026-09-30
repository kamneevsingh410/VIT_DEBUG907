from __future__ import annotations

import ast
import builtins
import io
import json
import math
import shutil
import sys
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.dataset import load_appsretrieval
from bench.stratified import load as load_stratified
from retrieval import workflows
from retrieval.embed import Embedder, set_threads
from retrieval.index import build, load

SOURCE_DB = Path("out/real.db")
DB = Path("out/evolutionary.db")
RESULTS = Path("bench/results")
BUILTINS = set(dir(builtins))


def _short_names(taken: set[str]):
    import itertools
    import keyword
    letters = "abcdefghijklmnopqrstuvwxyz"
    for size in (1, 2):
        for combo in itertools.product(letters, repeat=size):
            name = "".join(combo)
            if name not in taken and not keyword.iskeyword(name) and name not in BUILTINS:
                yield name


def rename_variables(src: str, style: str = "var", keep: frozenset = frozenset()) -> str | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    stored, protected = [], set(BUILTINS) | set(keep)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stored.append(node.id)
        elif isinstance(node, ast.arg):
            stored.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            protected.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            protected.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                protected.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            protected.update(node.names)
    mapping: dict[str, str] = {}
    all_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | set(protected) \
        | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    fresh = _short_names(all_names)
    for name in stored:
        if name not in protected and name not in mapping and not name.startswith("__"):
            mapping[name] = f"var{len(mapping) + 1}" if style == "var" else next(fresh)
    if not mapping:
        return None
    out, prev = [], None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if (tok.type == tokenize.NAME and tok.string in mapping
                    and not (prev is not None and prev.type == tokenize.OP and prev.string == ".")):
                tok = tok._replace(string=mapping[tok.string])
            out.append(tok)
            if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
                prev = tok
        return tokenize.untokenize(out)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _safe_pair(a: ast.stmt, b: ast.stmt) -> bool:
    if a.lineno == a.end_lineno == b.lineno - 1 == b.end_lineno - 1:
        if isinstance(a, (ast.Import, ast.ImportFrom)) and isinstance(b, (ast.Import, ast.ImportFrom)):
            return True
        if isinstance(a, ast.Assign) and isinstance(b, ast.Assign):
            no_calls = not any(isinstance(n, ast.Call) for n in ast.walk(a.value)) and \
                not any(isinstance(n, ast.Call) for n in ast.walk(b.value))
            return no_calls and not (_names(a) & _names(b))
    return False


def reorder_statements(src: str) -> str | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    bodies = [tree.body] + [n.body for n in ast.walk(tree)
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    lines = src.splitlines(keepends=True)
    for body in bodies:
        for a, b in zip(body, body[1:]):
            if _safe_pair(a, b):
                i, j = a.lineno - 1, b.lineno - 1
                indent_a = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                indent_b = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                if indent_a != indent_b:
                    continue
                lines[i], lines[j] = lines[j], lines[i]
                return "".join(lines)
    return None


def first_rank(ids: list[str], relevant: str, index) -> int | None:
    for rank, uid in enumerate(ids, 1):
        if uid == relevant:
            return rank
    return None


def ndcg(rank: int | None) -> float:
    return 1.0 / math.log2(rank + 1) if rank and rank <= 10 else 0.0


def main(argv: list[str] | None = None) -> int:
    set_threads(8)
    corpus, queries, qrels = load_appsretrieval()
    picked = set(load_stratified())
    sample = {q: t for q, t in queries.items() if q in picked and q in qrels}
    golds = sorted({next(iter(qrels[q])) for q in sample})

    variants = {"v_rename": {}, "v_rename_short": {}, "v_reorder": {}}
    skipped = {k: 0 for k in variants}
    for did in golds:
        for name, fn in (("v_rename", rename_variables),
                         ("v_rename_short", lambda src: rename_variables(src, style="short")),
                         ("v_reorder", reorder_statements)):
            new = fn(corpus[did])
            if new is None or new == corpus[did]:
                skipped[name] += 1
            else:
                variants[name][did] = new

    emb = Embedder()
    if not DB.exists():
        shutil.copyfile(SOURCE_DB, DB)
    build_stats = {}
    for name, docs in variants.items():
        st = build(docs, DB, version=name, views=("code",), embedder=emb, corpus_kind="apps",
                   show_progress=True)
        build_stats[name] = st.as_dict()

    index = load(DB, versions="all")
    per_query = []
    try:
        emb.encode(list(sample.values()))
        for qid, text in sample.items():
            gold = next(iter(qrels[qid]))
            off = workflows.shipped_search(index, text, emb, top_k=300, collapse=False).ids
            on = workflows.shipped_search(index, text, emb, top_k=10, collapse=True).ids
            ident = index.identity_of
            first = lambda ids: next((r for r, u in enumerate(ids, 1) if ident[u] == gold), None)
            row = {"qid": qid, "gold": gold,
                   "off_rank": first(off[:10]), "on_rank": first(on),
                   "off_distinct": len({ident[u] for u in off[:10]}),
                   "on_distinct": len({ident[u] for u in on[:10]})}
            for version in ("test", "v_rename", "v_rename_short", "v_reorder"):
                if version != "test" and gold not in variants[version]:
                    continue
                keep = [u for u in off if index.version_of[u] == "test" and ident[u] != gold
                        or ident[u] == gold and index.version_of[u] == version]
                target = next((u for u in keep if ident[u] == gold), None)
                row[f"probe_{version}"] = first_rank(keep[:10], target, index) if target else None
            per_query.append(row)
    finally:
        index.close()

    n = len(per_query)
    mean = lambda xs: sum(xs) / max(1, len(xs))
    summary = {
        "queries": n, "variants": {k: len(v) for k, v in variants.items()}, "skipped": skipped,
        "build": build_stats,
        "collapse_off": {"ndcg_at_10": mean([ndcg(r["off_rank"]) for r in per_query]),
                         "distinct_in_top10": mean([r["off_distinct"] for r in per_query])},
        "collapse_on": {"ndcg_at_10": mean([ndcg(r["on_rank"]) for r in per_query]),
                        "distinct_in_top10": mean([r["on_distinct"] for r in per_query])},
        "probe": {},
    }
    for version in ("v_rename", "v_rename_short", "v_reorder"):
        rows = [r for r in per_query if f"probe_{version}" in r]
        summary["probe"][version] = {
            "queries": len(rows),
            "original_ndcg": mean([ndcg(r["probe_test"]) for r in rows]),
            "variant_ndcg": mean([ndcg(r[f"probe_{version}"]) for r in rows]),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "evolutionary.json").write_text(json.dumps({"summary": summary, "rows": per_query},
                                                          indent=1), encoding="utf-8")
    write_md(summary, variants)
    print(json.dumps(summary, indent=1))
    return 0


def write_md(s: dict, variants: dict) -> None:
    off, on = s["collapse_off"], s["collapse_on"]
    lines = ["# Bonus benchmark: evolving code and a memorisation probe", "",
             f"Validated 1,000-query sample. For each relevant solution, synthetic later "
             f"versions were added to a copy of the index under the same ids: "
             f"**{s['variants']['v_rename']}** with local variables renamed to var1, var2, ... and "
             f"**{s['variants']['v_rename_short']}** renamed to short names not already in the file "
             f"(style-preserving control; skipped "
             f"{s['skipped']['v_rename']}: unparseable or nothing to rename) and "
             f"**{s['variants']['v_reorder']}** with one provably safe pair of adjacent "
             f"statements swapped (skipped {s['skipped']['v_reorder']}: no safe pair). "
             "`tools/evolutionary.py`.", "",
             "## Near-duplicate collapse, all versions searched together", "",
             "| | NDCG@10 | distinct solutions in the top 10 |", "|---|---:|---:|",
             f"| collapse OFF | {off['ndcg_at_10']:.4f} | {off['distinct_in_top10']:.2f} |",
             f"| collapse ON | {on['ndcg_at_10']:.4f} | {on['distinct_in_top10']:.2f} |", "",
             "NDCG@10 here counts the first appearance of the relevant solution in any version.",
             "", "## Memorisation probe", "",
             "The corpus as it would be if the relevant solution had been rewritten: only the "
             "variant is relevant, every other document is unchanged. A small drop means the "
             "encoder follows the logic rather than memorised text (CoIR AppsRetrieval is public, "
             "so memorisation is a fair worry).", "",
             "| relevant solution | queries | NDCG@10 original | NDCG@10 variant | change |",
             "|---|---:|---:|---:|---:|"]
    for version, p in s["probe"].items():
        lines.append(f"| {version} | {p['queries']} | {p['original_ndcg']:.4f} | "
                     f"{p['variant_ndcg']:.4f} | {p['variant_ndcg'] - p['original_ndcg']:+.4f} |")
    delta = {v: p["variant_ndcg"] - p["original_ndcg"] for v, p in s["probe"].items()}
    lines += ["", "**Reading.** "
              + (f"Style-preserving renaming costs only {delta['v_rename_short']:+.4f} and a safe "
                 f"reorder {delta['v_reorder']:+.4f}, so exact-text memorisation of the public "
                 f"corpus cannot explain the {delta['v_rename']:+.4f} of the var1/var2 rename: that "
                 "drop comes from the naming scheme itself (every renamed solution shares the same "
                 "repeated 'var' tokens, and descriptive names like `ans` or `graph` disappear)."
                 if delta.get("v_rename_short", -1) > -0.05 else
                 f"Even style-preserving renaming costs {delta['v_rename_short']:+.4f}: the encoder "
                 "relies on the exact identifiers. That is consistent with memorisation of the "
                 "public corpus AND with genuine use of variable names; this probe cannot separate "
                 "the two. One genuine signal is lost by any rename: AppsRetrieval statements use the "
                 "same letters as their solutions ('given $n$ integers $a_1..a_n$' <-> "
                 "`n = int(input())`, `a = ...`), and renaming breaks that link. A safe reorder, which "
                 "also changes the exact text but keeps every identifier, costs only "
                 f"{delta['v_reorder']:+.4f}.")]
    example = next(iter(variants["v_rename"].items()), None)
    if example:
        lines += ["", f"Example rename (`{example[0]}`, first lines):", "", "```python",
                  "\n".join(example[1].strip().splitlines()[:8]), "```"]
    (RESULTS / "evolutionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
