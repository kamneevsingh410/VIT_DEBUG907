from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("bench/results")
MATCHED_BAND = (0.15, 0.35)

JS_RESERVED = set("""
break case catch class const continue debugger default delete do else export extends
finally for function if import in instanceof new return super switch this throw try typeof
var void while with yield let static enum await implements package protected interface
private public null true false undefined NaN Infinity arguments async of get set from as
require module exports console process Buffer global globalThis window document
Object Array String Number Boolean Symbol BigInt Math JSON Date RegExp Error TypeError
RangeError SyntaxError Promise Map Set WeakMap WeakSet Proxy Reflect Intl URL
setTimeout setInterval clearTimeout clearInterval setImmediate queueMicrotask
parseInt parseFloat isNaN isFinite encodeURIComponent decodeURIComponent
""".split())
_JS_TOKEN = re.compile(r"""
    (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<string>'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"|`(?:\\.|[^`\\])*`)
  | (?P<name>[A-Za-z_$][\w$]*)
  | (?P<other>\s+|.)
""", re.S | re.X)
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")


def _declared_js(code: str) -> list[str]:
    names: list[str] = []

    def simple(part: str) -> str | None:
        part = part.strip().lstrip(".").strip()
        part = part.split("=", 1)[0].strip()
        if ":" in part:
            part = part.split(":", 1)[1].strip()
        m = _IDENT.fullmatch(part)
        return m.group(0) if m else None

    for m in re.finditer(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", code):
        names.append(m.group(1))
    for m in re.finditer(r"\b(?:const|let|var)\s*[{\[]([^}\]]*)[}\]]", code):
        names += [n for n in map(simple, m.group(1).split(",")) if n]
    params = [m.group(1) for m in re.finditer(r"\bfunction\b[^(]*\(([^)]*)\)", code)]
    params += [m.group(1) for m in re.finditer(r"\(([^()]*)\)\s*=>", code)]
    params += [m.group(1) for m in re.finditer(r"^\s*(?:static\s+)?(?:async\s+)?\*?[#\w$]+\s*\(([^()]*)\)\s*\{",
                                               code, re.M)]
    for p in params:
        names += [n for n in map(simple, p.replace("{", ",").replace("}", ",")
                                 .replace("[", ",").replace("]", ",").split(",")) if n]
    names += [m.group(1) for m in re.finditer(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*=>", code)]
    names += [m.group(1) for m in re.finditer(r"\bcatch\s*\(\s*([A-Za-z_$][\w$]*)", code)]
    seen, out = set(), []
    for n in names:
        if n not in seen and n not in JS_RESERVED:
            seen.add(n)
            out.append(n)
    return out


def rename_js(code: str) -> str | None:
    tokens = [(m.lastgroup, m.group(0)) for m in _JS_TOKEN.finditer(code)]
    code_only = "".join('""' if kind == "string" else " " if kind == "comment" else t
                        for kind, t in tokens)
    declared = _declared_js(code_only)
    if not declared:
        return None
    taken = {t for kind, t in tokens if kind == "name"}
    fresh = (n for n in _short_js_names() if n not in taken)
    mapping = {name: next(fresh) for name in declared}
    out = []
    significant = [i for i, (kind, t) in enumerate(tokens)
                   if kind != "comment" and not (kind == "other" and t.isspace())]
    position = {i: k for k, i in enumerate(significant)}
    for i, (kind, text) in enumerate(tokens):
        if kind == "name" and text in mapping:
            k = position[i]
            before = tokens[significant[k - 1]][1] if k > 0 else ""
            after = tokens[significant[k + 1]][1] if k + 1 < len(significant) else ""
            if before not in (".", "?.") and not (after == ":" and before in ("{", ",")):
                text = mapping[text]
        out.append(text)
    renamed = "".join(out)
    return renamed if renamed != code else None


def _short_js_names():
    import itertools
    letters = "abcdefghijklmnopqrstuvwxyz"
    for size in (1, 2):
        for combo in itertools.product(letters, repeat=size):
            name = "".join(combo)
            if name not in JS_RESERVED:
                yield name


def rename_python(code: str) -> str | None:
    from tools.evolutionary import rename_variables
    return rename_variables(textwrap.dedent(code), style="short", keep=frozenset({"self", "cls"}))


def rename(sid: str, code: str) -> str | None:
    path = sid.split("::", 1)[0]
    if path.endswith(".py"):
        return rename_python(code)
    if path.endswith((".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")):
        return rename_js(code)
    return None


def changed_fraction(before: str, after: str) -> float:
    a, b = _IDENT.findall(before), _IDENT.findall(after)
    if not a or len(a) != len(b):
        return 0.0
    return sum(x != y for x, y in zip(a, b)) / len(a)


def questions(name: str, root: Path) -> tuple[dict[str, str], list[dict]]:
    from bench.repo_eval import pseudo_queries
    from pipeline.repo import scan_repo
    from tools.local_repo_test import python_docstring_queries

    corpus = scan_repo(root).corpus
    py_q, stripped, _ = python_docstring_queries(corpus)
    js_q, _ = pseudo_queries(root, corpus)
    qs = py_q + [{"text": q.text, "answer": q.answer} for q in js_q]
    return stripped, [q for q in qs if q["answer"] in stripped]


def ndcg(rank: int | None) -> float:
    return 1.0 / math.log2(rank + 1) if rank and rank <= 10 else 0.0


def probe_repo(name: str, root: Path, emb) -> dict:
    import numpy as np
    from bench.significance import paired_bootstrap

    started = time.perf_counter()
    corpus, qs = questions(name, root)
    ids = list(corpus)
    col = {sid: i for i, sid in enumerate(ids)}
    variants = {}
    touched = []
    for q in qs:
        a = q["answer"]
        if a not in variants:
            new = rename(a, corpus[a])
            if new is not None:
                variants[a] = new
                touched.append(changed_fraction(corpus[a], new))
    kept = [q for q in qs if q["answer"] in variants]
    from pipeline.query_proc import process
    from retrieval.index import build, load
    db = Path("out") / f"memorisation_{name}.db"
    if db.exists():
        db.unlink()
    build(corpus, db, version="original", views=("code",), embedder=emb,
          corpus_kind="repo", show_progress=True)
    build(variants, db, version="renamed", views=("code",), embedder=emb,
          corpus_kind="repo", show_progress=True)
    index = load(db, versions="all")
    try:
        mat, uids = index.matrices["code"], index.matrix_ids["code"]
        row = {(index.identity_of[u], index.version_of[u]): i for i, u in enumerate(uids)}
        D = np.asarray(mat[[row[(s, "original")] for s in ids]], dtype="float32")
        V = {a: np.asarray(mat[row[(a, "renamed")]], dtype="float32") for a in variants}
    finally:
        index.close()
    Q = np.asarray(emb.encode([process(q["text"]).embedding_text or q["text"] for q in kept]),
                   dtype="float32")
    orig, var = {}, {}
    for i, q in enumerate(kept):
        scores = D @ Q[i]
        a = col[q["answer"]]
        own = scores[a]
        other = np.delete(scores, a)
        rank_o = 1 + int((other > own).sum())
        rank_v = 1 + int((other > float(V[q["answer"]] @ Q[i])).sum())
        orig[str(i)], var[str(i)] = ndcg(rank_o), ndcg(rank_v)
    cmp = paired_bootstrap(var, orig, "renamed", "original", iterations=2000)
    o, v = float(np.mean(list(orig.values()))), float(np.mean(list(var.values())))
    return {"repo": name, "functions": len(ids), "questions": len(qs), "probed": len(kept),
            "not_renamable": len(qs) - len(kept),
            "identifier_tokens_changed": round(float(np.mean(touched)), 3) if touched else 0.0,
            "ndcg_original": round(o, 4), "ndcg_renamed": round(v, 4),
            "diff": round(cmp.diff.mean, 4), "low": round(cmp.diff.low, 4),
            "high": round(cmp.diff.high, 4), "p": round(cmp.p_value, 4),
            "relative_drop": round((o - v) / o, 3) if o else None,
            "seconds": round(time.perf_counter() - started, 1)}


def apps_reference() -> dict:
    s = json.loads((RESULTS / "evolutionary.json").read_text(encoding="utf-8"))["summary"]
    p = s["probe"]["v_rename_short"]
    o, v = p["original_ndcg"], p["variant_ndcg"]
    from bench.dataset import load_appsretrieval
    from tools.evolutionary import rename_variables
    corpus, _, _ = load_appsretrieval()
    rows = json.loads((RESULTS / "evolutionary.json").read_text(encoding="utf-8"))["rows"]
    touched: dict[str, float] = {}
    for gold in sorted({r["gold"] for r in rows if "probe_v_rename_short" in r}):
        new = rename_variables(corpus[gold], style="short")
        if new:
            touched[gold] = changed_fraction(corpus[gold], new)
    band = [r for r in rows if "probe_v_rename_short" in r
            and MATCHED_BAND[0] <= touched.get(r["gold"], -1) <= MATCHED_BAND[1]]
    mo = sum(ndcg(r["probe_test"]) for r in band) / max(1, len(band))
    mv = sum(ndcg(r["probe_v_rename_short"]) for r in band) / max(1, len(band))
    return {"repo": "AppsRetrieval (public)", "probed": p["queries"], "ndcg_original": round(o, 4),
            "ndcg_renamed": round(v, 4), "diff": round(v - o, 4),
            "relative_drop": round((o - v) / o, 3),
            "identifier_tokens_changed": round(sum(touched.values()) / max(1, len(touched)), 3),
            "matched": {"band": MATCHED_BAND, "probed": len(band), "ndcg_original": round(mo, 4),
                        "ndcg_renamed": round(mv, 4), "diff": round(mv - mo, 4),
                        "relative_drop": round((mo - mv) / mo, 3) if mo else None,
                        "identifier_tokens_changed": round(
                            sum(touched[r["gold"]] for r in band) / max(1, len(band)), 3)}}


def render(rows: list[dict], ref: dict, private: set[str]) -> str:
    lines = ["# Memorisation probe on code the encoder cannot have memorised (2.2)", "",
             "The AppsRetrieval rename probe (`evolutionary.md`) cannot tell memorisation of "
             "the public corpus from genuine use of names. Here the same rename, local "
             "variables to short names not already in the snippet, is applied to the answers "
             "of held-out questions on real repositories. The team's KNode repositories are "
             "private and were never published, so the encoder cannot have memorised them. "
             "`tools/memorisation_probe.py`. **Aggregate numbers only**: no code, identifiers, "
             "paths or questions.", "",
             "| corpus | questions probed | identifier tokens renamed | NDCG@10 original "
             "| renamed | change [95% CI] | relative drop |",
             "|---|---:|---:|---:|---:|---|---:|",
             f"| {ref['repo']} | {ref['probed']:,} | {ref['identifier_tokens_changed']:.0%} | "
             f"{ref['ndcg_original']:.4f} | "
             f"{ref['ndcg_renamed']:.4f} | {ref['diff']:+.4f} | {ref['relative_drop']:.0%} |"]
    m = ref["matched"]
    lines.append(f"| AppsRetrieval, dose-matched ({m['band'][0]:.0%}–{m['band'][1]:.0%} of tokens "
                 f"renamed) | {m['probed']:,} | {m['identifier_tokens_changed']:.0%} | "
                 f"{m['ndcg_original']:.4f} | {m['ndcg_renamed']:.4f} | {m['diff']:+.4f} | "
                 + (f"{m['relative_drop']:.0%}" if m["relative_drop"] is not None else "-") + " |")
    for r in rows:
        label = f"{r['repo']} ({'private' if r['repo'] in private else 'public'})"
        lines.append(f"| {label} | {r['probed']:,} | {r['identifier_tokens_changed']:.0%} | "
                     f"{r['ndcg_original']:.4f} | {r['ndcg_renamed']:.4f} | {r['diff']:+.4f} "
                     f"[{r['low']:+.4f}, {r['high']:+.4f}] | "
                     + (f"{r['relative_drop']:.0%}" if r["relative_drop"] is not None else "-") + " |")
    lines += ["", "Questions: Python docstrings (removed from the indexed answer) and JS/TS "
              "comments above a function (never part of it); questions whose text appears in "
              "any indexed snippet were dropped. A question is probed when its answer has at "
              "least one local name to rename. Only the answer changes; every other function "
              "keeps its original vector.", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from retrieval.embed import Embedder, set_threads
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", required=True, help="NAME=PATH (read-only)")
    parser.add_argument("--private", action="append", default=[],
                        help="names of repositories that were never published")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    set_threads(args.threads)
    emb = Embedder()
    rows = []
    for spec in args.repo:
        name, _, path = spec.partition("=")
        row = probe_repo(name, Path(path), emb)
        rows.append(row)
        print(json.dumps(row), flush=True)
    ref = apps_reference()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "memorisation.json").write_text(json.dumps({"apps": ref, "repos": rows}, indent=1),
                                               encoding="utf-8")
    (RESULTS / "memorisation.md").write_text(render(rows, ref, set(args.private)), encoding="utf-8")
    print(f"wrote {RESULTS / 'memorisation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
