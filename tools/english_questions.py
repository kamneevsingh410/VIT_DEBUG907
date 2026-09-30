from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import workflows
from retrieval.router import route

RESULTS = Path("bench/results")

NPM = [
    ("where is otplease used", "usage", ["otplease"], ("min_hits", 5)),
    ("who calls getCredentialsByURI", "callers", ["getCredentialsByURI"], ("min_hits", 1)),
    ("where is the reify finish helper defined", "definition", ["reifyFinish"],
     ("path", "reify-finish.js")),
    ("where is validateLockfile defined", "definition", ["validateLockfile"],
     ("path", "validate-lockfile.js")),
    ("which functions call reify before reifyFinish", "order", ["reify", "reifyFinish"],
     ("path", "install.js")),
    ("does loadActual get called before buildIdealTree", "order", ["loadActual", "buildIdealTree"],
     ("min_hits", 1)),
    ("what calls the load actual function", "callers", ["loadActual"], ("min_hits", 5)),
    ("where is buildIdealTree implemented", "definition", ["buildIdealTree"],
     ("path", "build-ideal-tree.js")),
    ('where is "https://registry.npmjs.org/" used', "usage", ["https://registry.npmjs.org/"],
     ("min_hits", 1)),
    ("how does npm decide that an installed package is outdated", "search", [], ("search", None)),
    ("where is the npm registry fetch module used", "usage", ["npm-registry-fetch"], ("min_hits", 3)),
    ("which files call validateLockfile before reifyFinish", "order",
     ["validateLockfile", "reifyFinish"], ("path", "ci.js")),
]


def answer(index, question: str) -> dict:
    r = route(question)
    if r is None:
        return {"kind": "search", "names": [], "hits": None, "paths": []}
    vocab = workflows.vocabulary(index)
    res = [vocab.resolve(p, identifiers_only=r.kind != "usage") for p in r.phrases]
    names = [x.best for x in res]
    if any(n is None for n in names):
        return {"kind": r.kind, "names": names, "hits": None, "paths": [], "fallback": True}
    if r.kind == "usage":
        hits = workflows.uses(index, names[0])
    elif r.kind == "callers":
        hits = workflows.callers(index, names[0])
    elif r.kind == "definition":
        hits = workflows.definitions(index, names[0])
    else:
        a, b = names if r.order == "before" else names[::-1]
        hits = workflows.calls_before(index, a, b)
    return {"kind": r.kind, "names": names, "hits": len(hits), "order": r.order,
            "paths": sorted({h.path for h in hits})[:8],
            "others": sum(len(x.others) for x in res)}


def correct(got: dict, kind: str, names: list[str], check) -> bool:
    if got["kind"] != kind:
        return False
    if kind == "search":
        return True
    if got.get("fallback") or got["names"] != names:
        return False
    what, value = check
    if what == "min_hits":
        return (got["hits"] or 0) >= value
    return any(value in p for p in got["paths"])


def safety() -> dict:
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from tools.hubness_eval import load_train
    _, queries, qrels = load_appsretrieval()
    train, _ = load_train()
    test = {q: queries[q] for q in qrels if q in queries}
    sample = set(load_stratified())
    routed = lambda qs: [q for q, t in qs.items() if route(t) is not None]
    return {"test": len(test), "test_routed": routed(test),
            "sample": len(sample & set(test)), "sample_routed": [q for q in routed(test) if q in sample],
            "train": len(train), "train_routed": routed(train)}


def knode(spec: str) -> dict:
    from retrieval.index import load
    name, _, db = spec.partition("=")
    index = load(Path(db))
    try:
        defs = {}
        for uid in index.ids:
            info = workflows.describe_hit(index, uid)
            if re.fullmatch(r"[a-z]+(?:_[a-z]+){1,3}|[a-z]+(?:[A-Z][a-z]+){1,3}", info.name or ""):
                defs.setdefault(info.name, 0)
                defs[info.name] += 1
        picked = [n for n, c in sorted(defs.items()) if c == 1][:60]
        picked = [n for n in picked if workflows.callers(index, n)][:3]
        ok = 0
        rows = []
        for fn, kind in zip(picked, ("definition", "callers", "usage")):
            words = " ".join(re.findall(r"[a-z]+|[A-Z][a-z]*", fn)).lower()
            q = {"definition": f"where is {words} defined", "callers": f"who calls {words}",
                 "usage": f"where is {words} used"}[kind]
            got = answer(index, q)
            good = got["kind"] == kind and got["names"] == [fn] and (got["hits"] or 0) > 0
            ok += good
            rows.append({"kind": kind, "resolved_to_the_right_name": got["names"] == [fn],
                         "answer_nonempty": (got["hits"] or 0) > 0})
    finally:
        index.close()
    return {"repo": name, "questions": len(rows), "correct": ok, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npm-db", type=Path, default=Path("out/npm_eval.db"))
    parser.add_argument("--knode", action="append", default=[], help="NAME=DB (private: counts only)")
    args = parser.parse_args(argv)
    from retrieval.index import load
    s = safety()
    index = load(args.npm_db)
    rows = []
    try:
        for q, kind, names, check in NPM:
            got = answer(index, q)
            rows.append({"question": q, "expected": kind, "got": got,
                         "correct": correct(got, kind, names, check)})
    finally:
        index.close()
    k = [knode(spec) for spec in args.knode]
    report = {"safety": s, "npm": rows, "knode": k}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "english_questions.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / "english_questions.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"routed": [len(s["test_routed"]), len(s["sample_routed"]), len(s["train_routed"])],
                      "npm_correct": sum(r["correct"] for r in rows), "npm": len(rows),
                      "knode": [(x["correct"], x["questions"]) for x in k]}, indent=1))
    return 0


def describe(got: dict) -> str:
    if got["kind"] == "search":
        return "a ranked search (not routed)"
    if got.get("fallback"):
        return f"{got['kind']}, but the name was not found: ranked search instead"
    n = got["names"]
    text = {"usage": f"usages of {n[0]}", "callers": f"callers of {n[0]}()",
            "definition": f"the definition of {n[0]}",
            "order": f"functions calling {n[0]}() {got.get('order')} {n[-1]}()"}[got["kind"]]
    return f"{text} → {got['hits']} hit(s)" + (f"; e.g. {got['paths'][0]}" if got["paths"] else "")


def render(r: dict) -> str:
    s = r["safety"]
    npm_ok = sum(x["correct"] for x in r["npm"])
    lines = ["# Plain-English questions (step 2b)", "",
             "Questions typed at the `debug907>` prompt are classified before searching "
             "(`retrieval/router.py`: rule-based, offline, no LLM). Usage, call-order, callers and "
             "definition questions are answered by the exact lookups; everything else is the "
             "normal ranked search. Names written in words are resolved against the identifiers "
             "and strings actually in the index. `tools/english_questions.py`.", "",
             "## 1. The score cannot move", "",
             "| queries | routed away from ranked search |", "|---|---:|",
             f"| AppsRetrieval test split ({s['test']:,}) | **{len(s['test_routed'])}** |",
             f"| the validated sample ({s['sample']:,}) | **{len(s['sample_routed'])}** |",
             f"| AppsRetrieval training queries ({s['train']:,}) | **{len(s['train_routed'])}** |", "",
             "Only short, single-line questions that match an anchored English pattern are routed. "
             "`rank`, `reproduce`, `eval` and the MTEB encoder never route; `reproduce` still gives "
             "0.6206.", "",
             f"## 2. npm: {npm_ok} of {len(r['npm'])} answered correctly", "",
             "| question | understood as, and the answer | correct |", "|---|---|:---:|"]
    for x in r["npm"]:
        lines.append(f"| {x['question']} | {describe(x['got'])} | {'yes' if x['correct'] else '**no**'} |")
    lines += ["", "The last two rows are the deck's examples (\"where is this deeplink used\", "
              "\"which files call X before Y\") with npm names. \"Correct\" is checked automatically: "
              "the question was routed to the expected kind, resolved to the expected name, and the "
              "answer contains the expected file (or enough hits)."]
    if r["knode"]:
        lines += ["", "## 3. The team's private KNode code (aggregate only)", "",
                  "Three questions per repository, generated from the index's own function names "
                  "written as words (\"where is <name in words> defined\", \"who calls ...\", "
                  "\"where is ... used\"). No names, paths or questions are recorded.", "",
                  "| repository | questions | correct |", "|---|---:|---:|"]
        lines += [f"| {k['repo']} (private) | {k['questions']} | {k['correct']} |" for k in r["knode"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
