from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.naive_baseline import LIMIT_GB, working_set_gb

RESULTS = Path("bench/results")
STORE = RESULTS / "generalisation.json"
MAX_DOCS = 25_000

CANDIDATES = ["CodeSearchNetRetrieval", "StackOverflowQA", "DS1000Retrieval", "FreshStackRetrieval",
              "HumanEvalRetrieval", "MBPPRetrieval", "WikiSQLRetrieval", "CosQA",
              "COIRCodeSearchNetRetrieval", "CodeSearchNetCCRetrieval", "CodeFeedbackST",
              "CodeFeedbackMT", "SyntheticText2SQL", "SWEbenchCodeRetrieval", "RARbCode",
              "CodeTransOceanContest", "CodeTransOceanDL"]

PUBLISHED = {
    "StackOverflowQA": ("91.2", "model card CoIR table, seq 8,192; 90.88 in the MTEB results repo"),
    "CosQA": ("43.47", "model card CoIR table, seq 8,192; 42.18 in the MTEB results repo"),
    "AppsRetrieval": ("57.54", "model card CoIR table, seq 8,192; 56.41 in the MTEB results repo"),
}
KIND = {"COIRCodeSearchNetRetrieval": "NL→code", "CodeSearchNetCCRetrieval": "code→code",
        "CodeTransOceanContest": "code→code", "CodeTransOceanDL": "code→code",
        "CodeFeedbackST": "NL→code (chat)", "CodeFeedbackMT": "NL→code (multi-turn)",
        "SWEbenchCodeRetrieval": "issue→code", "RARbCode": "reasoning→code"}


def sizes(name: str) -> tuple[int | None, int | None]:
    import mteb
    st = mteb.get_task(name).metadata.descriptive_stats or {}
    test = st.get("test", {}) if isinstance(st, dict) else {}
    return test.get("num_documents"), test.get("num_queries")


def load_store() -> dict:
    return json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {"runs": {}, "catalog": []}


def save_store(store: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    STORE.write_bytes((json.dumps(store, indent=1) + "\n").encode("utf-8"))
    (RESULTS / "generalisation.md").write_bytes(render(store).encode("utf-8"))


def catalog() -> list[dict]:
    rows = []
    for name in CANDIDATES:
        try:
            docs, queries = sizes(name)
        except Exception as exc:
            rows.append({"task": name, "error": type(exc).__name__})
            continue
        rows.append({"task": name, "documents": docs, "queries": queries,
                     "kind": KIND.get(name, "NL→code"),
                     "runnable": docs is not None and docs <= MAX_DOCS
                     and KIND.get(name, "NL→code") == "NL→code"})
    return rows


def watchdog(state: dict) -> None:
    while True:
        gb = working_set_gb()
        state["peak"] = max(state.get("peak", 0.0), gb)
        if gb > LIMIT_GB:
            print(f"ABORT: memory {gb:.1f} GB > {LIMIT_GB} GB", flush=True)
            os._exit(3)
        time.sleep(2)


def run(name: str, subsets: list[str] | None, threads: int) -> dict:
    import mteb
    from mteb.cache import ResultCache
    from encoder import PrePostPipelineEncoder
    from retrieval.embed import set_threads

    set_threads(threads)
    task = mteb.get_task(name, hf_subsets=subsets) if subsets else mteb.get_task(name)
    model = PrePostPipelineEncoder()
    assert model.table_for(task.metadata) is None, "the AppsRetrieval hub table must be off here"
    state: dict = {}
    threading.Thread(target=watchdog, args=(state,), daemon=True).start()
    started = time.perf_counter()
    result = mteb.evaluate(model, [task], overwrite_strategy="always",
                           cache=ResultCache(cache_path="out/mteb_cache_generalisation"))
    minutes = (time.perf_counter() - started) / 60
    payload = list(result.task_results)[0].to_dict()
    tag = name + ("_" + "_".join(subsets) if subsets else "")
    Path("out").mkdir(exist_ok=True)
    Path(f"out/generalisation_{tag}.json").write_text(json.dumps(payload, indent=1, default=str),
                                                      encoding="utf-8")
    per_subset = {}
    for split, entries in payload["scores"].items():
        for s in entries:
            per_subset[s.get("hf_subset", "default")] = {
                "split": split, "ndcg_at_10": s["ndcg_at_10"], "mrr_at_10": s["mrr_at_10"],
                "recall_at_100": s.get("recall_at_100")}
    docs, queries = sizes(name)
    per = (mteb.get_task(name).metadata.descriptive_stats or {}).get("test", {}).get(
        "hf_subset_descriptive_stats", {}) or {}
    for sub, s in per_subset.items():
        if sub in per:
            s["documents"], s["queries"] = per[sub].get("num_documents"), per[sub].get("num_queries")
    return {"task": name, "subsets": subsets, "per_subset": per_subset,
            "minutes": round(minutes, 1), "peak_memory_gb": round(state.get("peak", 0.0), 1),
            "documents_all_subsets": docs, "queries_all_subsets": queries,
            "encoder": "PrePostPipelineEncoder() defaults: raw documents, seq 1024, fp32, no hub table",
            "finished_at": time.strftime("%Y-%m-%d %H:%M")}


def render(store: dict) -> str:
    lines = ["# Generalisation: held-out MTEB code-retrieval tasks (H1)", "",
             "The submitted encoder, `PrePostPipelineEncoder()` with its defaults (gte-modernbert-base "
             "@ e7f32e3c, fp32, seq 1024, raw documents), through the official `mteb.evaluate`. "
             "**No tuning on any of these tasks**, and the AppsRetrieval hub table is off (the "
             "script asserts it). `tools/generalisation.py`. Every task here is held out: none was "
             "used for any choice.", "",
             "| task | subset | docs | queries | NDCG@10 | MRR@10 | R@100 | encode + eval | published (source) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    rows = []
    for key, r in store.get("runs", {}).items():
        pub, src = PUBLISHED.get(r["task"], (None, None))
        for sub, s in r["per_subset"].items():
            rows.append((r["task"], sub, s.get("documents", r.get("documents_all_subsets")),
                         s.get("queries", r.get("queries_all_subsets")), s,
                         f"{r['minutes']} min" + (f" (one run, {len(r['per_subset'])} languages)"
                                                  if len(r["per_subset"]) > 1 else ""),
                         f"{pub} ({src})" if pub else "none comparable"))
    for task, sub, d, q, s, mins, pub in rows:
        r100 = f"{s['recall_at_100'] * 100:.2f}" if s.get("recall_at_100") is not None else "-"
        lines.append(f"| {task} | {sub} | {d if d is not None else '-'} | {q if q is not None else '-'} | "
                     f"**{s['ndcg_at_10'] * 100:.2f}** | {s['mrr_at_10'] * 100:.2f} | {r100} | "
                     f"{mins} | {pub} |")
    cosqa = RESULTS / "second_benchmark.json"
    if cosqa.exists():
        c = json.loads(cosqa.read_text(encoding="utf-8"))["raw"]
        lines.append(f"| CosQA (`second_benchmark.md`) | default | 20,604 | 500 | "
                     f"**{c['ndcg_at_10'] * 100:.2f}** | {c['mrr_at_10'] * 100:.2f} | - | {c['minutes']} min | "
                     "43.47 (model card CoIR table, seq 8,192; 42.18 in the MTEB results repo) |")
    lines += ["", "Scores ×100. Times include encoding on a laptop CPU (8 threads), with "
              "vectors cached by text.", ""]
    notes = store.get("notes", {})
    if notes:
        lines += ["## Reading each result", ""] + [f"- **{k}**: {v}" for k, v in notes.items()] + [""]
    cat = store.get("catalog", [])
    if cat:
        lines += ["## Every candidate task in the installed MTEB", "",
                  "| task | kind | documents | queries | run? |", "|---|---|---:|---:|---|"]
        done = {r["task"] for r in store.get("runs", {}).values()} | {"CosQA"}
        for c in cat:
            if "error" in c:
                lines.append(f"| {c['task']} | - | - | - | not available ({c['error']}) |")
                continue
            why = ("run" if c["task"] in done else
                   "not run: over 25,000 documents" if (c["documents"] or 0) > MAX_DOCS else
                   "not run: not natural language → code" if c["kind"] != "NL→code" else
                   "not run (time)")
            lines.append(f"| {c['task']} | {c['kind']} | {c['documents'] or '-'} | "
                         f"{c['queries'] or '-'} | {why} |")
        lines.append("")
    lines += ["The model card's CodeSearchNet-<language> numbers are for CoIR's CodeSearchNet "
              "(`COIRCodeSearchNetRetrieval`, about one million documents), a different and harder "
              "corpus than MTEB's `CodeSearchNetRetrieval` (1,000 per language), so they are not "
              "listed as comparable.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--subsets", default=None, help="comma-separated hf_subsets")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    store = load_store()
    if args.list:
        store["catalog"] = catalog()
        save_store(store)
        for c in store["catalog"]:
            print(json.dumps(c, ensure_ascii=True))
        return 0
    if args.render:
        save_store(store)
        return 0
    subsets = args.subsets.split(",") if args.subsets else None
    r = run(args.task, subsets, args.threads)
    key = args.task + ("[" + ",".join(subsets) + "]" if subsets else "")
    store.setdefault("runs", {})[key] = r
    save_store(store)
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
