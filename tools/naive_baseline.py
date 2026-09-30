from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("bench/results")
NAME = "Alibaba-NLP/gte-modernbert-base"
LIMIT_GB = 12.0


def working_set_gb() -> float:
    if os.name != "nt":
        return 0.0

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return 0.0
    return pmc.WorkingSetSize / 1e9


def watchdog(peak: dict, tag: str = "") -> None:
    while True:
        gb = working_set_gb()
        peak["gb"] = max(peak.get("gb", 0.0), gb)
        if gb > LIMIT_GB:
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / f"naive_baseline{tag}_aborted.json").write_text(json.dumps(
                {"aborted": f"memory {gb:.1f} GB > {LIMIT_GB} GB limit"}), encoding="utf-8")
            print(f"ABORT: memory {gb:.1f} GB > {LIMIT_GB} GB", flush=True)
            os._exit(3)
        time.sleep(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pinned", action="store_true",
                        help="the stock SentenceTransformer on this repo's pinned snapshot (the "
                             "model card's current revision) instead of mteb.get_model, whose "
                             "registered revision predates the sentence-transformers config and "
                             "silently falls back to MEAN pooling")
    parser.add_argument("--render", action="store_true",
                        help="only re-render the report(s) from their JSON, with the current "
                             "artifact numbers")
    args = parser.parse_args(argv)
    if args.render:
        for tag in ("", "_pinned"):
            path = RESULTS / f"naive_baseline{tag}.json"
            if path.exists():
                report = {**json.loads(path.read_text(encoding="utf-8")), **reference_rows()}
                report["beats_ours"] = report["naive"]["ndcg_at_10"] > report["ours"]["ndcg_at_10"]
                path.write_bytes((json.dumps(report, indent=1) + "\n").encode("utf-8"))
                (RESULTS / f"naive_baseline{tag}.md").write_bytes(render(report).encode("utf-8"))
        return 0
    import mteb
    import torch
    from mteb.cache import ResultCache

    torch.set_num_threads(args.threads)
    peak: dict = {}
    threading.Thread(target=watchdog, args=(peak, "_pinned" if args.pinned else ""),
                     daemon=True).start()
    started = time.perf_counter()
    how = "mteb.get_model (as MTEB registers it)"
    try:
        if args.pinned:
            raise RuntimeError("--pinned")
        model = mteb.get_model(NAME, device="cpu")
    except Exception as exc:
        from retrieval.embed import ensure_model
        from sentence_transformers import SentenceTransformer
        how = ("stock SentenceTransformer on the pinned snapshot, default settings" if args.pinned
               else f"stock SentenceTransformer on the pinned local snapshot (mteb.get_model "
                    f"failed: {type(exc).__name__})")
        model = SentenceTransformer(str(ensure_model(NAME)), device="cpu")
    st = getattr(model, "model", model)
    max_len = getattr(st, "max_seq_length", None)
    revision = getattr(getattr(model, "mteb_model_meta", None), "revision", None)
    print(f"model: {how}; max_seq_length={max_len}; revision={revision}", flush=True)
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(model, [task], overwrite_strategy="always",
                           cache=ResultCache(cache_path="out/mteb_cache_naive" + ("_pinned" if args.pinned else "")),
                           encode_kwargs={"batch_size": args.batch_size},
                           prediction_folder="out/mteb_naive_predictions" + ("_pinned" if args.pinned else ""))
    task_result = list(getattr(result, "task_results", result))[0]
    payload = task_result.to_dict() if hasattr(task_result, "to_dict") else task_result
    tag = "_pinned" if args.pinned else ""
    Path(f"out/mteb_naive{tag}_results.json").write_text(json.dumps(payload, indent=2, default=str),
                                                   encoding="utf-8")
    s = payload["scores"]["test"][0]
    report = {"how": how, "max_seq_length": max_len, "revision": revision,
              "pinned": bool(args.pinned),
              "naive": {k: s[k] for k in KEYS}, **reference_rows(),
              "minutes": round((time.perf_counter() - started) / 60, 1),
              "peak_memory_gb": round(peak.get("gb", 0.0), 1), "batch_size": args.batch_size}
    report["beats_ours"] = s["ndcg_at_10"] > report["ours"]["ndcg_at_10"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"naive_baseline{tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / f"naive_baseline{tag}.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


KEYS = ("ndcg_at_10", "mrr_at_10", "recall_at_100")


def reference_rows() -> dict:
    ours = json.loads(Path("appsretrieval_results.json").read_text(encoding="utf-8"))["scores"]["test"][0]
    meta = json.loads(Path("appsretrieval_results.meta.json").read_text(encoding="utf-8"))
    plain = next(p for p in meta["previous_artifacts"]
                 if "no-hubness" in p.get("configuration", "") and "code-view" in p["configuration"])
    raw = json.loads((RESULTS / "doc_variant_raw.json").read_text(encoding="utf-8"))
    return {"ours": {k: ours[k] for k in KEYS}, "ours_plain_cosine": {k: plain[k] for k in KEYS},
            "raw_plain_1024_ndcg": raw["groups"]["full"]["raw"]["ndcg_at_10"]}


def pinned_result() -> str:
    path = RESULTS / "naive_baseline_pinned.json"
    if not path.exists():
        return ", once that run has completed"
    n = json.loads(path.read_text(encoding="utf-8"))["naive"]
    return f": NDCG@10 {n['ndcg_at_10']:.5f}, MRR@10 {n['mrr_at_10']:.5f}"


def render(r: dict) -> str:
    n, o, p = r["naive"], r["ours"], r["ours_plain_cosine"]
    pinned = bool(r.get("pinned"))
    verdict = ("**This naive baseline BEATS the submitted artifact.**" if r["beats_ours"]
               else "The submitted artifact beats this naive baseline.")
    what = ("The stock SentenceTransformer on this repo's pinned snapshot (the model card's "
            "current revision, its own CLS pooling), default settings." if pinned else
            "`mteb.get_model` exactly as MTEB registers the model.")
    lines = [
        "# The naive baseline: the model as downloaded" + (", loaded correctly" if pinned else ""),
        "", verdict, "", what, "",
        f"`tools/naive_baseline.py{' --pinned' if pinned else ''}`: {r['how']}; max sequence "
        f"length {r['max_seq_length']}, revision {r['revision']}, documents raw, no "
        "preprocessing, no hub table, official `mteb.evaluate` on AppsRetrieval (encode batch "
        f"size {r['batch_size']}, which does not change the vectors). {r['minutes']} min, peak "
        f"memory {r['peak_memory_gb']} GB.", "",
        "| system | NDCG@10 | MRR@10 | R@100 |", "|---|---:|---:|---:|",
        f"| naive: {'stock model, pinned, CLS pooling' if pinned else 'mteb.get_model as registered'}, "
        f"seq {r['max_seq_length']} | {n['ndcg_at_10']:.5f} | {n['mrr_at_10']:.5f} | "
        f"{n['recall_at_100']:.5f} |",
        f"| ours, submitted (raw documents, seq 1024, hubness correction) | {o['ndcg_at_10']:.5f} | "
        f"{o['mrr_at_10']:.5f} | {o['recall_at_100']:.5f} |",
        f"| ours, earlier artifact (code view, seq 1024, plain cosine) | {p['ndcg_at_10']:.5f} | "
        f"{p['mrr_at_10']:.5f} | {p['recall_at_100']:.5f} |", "",
        f"For scale: our raw-document plain-cosine run at 1,024 tokens (this repo's loader and "
        f"evaluation, not a stock 8,192-token run) scores {r['raw_plain_1024_ndcg']:.4f} on the full "
        "split (`doc_variant_raw.md`).", "",
        "Published on the model card for the same model (CoIR apps, seq 8,192): 57.54.", ""]
    if not pinned:
        lines += [
            "## Why so low: MTEB's registry loads an old revision with the wrong pooling", "",
            "`mteb.get_model` pins revision `7ca8b4ca`, which predates the model's "
            "sentence-transformers config. Loading it prints \"No sentence-transformers model "
            "found ... Creating a new one with mean pooling\". gte-modernbert is trained with CLS "
            "pooling (its current `1_Pooling` config), so these vectors are not the model's. "
            "Encoded side by side with this repo's pinned snapshot, the same 12 AppsRetrieval "
            "documents come out at cosine 0.22-0.52 and 6 queries at 0.41-0.58, at any batch size "
            "and at sequence length 1,024 too.", "",
            "So this row is what a team gets by calling `mteb.get_model` as registered, not what "
            "the model can do. The correctly loaded stock model at its default 8,192 tokens is "
            "`naive_baseline_pinned.md` (`tools/naive_baseline.py --pinned`)" + pinned_result() + ".", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
