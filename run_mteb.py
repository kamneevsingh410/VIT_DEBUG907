from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from retrieval.embed import set_threads

def run_mteb(args: argparse.Namespace) -> int:
    try:
        import mteb
    except ImportError:
        print("mteb is not installed. Run: pip install -r requirements.txt")
        return 1

    from encoder import PrePostPipelineEncoder, describe_api

    print("Installed MTEB API (read from the installed library, not assumed):")
    describe_api()
    print(f"\nthreads: {set_threads(args.threads)}")

    from retrieval.embed import MODELS
    spec = MODELS[args.model]
    hub_table = None
    if args.hubness and not args.no_hubness:
        hub_table = json.loads(Path(args.hubness).read_text(encoding="utf-8"))
        want = "raw" if args.raw_docs else "code"
        if hub_table.get("document_text", "code") != want:
            print(f"the hub table {args.hubness} is for {hub_table.get('document_text', 'code')!r} "
                  f"documents, but this run embeds {want!r} documents")
            return 1
        print(f"hubness correction ON: {hub_table['source']} k={hub_table['k']} "
              f"beta={hub_table['beta']} ({len(hub_table['hubs']):,} documents)")
    model = PrePostPipelineEncoder(model_name=spec.name, view=args.view, hub_table=hub_table,
                                   raw_documents=args.raw_docs)
    model.embedder.quantize = args.quantize
    model.embedder.prefix = spec.doc_prefix
    model.embedder.batch_size = 32
    model.embedder.max_seq_length = args.seq
    print(f"encoder: {spec.name}  view={'raw' if args.raw_docs else args.view}  "
          f"precision={'int8' if args.quantize else 'fp32'}")
    started = time.perf_counter()

    result = None
    errors: list[str] = []
    for attempt in ("evaluate", "MTEB"):
        try:
            if attempt == "evaluate" and hasattr(mteb, "evaluate"):
                task = mteb.get_task(args.task)
                from mteb.cache import ResultCache
                result = mteb.evaluate(model, [task],
                                       overwrite_strategy="always",
                                       cache=ResultCache(cache_path="out/mteb_cache"),
                                       prediction_folder=args.predictions)
                break
            if attempt == "MTEB":
                tasks = mteb.get_tasks(tasks=[args.task])
                result = mteb.MTEB(tasks=tasks).run(model, output_folder="out/mteb")
                break
        except Exception as exc:
            errors.append(f"{attempt}: {type(exc).__name__}: {exc}")

    if result is None:
        print("Could not run MTEB with any known call shape:")
        for line in errors:
            print(f"  {line}")
        return 1

    elapsed = time.perf_counter() - started
    task_result = list(getattr(result, "task_results", result))[0]
    payload = task_result.to_dict() if hasattr(task_result, "to_dict") else task_result

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwall clock: {elapsed / 60:.1f} min")
    print(f"wrote {out}")
    if out.resolve() == Path("appsretrieval_results.json").resolve():
        print("\nATTACH THIS FILE TO A GITHUB RELEASE - without it there is no screening.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="official MTEB AppsRetrieval run")
    parser.add_argument("--task", default="AppsRetrieval",
                        help="MTEB task (default AppsRetrieval, the screening task); another "
                             "CoIR task, e.g. CosQA, runs held out with no hubness table")
    parser.add_argument("--raw-docs", action="store_true",
                        help="accepted for compatibility: raw documents are the default")
    parser.add_argument("--code-view", action="store_true",
                        help="the previous config: code-view documents with their own hub table "
                             "(0.53926, recorded in appsretrieval_results.meta.json)")
    parser.add_argument("--mteb", action="store_true",
                        help="accepted for compatibility; MTEB is the only mode")
    parser.add_argument("--quantize", action="store_true",
                        help="int8 dynamic quantisation: ~2.8x faster on CPU")
    parser.add_argument("--model", default="gte-modernbert",
                        help="encoder key from retrieval.embed.MODELS")
    parser.add_argument("--seq", type=int, default=1024,
                        help="max_seq_length; 1024 measured +0.0186 "
                             "over 512 on gte-modernbert (p~0.000)")
    parser.add_argument("--view", default="code",
                        help="which document view Track A embeds (nl|code|full)")
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads (8 by day; the machine is shared)")
    parser.add_argument("--out", default="out/mteb_appsretrieval_results.json",
                        help="where to write the result JSON (the shipped artifact is "
                             "appsretrieval_results.json; pass it explicitly to replace it)")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="also save MTEB's per-query predictions to this folder")
    parser.add_argument("--hubness", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "apps_hubness.json",
                        help="hub table (tools/hubness_eval.py) applied inside the encoder "
                             "(default: the committed data/apps_hubness.json)")
    parser.add_argument("--no-hubness", action="store_true",
                        help="plain cosine, as in the previous artifact (0.50431)")
    args = parser.parse_args(argv)
    args.raw_docs = not args.code_view
    if args.code_view and args.hubness == parser.get_default("hubness"):
        args.hubness = args.hubness.with_name("apps_hubness_codeview.json")
    if args.task != "AppsRetrieval" and not args.no_hubness:
        print(f"hubness correction OFF: its table is for AppsRetrieval, not {args.task}")
        args.no_hubness = True
    return run_mteb(args)


if __name__ == "__main__":
    raise SystemExit(main())
