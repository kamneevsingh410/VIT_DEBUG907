from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bench.dataset import describe, load
from retrieval.display import safe
from retrieval.index import build, load as load_index
from retrieval.embed import Embedder, check_environment, configure_hub_environment
from retrieval.rerank import RerankConfig

configure_hub_environment()
from retrieval.search import SearchConfig, search

DEFAULT_DB = Path("out/real.db")
SELFTEST_DB = Path("out/selftest.db")


def _snippet_preview(text: str, lines: int = 4) -> list[str]:
    body = [line for line in text.splitlines() if line.strip()]
    return body[:lines]


def cmd_describe(args: argparse.Namespace) -> int:
    corpus, queries, qrels, source = load(limit=args.limit)
    print(f"dataset: {source}\n")
    print(describe(corpus, queries, qrels))
    print("\nfirst three queries:")
    for qid, text in list(queries.items())[:3]:
        print(f"  {qid}  {safe(text)}")
    print("\nfirst snippet:")
    first = next(iter(corpus.values()))
    for line in _snippet_preview(first, 6):
        print(f"  | {safe(line)}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from retrieval import workflows
    from retrieval.embed import set_threads

    print(f"threads   {set_threads(args.threads)}")
    try:
        report = workflows.index_apps(args.db, limit=args.limit, show_progress=True,
                                      reset=args.reset, drop_vectors=args.drop_vectors)
    except RuntimeError as exc:
        print(f"VERIFY FAILED: {exc}")
        return 1
    except Exception as exc:
        print(f"cannot index CoIR AppsRetrieval: {type(exc).__name__}: {exc}")
        return 1
    print(f"dataset   {report.source}")
    print(f"database  {args.db}  (version {report.version})")
    _print_build(report)
    return 0


def version_info() -> list[str]:
    import subprocess
    try:
        import tomllib
        version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    except Exception:
        version = "unknown"
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        commit = (commit + (" (uncommitted changes)" if dirty else "")) if commit else ""
    except (OSError, subprocess.SubprocessError):
        commit = ""
    from retrieval.embed import MAX_SEQ_LENGTH, MODEL_NAME, QUANTIZE, _spec
    spec = _spec(MODEL_NAME)
    return [f"debug907 {version}",
            f"  commit         {commit or 'unknown (not a git checkout, e.g. the Docker image)'}",
            f"  model          {MODEL_NAME} @ {spec.revision[:8] if spec and spec.revision else '?'}",
            f"  configuration  seq {MAX_SEQ_LENGTH}, {'int8' if QUANTIZE else 'fp32'}, CPU, dense cosine; "
            "AppsRetrieval: raw documents + hubness correction (k 20, beta 0.75); "
            "repositories: code view, no correction",
            f"  python         {sys.version.split()[0]}"]


def cmd_export_index(args: argparse.Namespace) -> int:
    from retrieval import prebuilt
    out = Path(args.file)
    if out.resolve() in PROTECTED or (ROOT / "data") in out.resolve().parents:
        print(f"refused: {out} is a shipped file of this repository")
        return 1
    try:
        entry = prebuilt.export_index(args.db, out)
    except prebuilt.PrebuiltError as exc:
        print(f"refused: {exc}")
        return 1
    print(json.dumps(entry, indent=1))
    if args.pin:
        prebuilt.MANIFEST.write_bytes((json.dumps(entry, indent=1) + "\n").encode("utf-8"))
        print(f"pinned in {prebuilt.MANIFEST.relative_to(ROOT)}")
    return 0


def cmd_import_index(args: argparse.Namespace) -> int:
    from retrieval import prebuilt
    why = output_refused(Path(args.db), "db")
    if why:
        print(f"refused: {why}")
        return 1
    print(f"importing {args.source}")
    try:
        prebuilt.import_index(args.source, Path(args.db), expected_sig=Embedder().signature,
                              force=args.force)
    except prebuilt.PrebuiltError as exc:
        print(f"REFUSED: {exc}")
        return 1
    except OSError as exc:
        print(f"cannot read or download it: {exc}")
        return 1
    print("\ncheck it yourself: debug907 verify-index --db " + str(args.db)
          + "\nthen:              debug907 reproduce --db " + str(args.db))
    return 0


def cmd_verify_index(args: argparse.Namespace) -> int:
    from retrieval import prebuilt
    from retrieval.embed import set_threads
    print(f"threads   {set_threads(args.threads)}")
    if not Path(args.db).exists():
        print(f"no index at {args.db}")
        return 1
    try:
        r = prebuilt.verify_index(Path(args.db), sample=args.sample, seed=args.seed)
    except prebuilt.PrebuiltError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"index     {r['db']}")
    print(f"checked   {r['checked']} of {r['sampled']} sampled snippets (seed {args.seed})")
    if r["min_cosine"] is not None:
        print(f"cosine    min {r['min_cosine']:.7f}  mean {r['mean_cosine']:.7f}  (worst {r['worst']})")
    if r["key_mismatch"]:
        print(f"text      {len(r['key_mismatch'])} snippet(s) whose stored text key does not "
              f"match their source, e.g. {r['key_mismatch'][:3]}")
    if r["missing_vectors"]:
        print(f"vectors   {len(r['missing_vectors'])} missing")
    print("VERIFY PASSED - the sampled vectors are what this code produces from the source"
          if r["ok"] else "VERIFY FAILED - the index does not match what this code produces")
    return 0 if r["ok"] else 1


def _print_build(report) -> None:
    st = report.stats
    print(f"  snippets         {report.snippets} (verified on disk)")
    print(f"  vectors new      {st.encoded_vectors}   reused from this index "
          f"{st.reused_vectors} ({st.as_dict()['reuse_pct']}%)")
    print(f"  texts            {st.model_encoded} run through the model, "
          f"{st.text_cache_hits} served by the text cache")
    print(f"  time             {report.seconds:.0f}s (encode {st.encode_ms / 1000:.1f}s)")


EXPECTED_SAMPLE_NDCG = 0.6206
REPRODUCE_TOLERANCE = 0.003
FULL_TOLERANCE = 0.001


def full_split_targets(path: Path = Path("appsretrieval_results.json")) -> dict[str, float]:
    scores = json.loads(Path(path).read_text(encoding="utf-8"))["scores"]["test"][0]
    return {"ndcg_at_10": round(scores["ndcg_at_10"], 5),
            "mrr_at_10": round(scores["mrr_at_10"], 5)}


def compare_mteb_result(path: Path) -> tuple[bool, list[str]]:
    got = json.loads(Path(path).read_text(encoding="utf-8"))["scores"]["test"][0]
    ok, lines = True, []
    for key, target in full_split_targets().items():
        delta = got[key] - target
        ok &= abs(delta) <= FULL_TOLERANCE
        lines.append(f"          {key:<11} expected {target:.5f} +/- {FULL_TOLERANCE}   "
                     f"got {got[key]:.5f} ({delta:+.5f})")
    return ok, lines


def reproduce_full_mteb(threads: int) -> bool:
    import run_mteb

    out = Path("out/reproduce_full_mteb.json")
    print("mteb      running the official MTEB evaluation on the full test split ...")
    if run_mteb.main(["--threads", str(threads), "--out", str(out)]) != 0:
        print("mteb      FAILED to run")
        return False
    ok, lines = compare_mteb_result(out)
    print(f"mteb      {out}")
    print("\n".join(lines))
    return ok


def cmd_reproduce(args: argparse.Namespace) -> int:
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from retrieval.embed import set_threads

    print(f"threads   {set_threads(args.threads)}")
    if not args.db.exists():
        print(f"no index at {args.db}. Build it first: debug907 index --db {args.db}")
        return 1
    corpus, queries, qrels = load_appsretrieval()
    if args.full:
        queries = {q: t for q, t in queries.items() if q in qrels}
    else:
        picked = set(load_stratified())
        queries = {q: t for q, t in queries.items() if q in picked}
        qrels = {q: r for q, r in qrels.items() if q in picked}

    from retrieval import workflows

    index = load_index(args.db)
    try:
        if len(index.ids) != len(corpus):
            print(f"index has {len(index.ids)} snippets but the corpus has {len(corpus)}; "
                  "rebuild it with --real index")
            return 1
        sig = index.conn.execute(
            "SELECT value FROM meta WHERE key='embedder_sig'").fetchone()
        print(f"index     {args.db}  ({len(index.ids)} snippets, {sig[0] if sig else '?'})")
        embedder = Embedder()
        print(f"queries   {len(queries)} "
              + ("(the full test split)" if args.full else "(validated stratified sample)"))
        result = workflows.evaluate_shipped(index, queries, qrels, embedder,
                                            show_progress=True)
        run, scores = result.run, result.scores

        example = next(q for q in sorted(queries) if run[q] and run[q][0] in qrels[q])
        gold = next(iter(qrels[example]))
        print(f"\nexample   {queries[example][:150].strip()}...")
        print(f"          relevant snippet {gold}")
        for rank, sid in enumerate(run[example][:3], start=1):
            mark = "  <- relevant" if sid in qrels[example] else ""
            first = next((ln for ln in index.content(sid).splitlines() if ln.strip()), "")
            print(f"   {rank}. {sid:<8} {safe(first[:70])}{mark}")
    finally:
        index.close()

    if args.full:
        from tools.score_csv import score as artifact_metrics
        got = artifact_metrics(run, qrels)
        print("\nlocal     the tool's own search path, same metric definitions:")
        ok = True
        print(f"result    {scores}")
        for key, target in full_split_targets().items():
            delta = got[key] - target
            ok &= abs(delta) <= FULL_TOLERANCE
            print(f"          {key:<11} expected {target:.5f} +/- {FULL_TOLERANCE}   "
                  f"got {got[key]:.5f} ({delta:+.5f})")
        print(f"          query p50 {result.p50_ms:.1f} ms   total {result.seconds:.0f}s\n")
        ok &= reproduce_full_mteb(args.threads)
        if not ok:
            print("REPRODUCE FAILED - the full split does not match appsretrieval_results.json")
            return 1
        print("REPRODUCE PASSED - official MTEB path and the tool's search path both "
              "match the shipped artifact on the full test split")
        return 0

    delta = scores.ndcg_at_10 - EXPECTED_SAMPLE_NDCG
    print(f"\nresult    {scores}")
    print(f"          expected NDCG@10 {EXPECTED_SAMPLE_NDCG:.4f} +/- {REPRODUCE_TOLERANCE}"
          f"   got {scores.ndcg_at_10:.4f} ({delta:+.4f})")
    print(f"          query p50 {result.p50_ms:.1f} ms   "
          f"total {result.seconds:.0f}s")
    if abs(delta) > REPRODUCE_TOLERANCE:
        print("REPRODUCE FAILED - the pipeline does not reproduce the shipped number")
        return 1
    print("REPRODUCE PASSED - real corpus, real encoder, shipped number reproduced")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from retrieval import workflows

    if not workflows.index_is_usable(args.db):
        print(f"no index at {args.db}")
        print("  build one:  debug907 index-repo <folder>   or   debug907 index")
        return 1
    index = load_index(args.db, versions=_versions_arg(args))
    try:
        embedder = Embedder()
        load_started = time.perf_counter()
        embedder.warm()
        load_ms = (time.perf_counter() - load_started) * 1000

        started = time.perf_counter()
        if args.track == "a":
            repo = getattr(index, "corpus_kind", "") == "repo"
            result = workflows.shipped_search(index, args.query, embedder, top_k=args.top_k,
                                              include_tests=args.include_tests or not repo,
                                              hubness=False if args.no_hubness else None)
        else:
            config = SearchConfig(
                rerank=RerankConfig() if not args.no_rerank else RerankConfig.off(),
                collapse_versions=args.collapse)
            result = search(index, args.query, config, top_k=args.top_k,
                            embedder=embedder)
        total_ms = (time.perf_counter() - started) * 1000

        print(f"model load {load_ms / 1000:.1f}s (once per process; not part of query latency)")
        print(f'query     "{args.query}"')
        print(f"track     {args.track.upper()}   category: {result.processed.category}")
        print(f"latency   {total_ms:.2f} ms total  "
              + "  ".join(f"{k}={v:.2f}" for k, v in result.timings_ms.items()
                          if k != "total"))
        print()
        for note in result.notes:
            print(f"note      {note}")
        if not result.hits:
            print("no results")
            return 0
        for rank, (snippet_id, score) in enumerate(result.hits, start=1):
            category = index.categories.get(snippet_id, "?")
            view = ""
            if result.query_vector:
                from retrieval.dense import best_view
                vname, vscore = best_view(index, result.query_vector, snippet_id)
                view = f"  via {vname} cos={vscore:.3f}" if vname else ""
            where = ""
            if len(index.loaded_versions) > 1:
                found = result.versions.get(snippet_id) or [index.version_of[snippet_id]]
                where = f"  (in {', '.join(found)})"
            print(safe(f"{rank:>3}. {snippet_id:<10} score={score:.4f}  [{category}]{view}{where}"))
            for line in _snippet_preview(index.content(snippet_id), args.lines):
                print(f"       | {safe(line)}")
        if args.explain:
            print("\nexplain")
            print(f"  category    {result.processed.category}")
            print(f"  identifiers {result.processed.identifiers}")
            print(f"  expansions  {result.processed.expansions[:12]}")
            print(f"  tokens      {result.processed.tokens[:16]}")
            print(f"  stages      {json.dumps(result.stage_counts)}")
    finally:
        index.close()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.qrels or args.rankings:
        if not (args.qrels and args.rankings):
            print("eval with your own labels needs both --qrels <file> and --rankings <csv>")
            return 2
        return cmd_eval_qrels(args)
    from bench.dataset import load_appsretrieval
    from retrieval import workflows
    from retrieval.embed import set_threads

    print(f"threads   {set_threads(args.threads)}")
    if not workflows.index_is_usable(args.db):
        print(f"no usable index at {args.db}. Build it: debug907 index --db {args.db}")
        return 1
    corpus, queries, qrels = load_appsretrieval()
    if args.sample:
        from bench.stratified import load as load_stratified
        picked = set(load_stratified())
        queries = {q: t for q, t in queries.items() if q in picked}
    queries = {q: t for q, t in queries.items() if q in qrels}
    if args.limit:
        queries = dict(list(queries.items())[:args.limit])
    qrels = {q: qrels[q] for q in queries}
    index = load_index(args.db)
    try:
        if len(index.ids) != len(corpus):
            print(f"index has {len(index.ids)} snippets, corpus has {len(corpus)}; "
                  "rebuild it with debug907 index")
            return 1
        result = workflows.evaluate_shipped(index, queries, qrels, Embedder(),
                                            show_progress=True)
    finally:
        index.close()
    print(f"dataset   CoIR AppsRetrieval (test), {len(queries)} queries"
          + (" (stratified sample)" if args.sample else ""))
    print("config    shipped: gte-modernbert-base, code view, dense only")
    print(f"result    {result.scores}")
    print(f"          query p50 {result.p50_ms:.1f} ms   total {result.seconds:.0f}s")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from tools.real_ablation import main as ablate_main
    return ablate_main(["--slow"] if args.slow else [])


def cmd_reindex(args: argparse.Namespace) -> int:
    from retrieval.versions import reindex

    corpus, _q, _r, source = load(limit=args.limit)
    v2 = dict(corpus)
    keys = sorted(v2)
    touched = max(1, int(len(keys) * args.change_pct / 100.0))
    for key in keys[:touched]:
        v2[key] = v2[key] + "\n    # revised in v2\n"

    print(f"dataset   {source}")
    print(f"changing  {touched}/{len(keys)} snippets ({args.change_pct}%)\n")
    report = reindex(corpus, v2, Path("out/versions.db"))
    print(report.summary())
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    import app

    db = args.db if getattr(args, "db_explicit", False) else None
    return app.main(db=db, debug=args.debug, versions=_versions_arg(args))


def cmd_doctor(_args: argparse.Namespace) -> int:
    print("debug907 doctor")
    info = version_info()
    print(f"  version               {info[0].split()[-1]}")
    for line in info[1:4]:
        label, _, value = line.strip().partition(" ")
        print(f"  {label:<21} {value.strip()}")
    print(f"  python                {sys.version.split()[0]}")
    import sqlite3
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(a)")
        print("  sqlite FTS5           available")
    except sqlite3.OperationalError:
        print("  sqlite FTS5           MISSING (BM25 arm disabled)")
    finally:
        conn.close()

    ok, message = check_environment()
    print(f"  encoder deps          {'OK' if ok else 'PROBLEM'}: {message}")
    if not ok:
        return 1
    try:
        emb = Embedder()
        print(f"  loading {emb.name} ...")
        dim = emb.dimension
        print(f"  bi-encoder            loaded, dim={dim}, max_seq={emb.max_seq_length}")
    except Exception as exc:
        print(f"  bi-encoder            FAILED: {type(exc).__name__}: {exc}")
        return 1
    print("  result                READY")
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    print("debug907 selftest  [FIXTURE: 30-snippet bundled sample, not real data]")
    print(f"  python           {sys.version.split()[0]}")

    import sqlite3
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(a)")
        print("  sqlite FTS5      available")
    except sqlite3.OperationalError:
        print("  sqlite FTS5      MISSING - BM25 arm will be disabled")
        return 1
    finally:
        conn.close()

    corpus, queries, qrels, source = load(fixture=True)
    print(f"  dataset          {source}")
    db = SELFTEST_DB
    if db.exists():
        db.unlink()
    ok, message = check_environment()
    print(f"  encoder deps     {'OK' if ok else 'unavailable'}: {message}")
    stats = build(corpus, db, use_embeddings=ok, show_progress=False)
    print(f"  index            {stats.snippets} snippets in {stats.elapsed_ms} ms "
          f"({stats.encoded_vectors} vectors encoded)")

    index = load_index(db)
    try:
        from bench.metrics import evaluate
        run = {qid: search(index, text, SearchConfig(), top_k=100).ids
               for qid, text in queries.items()}
        scores = evaluate(run, qrels)
        print(f"  retrieval        {scores}")
    finally:
        index.close()

    ok = scores.ndcg_at_10 > 0.5
    print(f"  result           {'PASS' if ok else 'FAIL'} (FIXTURE smoke test, not a score)")
    return 0 if ok else 1


QUICK_START = """\
quick start:
  debug907                                  the terminal tool
  debug907 search "reverse a linked list"   one-off query
  debug907 index-repo C:\\path\\to\\project    index your own JS/Python code
  debug907 sync C:\\path\\to\\repo [--watch]   index each git commit as a version
  debug907 rank --queries q.jsonl --out r.csv  rank a file of queries (batch)
  debug907 doctor                           check the environment
  debug907 selftest                         end-to-end smoke test
"""


def cmd_index_repo(args: argparse.Namespace) -> int:
    from retrieval import workflows
    from retrieval.embed import set_threads

    root = Path(args.path)
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1
    print(f"threads   {set_threads(args.threads)}")
    ok, message = check_environment()
    print(f"encoder   {message}")
    if not ok:
        print("cannot index without the encoder. Run: debug907 doctor")
        return 1
    db = Path(args.out)
    if refuse(db, "db"):
        return 1
    try:
        report = workflows.index_folder(root, db, version=args.version,
                                        include_vendored=args.include_vendored,
                                        all_languages=args.all_languages,
                                        show_progress=True)
    except ValueError as exc:
        print(exc)
        return 1
    print(f"\nindexed   {db}  (version {report.version})")
    _print_build(report)
    print(f"\nsearch it: debug907 interactive --db {db}")
    print(f"       or: debug907 search \"<your question>\" --db {db}")
    return 0


ROOT = Path(__file__).resolve().parent
PROTECTED = {ROOT / name for name in (
    "appsretrieval_results.json", "appsretrieval_rankings.csv", "appsretrieval_results.meta.json",
    "requirements.txt", "Dockerfile", "README.md")}


def output_refused(path: Path, kind: str) -> str | None:
    p = Path(path).resolve()
    if p in PROTECTED or p.name.endswith(".backup.json") or (ROOT / "data") in p.parents:
        return f"{path} is a shipped file of this repository"
    if not p.exists():
        return None
    if p.is_dir():
        return f"{path} is a folder"
    head = p.read_bytes()[:64]
    ok = {"csv": head.startswith(b"query_id,corpus_id,rank,score"),
          "jsonl": head.startswith(b'{"query_id"'),
          "db": head.startswith(b"SQLite format 3")}[kind]
    return None if ok else f"{path} already exists and is not an earlier {kind} output"


def refuse(path: Path, kind: str) -> bool:
    why = output_refused(path, kind)
    if why:
        print(f"refusing to write {path}: {why}. Choose another output path.")
    return bool(why)


def cmd_index_corpus(args: argparse.Namespace) -> int:
    from retrieval import workflows
    from retrieval.batch import InputError
    from retrieval.embed import set_threads

    db = Path(args.out) if args.out else Path("out") / f"{Path(args.file).stem}.db"
    if refuse(db, "db"):
        return 1
    print(f"threads   {set_threads(args.threads)}")
    ok, message = check_environment()
    print(f"encoder   {message}")
    if not ok:
        print("cannot index without the encoder. Run: debug907 doctor")
        return 1
    try:
        report = workflows.index_corpus(Path(args.file), db, show_progress=True)
    except InputError as exc:
        print(f"cannot read the corpus: {exc}")
        return 1
    print(f"\nindexed   {db}  ({report.snippets:,} snippets)")
    _print_build(report)
    print(f"\nrank a file of queries: debug907 rank --queries <file> --db {db} --out ranked.csv")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    from retrieval import batch, workflows
    from retrieval.embed import set_threads

    if refuse(Path(args.out), "csv") or (args.jsonl and refuse(Path(args.jsonl), "jsonl")):
        return 1
    print(f"threads   {set_threads(args.threads)}")
    if not workflows.index_is_usable(args.db):
        print(f"no usable index at {args.db}. Build one: debug907 index-corpus <file> "
              "(or index-repo, or index)")
        return 1
    try:
        queries = batch.read_queries(Path(args.queries))
    except batch.InputError as exc:
        print(f"cannot read the queries: {exc}")
        return 1
    index = load_index(args.db)
    try:
        started = time.perf_counter()
        run = batch.rank(index, queries, Embedder(), top_k=args.top,
                         hubness=False if args.no_hubness else None, show_progress=True)
        seconds = time.perf_counter() - started
        rows = batch.write_rankings_csv(run, Path(args.out))
        if args.jsonl:
            batch.write_rankings_jsonl(run, index, Path(args.jsonl))
    finally:
        index.close()
    print(f"ranked    {len(queries):,} queries, top {args.top}, in {seconds:.1f}s")
    print(f"wrote     {args.out}  ({rows:,} rows: query_id,corpus_id,rank,score)")
    if args.jsonl:
        print(f"wrote     {args.jsonl}  (with path, lines and name per result)")
    print(f"score it: debug907 eval --qrels <qrels file> --rankings {args.out}")
    return 0


def cmd_eval_qrels(args: argparse.Namespace) -> int:
    from retrieval import batch
    from tools.score_csv import read_rankings

    try:
        qrels = batch.read_qrels(Path(args.qrels))
        run = read_rankings(Path(args.rankings))
    except (batch.InputError, ValueError, OSError) as exc:
        print(f"cannot read the input: {exc}")
        return 1
    got = batch.evaluate(run, qrels)
    unranked = sum(1 for q in qrels if q not in run)
    print(f"rankings  {args.rankings}  ({len(run):,} ranked queries)")
    print(f"qrels     {args.qrels}  ({len(qrels):,} judged queries"
          + (f"; {unranked} of them not in the rankings, scored 0" if unranked else "") + ")")
    print(f"NDCG@10   {got['ndcg_at_10']:.5f}\nMRR@10    {got['mrr_at_10']:.5f}\n"
          f"R@100     {got['recall_at_100']:.5f}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    from retrieval import sync
    from retrieval.embed import set_threads

    repo = Path(args.path)
    if not repo.is_dir():
        print(f"not a directory: {repo}")
        return 1
    print(f"threads   {set_threads(args.threads)}")
    ok, message = check_environment()
    print(f"encoder   {message}")
    if not ok:
        print("cannot index without the encoder. Run: debug907 doctor")
        return 1
    db = Path(args.out) if args.out else Path("out") / f"{repo.resolve().name}_commits.db"
    if refuse(db, "db"):
        return 1
    kwargs = dict(include_vendored=args.include_vendored, all_languages=args.all_languages,
                  show_progress=True)
    print(f"index     {db}")
    try:
        if args.watch:
            print(f"watching  {repo} every {args.interval:g}s (Ctrl+C stops; nothing is "
                  "written to the repository)")
            return sync.watch(repo, db, interval=args.interval, **kwargs)
        print(sync.describe(sync.sync(repo, db, rev=args.rev, **kwargs)))
    except sync.SyncError as exc:
        print(f"sync failed: {exc}")
        return 1
    print(f"\nsearch it: debug907 interactive --db {db} --all-versions")
    return 0


def cmd_eval_repo(args: argparse.Namespace) -> int:
    from bench.repo_eval import evaluate_repo
    from retrieval.embed import set_threads

    root = Path(args.path)
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1
    print(f"threads   {set_threads(args.threads)}")
    db = Path(args.out) if args.out else Path("out") / f"{root.resolve().name}_eval.db"
    report = evaluate_repo(root, db, args.queries, Embedder(), show_progress=True)
    print(f"functions {report['functions']:,}   queries: {report['pseudo_stats']['kept']} from "
          f"comments, {report['handwritten']} hand-written")
    for name, row in report["table"]["all"].items():
        delta = (f"  {row['diff']:+.4f}{' *' if row['significant'] else ''}"
                 if "diff" in row else "")
        print(f"  {name:<30} NDCG@10 {row['ndcg_at_10']:.4f}  MRR {row['mrr']:.4f}"
              f"  R@100 {row['recall_at_100']:.4f}{delta}")
    print("wrote bench/results/repo_eval.md")
    return 0


def _version_flags(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group()
    group.add_argument("--version", default=None,
                       help="search one version of a multi-version index (default: latest)")
    group.add_argument("--all-versions", action="store_true",
                       help="search every version; results show which versions contain them")


def _versions_arg(args: argparse.Namespace) -> str | None:
    if getattr(args, "all_versions", False):
        return "all"
    return getattr(args, "version", None)


def build_parser() -> argparse.ArgumentParser:
    from banner import render

    class BannerParser(argparse.ArgumentParser):
        def format_help(self) -> str:
            return render() + "\n" + super().format_help()

    parser = BannerParser(
        prog="debug907",
        epilog=QUICK_START,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    class ExplicitDb(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            namespace.db_explicit = True

    parser.add_argument("--db", type=Path, default=DEFAULT_DB, action=ExplicitDb)
    parser.set_defaults(db_explicit=False)
    parser.add_argument("--real", action="store_true",
                        help="no-op: real CoIR AppsRetrieval is always used")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="show full tracebacks instead of one-line errors")

    class VersionAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            print("\n".join(version_info()))
            parser.exit(0)

    parser.add_argument("--version", action=VersionAction,
                        help="version, commit, model revision and configuration")
    sub = parser.add_subparsers(dest="command", required=False)

    db_after = argparse.ArgumentParser(add_help=False)
    db_after.add_argument("--db", type=Path, default=argparse.SUPPRESS, action=ExplicitDb,
                          help="index database to use")

    p_repo = sub.add_parser("index-repo",
                            help="index a JS/Python source tree for searching")
    p_repo.add_argument("path")
    p_repo.add_argument("--out", default="out/repo.db")
    p_repo.add_argument("--version", default="v1")
    p_repo.add_argument("--all-languages", action="store_true",
                        help="also index languages without a function extractor "
                             "(Java, Go, C/C++, C#, Rust, ...) as 60-line windows")
    p_repo.add_argument("--include-vendored", action="store_true",
                        help="also index node_modules/vendor (slow, noisy)")
    p_repo.add_argument("--threads", type=int, default=8,
                        help="CPU threads for encoding (default 8 keeps the machine usable)")

    p_sync = sub.add_parser("sync",
                            help="index a git commit (default HEAD) as a new version "
                                 "labelled with its short hash; read-only on the repo")
    p_sync.add_argument("path", help="a git repository")
    p_sync.add_argument("--out", default=None,
                        help="index database (default out/<repo name>_commits.db)")
    p_sync.add_argument("--rev", default="HEAD", help="commit, tag or branch to sync")
    p_sync.add_argument("--watch", action="store_true",
                        help="keep polling HEAD and sync every new commit (no hooks installed)")
    p_sync.add_argument("--interval", type=float, default=30.0,
                        help="--watch polling interval in seconds (default 30)")
    p_sync.add_argument("--all-languages", action="store_true",
                        help="also index languages without a function extractor")
    p_sync.add_argument("--include-vendored", action="store_true",
                        help="also index node_modules/vendor")
    p_sync.add_argument("--threads", type=int, default=8)

    sub.add_parser("describe", help="inspect the dataset before coding against it")

    p_index = sub.add_parser("index", help="build the AppsRetrieval index (shipped config: raw documents)",
                             parents=[db_after])
    p_index.add_argument("--version", default=None, help=argparse.SUPPRESS)
    p_index.add_argument("--reset", action="store_true",
                         help="rebuild the text tables of every version (vectors are kept)")
    p_index.add_argument("--drop-vectors", action="store_true",
                         help="also discard this index's stored vectors (explicit; slow rebuild)")
    p_index.add_argument("--threads", type=int, default=8,
                         help="CPU threads for encoding (default 8)")

    p_rep = sub.add_parser("reproduce",
                           help="score the shipped config on real data; exits 1 if it drifts",
                           parents=[db_after])
    p_rep.add_argument("--threads", type=int, default=8,
                       help="CPU threads for encoding (default 8)")
    p_rep.add_argument("--full", action="store_true",
                       help="all 3,765 test queries, checked against appsretrieval_results.json")

    p_search = sub.add_parser("search", help="run a query", parents=[db_after])
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--lines", type=int, default=3)
    p_search.add_argument("--track", choices=["a", "b"], default="a",
                          help="a = dense only (shipped); b = +BM25+rerank, measured worse")
    p_search.add_argument("--explain", action="store_true")
    p_search.add_argument("--no-rerank", action="store_true")
    p_search.add_argument("--no-hubness", action="store_true",
                          help="plain cosine (the hubness correction is on for AppsRetrieval)")
    p_search.add_argument("--include-tests", action="store_true",
                          help="include test files (repo indexes hide them by default)")
    p_search.add_argument("--collapse", action="store_true",
                          help="one row per snippet across versions (automatic with "
                               "--all-versions on the shipped track)")
    _version_flags(p_search)

    p_eval = sub.add_parser("eval", help="NDCG@10 / MRR of the shipped config, real split",
                            parents=[db_after])
    p_eval.add_argument("--sample", action="store_true",
                        help="the validated 1,000-query stratified sample, not all queries")
    p_eval.add_argument("--threads", type=int, default=8)
    p_eval.add_argument("--limit", type=int, default=argparse.SUPPRESS,
                        help="score only the first N queries")
    p_eval.add_argument("--qrels", type=Path, default=None,
                        help="your own relevance labels (CSV/TSV/JSONL, or TREC qrels); "
                             "scores --rankings instead of the AppsRetrieval split")
    p_eval.add_argument("--rankings", type=Path, default=None,
                        help="a rankings CSV from `debug907 rank` (query_id,corpus_id,rank,score)")

    p_ic = sub.add_parser("index-corpus",
                          help="index any corpus of snippets from a JSONL/JSON/CSV file "
                               "(id, text[, path])")
    p_ic.add_argument("file")
    p_ic.add_argument("--out", default=None, help="index database (default out/<file name>.db)")
    p_ic.add_argument("--threads", type=int, default=8)

    p_rank = sub.add_parser("rank", help="rank a file of queries (id, text) and write a CSV "
                                         "query_id,corpus_id,rank,score", parents=[db_after])
    p_rank.add_argument("--queries", required=True, help="JSONL/JSON/CSV file of queries")
    p_rank.add_argument("--top", type=int, default=10, help="results per query (default 10)")
    p_rank.add_argument("--out", required=True, help="the rankings CSV to write")
    p_rank.add_argument("--jsonl", default=None,
                        help="also write JSONL with each result's path, lines and name")
    p_rank.add_argument("--no-hubness", action="store_true",
                        help="plain cosine (the hubness correction is on for AppsRetrieval)")
    p_rank.add_argument("--threads", type=int, default=8)

    p_abl = sub.add_parser("ablate", help="the real ablation table (stratified sample)")
    p_abl.add_argument("--slow", action="store_true",
                       help="include rows that need a fresh encoder pass")

    p_re = sub.add_parser("reindex", help="P1: full vs incremental index timing")
    p_re.add_argument("--change-pct", type=float, default=2.0)

    p_tool = sub.add_parser("interactive",
                            help="the terminal tool (same as running debug907 with no arguments)",
                            parents=[db_after])
    _version_flags(p_tool)

    p_er = sub.add_parser("eval-repo",
                          help="held-out benchmark on a real source tree (comment + "
                               "hand-written queries)")
    p_er.add_argument("path")
    p_er.add_argument("--queries", type=Path, default=None,
                      help="hand-written queries JSON (e.g. data/npm_queries.json)")
    p_er.add_argument("--out", default=None, help="index database (default out/<name>_eval.db)")
    p_er.add_argument("--threads", type=int, default=8)

    p_ex = sub.add_parser("export-index",
                          help="write the optional pre-built AppsRetrieval index file (.db.gz)",
                          parents=[db_after])
    p_ex.add_argument("file", help="output file, e.g. out/appsretrieval_index.db.gz")
    p_ex.add_argument("--pin", action="store_true",
                      help="also write its hashes to data/prebuilt_index.json")
    p_im = sub.add_parser("import-index",
                          help="OPTIONAL: install the pre-built index, verified against the "
                               "SHA-256 pinned in this repository", parents=[db_after])
    p_im.add_argument("source", help="the downloaded .db.gz file, or its https:// URL")
    p_im.add_argument("--force", action="store_true", help="replace an existing index at --db")
    p_vi = sub.add_parser("verify-index",
                          help="re-encode a random sample from scratch and compare with the "
                               "stored vectors", parents=[db_after])
    p_vi.add_argument("--sample", type=int, default=200, help="snippets to check (0 = all)")
    p_vi.add_argument("--seed", type=int, default=907)
    p_vi.add_argument("--threads", type=int, default=8)

    sub.add_parser("selftest", help="end-to-end smoke test")
    sub.add_parser("doctor", help="diagnose the environment and encoder")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return cmd_interactive(args)
    handlers = {
        "describe": cmd_describe, "index": cmd_index, "search": cmd_search,
        "eval": cmd_eval, "ablate": cmd_ablate, "reindex": cmd_reindex,
        "selftest": cmd_selftest, "doctor": cmd_doctor,
        "interactive": cmd_interactive, "index-repo": cmd_index_repo,
        "reproduce": cmd_reproduce, "eval-repo": cmd_eval_repo, "sync": cmd_sync,
        "index-corpus": cmd_index_corpus, "rank": cmd_rank,
        "export-index": cmd_export_index, "import-index": cmd_import_index,
        "verify-index": cmd_verify_index,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
