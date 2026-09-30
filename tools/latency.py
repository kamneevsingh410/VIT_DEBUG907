from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("bench/results")
DB = Path("out/real.db")
CACHE = Path("out/embed_cache.db")
QUERY = "sort the array and print the median of every window of size k"


def peak_rss_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                          counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / 1e6
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def cold_start(runs: int = 3) -> list[dict]:
    out = []
    for _ in range(runs):
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, "cli.py", "search", QUERY, "--db", str(DB),
                               "--top-k", "3"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        wall = time.perf_counter() - started
        load = re.search(r"model load ([\d.]+)s", proc.stdout)
        query = re.search(r"latency\s+([\d.]+) ms total", proc.stdout)
        out.append({"wall_s": round(wall, 2), "exit": proc.returncode,
                    "model_load_s": float(load.group(1)) if load else None,
                    "query_ms": float(query.group(1)) if query else None})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle", action="store_true", help="the machine is otherwise idle")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from retrieval import workflows
    from retrieval.embed import Embedder, set_threads
    from retrieval.index import load

    threads = set_threads(args.threads)
    report = {"idle": args.idle, "threads": threads,
              "machine": f"{platform.system()} {platform.release()} / {platform.machine()} / "
                         f"{os.cpu_count()} logical CPUs / Python {platform.python_version()}"}
    report["cold_start"] = cold_start()

    corpus, queries, _qrels = load_appsretrieval()
    picked = set(load_stratified())
    sample = [t for q, t in queries.items() if q in picked]

    started = time.perf_counter()
    index = load(DB)
    report["index_load_s"] = round(time.perf_counter() - started, 2)
    fresh = Embedder(cache=False)
    started = time.perf_counter()
    fresh.warm()
    report["model_load_s"] = round(time.perf_counter() - started, 2)
    try:
        latencies = []
        for text in sample:
            t = time.perf_counter()
            workflows.shipped_search(index, text, fresh, top_k=10)
            latencies.append((time.perf_counter() - t) * 1000)
        cached = Embedder()
        cached.encode(sample)
        hits = []
        for text in sample:
            t = time.perf_counter()
            workflows.shipped_search(index, text, cached, top_k=10)
            hits.append((time.perf_counter() - t) * 1000)
    finally:
        index.close()
    report["warm_query_ms"] = {"n": len(latencies), "p50": pct(latencies, 50),
                               "p95": pct(latencies, 95), "p99": pct(latencies, 99),
                               "max": max(latencies)}
    report["cached_query_ms"] = {"p50": pct(hits, 50), "p95": pct(hits, 95)}

    docs = list(corpus.values())[:200]
    started = time.perf_counter()
    fresh.encode(docs)
    seconds = time.perf_counter() - started
    report["encode_throughput"] = {"docs": len(docs), "seconds": round(seconds, 2),
                                   "docs_per_s": round(len(docs) / seconds, 2),
                                   "chars_per_s": round(sum(map(len, docs)) / seconds)}
    report["peak_rss_mb"] = round(peak_rss_mb())
    report["on_disk_mb"] = {"index": round(DB.stat().st_size / 1e6, 1),
                            "text_cache": round(CACHE.stat().st_size / 1e6, 1) if CACHE.exists()
                            else None}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report)
    print(json.dumps(report, indent=1))
    return 0


def write_md(r: dict) -> None:
    status = ("measured with the machine otherwise idle" if r["idle"]
              else "**measured under load** (the machine was in use)")
    w, c, e = r["warm_query_ms"], r["cached_query_ms"], r["encode_throughput"]
    colds = [x for x in r["cold_start"] if x["exit"] == 0]
    lines = ["# Latency profile", "",
             f"{r['machine']}, CPU only, {r['threads']} threads; {status}. `tools/latency.py`.", "",
             "| measure | value |", "|---|---:|"]
    if colds:
        lines += [f"| cold start: fresh process to first answer (3 runs) | "
                  f"{', '.join(f'{x['wall_s']:.1f}' for x in colds)} s |",
                  f"| of which model load | "
                  f"{', '.join(f'{x['model_load_s']:.1f}' for x in colds if x['model_load_s'])} s |"]
    lines += [f"| index load (8,765 snippets) | {r['index_load_s']} s |",
              f"| warm query, encoder forward pass included: p50 / p95 / p99 ({w['n']} queries) | "
              f"{w['p50']:.0f} / {w['p95']:.0f} / {w['p99']:.0f} ms |",
              f"| warm query, query vector cached: p50 / p95 | {c['p50']:.1f} / {c['p95']:.1f} ms |",
              f"| encode throughput ({e['docs']} documents, seq 1024) | {e['docs_per_s']} docs/s, "
              f"{e['chars_per_s']:,} chars/s |",
              f"| peak RAM (this process) | {r['peak_rss_mb']:,} MB |",
              f"| index on disk | {r['on_disk_mb']['index']} MB |",
              f"| text cache on disk (all encoders, all experiments) | "
              f"{r['on_disk_mb']['text_cache']} MB |", "",
              "Search itself is one matrix-vector product over 8,765 × 768 floats; nearly all "
              "query time is the encoder's forward pass over the query text (AppsRetrieval "
              "queries average ~1,600 characters)."]
    (RESULTS / "latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
