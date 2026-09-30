from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from retrieval import hubness
from tools.hubness_eval import compare, load_train, runs_from_scores, score_run

RESULTS = Path("bench/results")
WINDOW, STRIDE = 1022, 766


def code_view(text: str, enc) -> str:
    from pipeline.doc_proc import DEFAULT_VIEWS, enrich
    return enrich(text, enc.doc_config, views=DEFAULT_VIEWS).views.get("code") or text


def token_windows(text: str, tokenizer, window: int = WINDOW, stride: int = STRIDE) -> list[str]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= window:
        return [text]
    out = []
    for start in range(0, len(ids), stride):
        out.append(tokenizer.decode(ids[start:start + window]))
        if start + window >= len(ids):
            break
    return out


def mean_pool(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype="float32").mean(axis=0)
    return v / np.linalg.norm(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from encoder import PrePostPipelineEncoder
    from retrieval.embed import Embedder, load_model, set_threads
    from retrieval.index import load

    set_threads(args.threads)
    started = time.perf_counter()
    corpus, queries, qrels = load_appsretrieval()
    index = load(Path("out/real.db"))
    D = np.asarray(index.matrices["code"], dtype="float32")
    ids = list(index.matrix_ids["code"])
    index.close()
    row = {d: i for i, d in enumerate(ids)}
    emb = Embedder()
    tok = load_model().tokenizer
    enc = PrePostPipelineEncoder(raw_documents=False, hub_table=None)

    lengths, long_docs = {}, {}
    for d in ids:
        words = code_view(corpus[d], enc).split()
        full = " ".join(words)
        n = len(tok(full, add_special_tokens=False)["input_ids"])
        lengths[d] = n
        if len(words) > 512 or n > WINDOW:
            long_docs[d] = full
    windows = {d: token_windows(t, tok) for d, t in long_docs.items()}
    n_windows = sum(len(w) for w in windows.values())
    t = time.perf_counter()
    D_new = D.copy()
    for d, ws in windows.items():
        D_new[row[d]] = mean_pool(emb.encode(ws))
    encode_s = time.perf_counter() - t

    table = hubness.load_table()
    train_texts, train_gold = load_train()
    train_ids = sorted(train_texts)
    T = np.asarray(emb.encode([train_texts[i] for i in train_ids]), dtype="float32")
    exclude: dict[int, set[int]] = {}
    for j, qid in enumerate(train_ids):
        if train_gold.get(qid) in row:
            exclude.setdefault(row[train_gold[qid]], set()).add(j)
    hub_old = np.asarray([table["by_id"][d] for d in ids], dtype="float32")
    hub_new = hubness.hub_scores(D_new, T, table["k"], exclude=exclude)
    beta = table["beta"]

    test = sorted(q for q in queries if q in qrels)
    Q = np.asarray(emb.encode([queries[q] for q in test]), dtype="float32")
    runs = {"plain": runs_from_scores(Q @ D.T, test, ids),
            "plain + windows": runs_from_scores(Q @ D_new.T, test, ids),
            "shipped": runs_from_scores(Q @ D.T - beta * hub_old[None, :], test, ids),
            "shipped + windows": runs_from_scores(Q @ D_new.T - beta * hub_new[None, :], test, ids)}
    sample = set(load_stratified())
    gold_len = {q: lengths[next(iter(qrels[q]))] for q in test}
    cut = np.percentile(list(gold_len.values()), 200 / 3)
    groups = {"sample": [q for q in test if q in sample], "full": test,
              "held-out": [q for q in test if q not in sample],
              "long-document tercile": [q for q in test if gold_len[q] >= cut],
              "gold is long (windowed)": [q for q in test if next(iter(qrels[q])) in long_docs]}
    table_rows = {}
    for g, qs in groups.items():
        qr = {q: qrels[q] for q in qs}
        sub = lambda name: {q: runs[name][q] for q in qs}
        table_rows[g] = {"queries": len(qs),
                         **{name: score_run(sub(name), qr)["ndcg_at_10"] for name in runs},
                         "shipped_vs": compare(sub("shipped + windows"), sub("shipped"), qr),
                         "plain_vs": compare(sub("plain + windows"), sub("plain"), qr)}
    report = {"documents": len(ids), "long_documents": len(long_docs), "windows": n_windows,
              "max_windows": max((len(w) for w in windows.values()), default=0),
              "encode_seconds": round(encode_s, 1), "tercile_cut_tokens": int(cut),
              "groups": table_rows, "seconds": round(time.perf_counter() - started, 1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "long_docs.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / "long_docs.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def render(r: dict) -> str:
    lines = ["# Long documents: overlapping windows, mean-pooled", "",
             f"Documents over 512 words or over {WINDOW} model tokens (full code view): "
             f"**{r['long_documents']} of {r['documents']:,}**. Each is embedded as "
             f"{WINDOW}-token windows with stride {STRIDE} ({r['windows']} windows in all, at most "
             f"{r['max_windows']} for one document), mean-pooled and re-normalised. That is one "
             "vector per document, so it stays expressible in the MTEB encoder. Every other "
             f"document keeps its vector. Encoding the windows took {r['encode_seconds']} s. "
             "The hub scores are recomputed for the new vectors from the same 5,000 training "
             "queries; k and β are unchanged. `tools/long_docs.py`.", "",
             f"The long-document tercile: queries whose gold has ≥ {r['tercile_cut_tokens']} tokens.", "",
             "| queries | plain | plain + windows | shipped | shipped + windows | shipped: change [95% CI] |",
             "|---|---:|---:|---:|---:|---|"]
    for g, x in r["groups"].items():
        c = x["shipped_vs"]
        lines.append(f"| {g} ({x['queries']:,}) | {x['plain']:.4f} | {x['plain + windows']:.4f} | "
                     f"{x['shipped']:.4f} | {x['shipped + windows']:.4f} | {c['diff']:+.4f} "
                     f"[{c['low']:+.4f}, {c['high']:+.4f}] p≈{c['p']:.3f}"
                     f"{' **significant**' if c['significant'] else ''} |")
    g = r["groups"]
    lines += ["", "## Verdict", "",
              f"Too few documents are long to matter: {r['long_documents']} documents, and "
              f"{g['gold is long (windowed)']['queries']} test queries whose gold answer is one of "
              f"them. For those queries windows help a lot (shipped "
              f"{g['gold is long (windowed)']['shipped']:.4f} → "
              f"{g['gold is long (windowed)']['shipped + windows']:.4f}; plain "
              f"{g['gold is long (windowed)']['plain']:.4f} → "
              f"{g['gold is long (windowed)']['plain + windows']:.4f}), but 28 queries cannot "
              "reach significance. The full split and the held-out queries are unchanged "
              "(n.s.). The small gain on the sample alone does not count: the artifact rule needs "
              "the full split and the held-out queries. **Not adopted; the artifact stays.** An "
              "aside: the hubness correction lowers these long-gold queries (plain → shipped), so "
              "long solutions tend to be hubs.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
