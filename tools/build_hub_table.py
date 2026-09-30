from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-text", choices=("raw", "code"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--from-table", type=Path, default=Path("data/apps_hubness.json"),
                        help="the table whose k and beta are reused")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    from bench.dataset import load_appsretrieval
    from encoder import PrePostPipelineEncoder
    from retrieval import hubness
    from retrieval.embed import Embedder, set_threads
    from tools.hubness_eval import load_train

    set_threads(args.threads)
    base = json.loads(args.from_table.read_text(encoding="utf-8"))
    k, beta = base["k"], base["beta"]
    corpus, _, _ = load_appsretrieval()
    ids = sorted(corpus)
    enc = PrePostPipelineEncoder(raw_documents=args.document_text == "raw")
    texts = [enc.process_document(corpus[d]) for d in ids]
    emb = Embedder()
    D = np.asarray(emb.encode(texts, show_progress=True), dtype="float32")
    row = {d: i for i, d in enumerate(ids)}
    train_texts, train_gold = load_train()
    train_ids = sorted(train_texts)
    T = np.asarray(emb.encode([train_texts[i] for i in train_ids], show_progress=True),
                   dtype="float32")
    exclude: dict[int, set[int]] = {}
    for j, qid in enumerate(train_ids):
        if train_gold.get(qid) in row:
            exclude.setdefault(row[train_gold[qid]], set()).add(j)
    hub = hubness.hub_scores(D, T, k, exclude=exclude)
    values: dict[str, list[float]] = {}
    key_of = {}
    for d, text, h in zip(ids, texts, hub):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        values.setdefault(key, []).append(float(h))
        key_of[d] = key
    hubs = {key: sum(v) / len(v) for key, v in sorted(values.items())}
    table = {"source": "train-excl", "k": k, "beta": beta, "document_text": args.document_text,
             "made_by": "tools/build_hub_table.py: AppsRetrieval TRAINING queries only, own "
                        "answers excluded; k and beta from the previous table, not re-tuned",
             "hubs": hubs, "by_id": {d: hubs[key_of[d]] for d in ids}}
    args.out.write_text(json.dumps(table, separators=(",", ":")), encoding="utf-8")
    conflicts = sum(1 for v in values.values() if max(v) - min(v) > 1e-6)
    print(f"wrote {args.out}: {len(ids):,} documents, {len(hubs):,} distinct texts, "
          f"{conflicts} duplicate-text conflicts (mean used), k={k}, beta={beta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
