from __future__ import annotations

import csv
import json
import math

import pytest

import run_mteb
from tools import rankings_csv, score_csv


def test_mteb_run_never_overwrites_the_shipped_artifact_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_mteb, "run_mteb", lambda args: seen.setdefault("a", args) and 0)
    run_mteb.main([])
    assert seen["a"].out != "appsretrieval_results.json"
    assert seen["a"].out.startswith("out/")


def test_predictions_json_to_csv_top_k_ranked(tmp_path):
    preds = {"mteb_model_meta": {"model_name": "m", "revision": "r"},
             "default": {"test": {"q1": {"d1": 0.2, "d2": 0.9, "d3": 0.5},
                                  "q2": {"d9": 0.1}}}}
    src = tmp_path / "AppsRetrieval_predictions.json"
    src.write_text(json.dumps(preds), encoding="utf-8")
    out = tmp_path / "r.csv"
    n = rankings_csv.convert(src, out, top_k=2)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert n == 3
    assert list(rows[0]) == ["query_id", "corpus_id", "rank", "score"]
    assert [(r["query_id"], r["corpus_id"], r["rank"]) for r in rows] == [
        ("q1", "d2", "1"), ("q1", "d3", "2"), ("q2", "d9", "1")]


def test_exact_ties_follow_trec_eval_order(tmp_path):
    preds = {"default": {"test": {"q1": {"d1": 0.5, "d7": 0.5, "d3": 0.9}}}}
    src = tmp_path / "p.json"
    src.write_text(json.dumps(preds), encoding="utf-8")
    out = tmp_path / "r.csv"
    rankings_csv.convert(src, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["corpus_id"] for r in rows] == ["d3", "d7", "d1"]


def test_score_csv_ndcg_and_mrr(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("query_id,corpus_id,rank,score\n"
                    "q1,a,1,0.9\nq1,b,2,0.5\n"
                    "q2,c,1,0.8\nq2,d,2,0.7\n", encoding="utf-8")
    run = score_csv.read_rankings(path)
    qrels = {"q1": {"a": 1}, "q2": {"d": 1}, "q3": {"z": 1}}
    s = score_csv.score(run, qrels)
    assert s["ndcg_at_10"] == pytest.approx((1 + 1 / math.log2(3) + 0) / 3, abs=1e-6)
    assert s["mrr_at_10"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-6)
    assert s["queries"] == 3


def test_score_csv_rejects_malformed_rows(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("query_id,corpus_id,rank,score\nq1,a,one,0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        score_csv.read_rankings(path)


def test_the_artifact_bytes_match_the_recorded_hash_on_every_platform():
    import hashlib
    from pathlib import Path
    meta = json.loads(Path("appsretrieval_results.meta.json").read_text(encoding="utf-8"))
    raw = Path("appsretrieval_results.json").read_bytes()
    assert b"\r" not in raw
    assert hashlib.sha256(raw).hexdigest() == meta["artifact_sha256"]
    csv_path = Path("appsretrieval_rankings.csv")
    if csv_path.exists():
        data = csv_path.read_bytes()
        assert b"\r" not in data
        assert hashlib.sha256(data).hexdigest() == meta["rankings_csv_sha256"]
    attrs = Path(".gitattributes").read_text(encoding="utf-8") if Path(".gitattributes").exists() else ""
    assert not attrs or "appsretrieval_results*.json -text" in attrs
