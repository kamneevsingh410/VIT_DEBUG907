from __future__ import annotations

import csv
import json

import pytest

import cli
from retrieval import batch, workflows
from retrieval.index import load
from tests.test_vector_keys import TextEmbedder

SNIPPETS = [
    {"id": "s1", "text": "def load_config(path):\n    return yaml.safe_load(open(path))\n",
     "path": "cfg/loader.py"},
    {"id": "s2", "text": "def gcd(a, b):\n    return a if b == 0 else gcd(b, a % b)\n"},
    {"id": "s3", "text": "function fetchUrl(url) {\n  return http.get(url).body\n}\n",
     "path": "net/fetch.js"},
    {"id": "s4", "text": "def reverse_list(xs):\n    return xs[::-1]\n"},
]
QUERIES = [{"id": "q1", "text": "load the yaml config file"},
           {"id": "q2", "text": "greatest common divisor of two numbers"},
           {"id": "q3", "text": "download a web page over http"}]
QRELS = {"q1": {"s1": 1}, "q2": {"s2": 1}, "q3": {"s3": 1}}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def corpus_db(tmp_path):
    src = tmp_path / "snippets.jsonl"
    write_jsonl(src, SNIPPETS)
    db = tmp_path / "corpus.db"
    report = workflows.index_corpus(src, db, embedder=TextEmbedder())
    assert report.snippets == 4
    return db


def test_readers_accept_jsonl_csv_json_and_trec(tmp_path):
    c = tmp_path / "c.csv"
    c.write_text("id,text,path\na,def f(): pass,x.py\nb,def g(): pass,\n", encoding="utf-8")
    corpus, paths = batch.read_corpus(c)
    assert corpus == {"a": "def f(): pass", "b": "def g(): pass"} and paths == {"a": "x.py"}
    j = tmp_path / "q.json"
    j.write_text(json.dumps({"q1": "find a thing"}), encoding="utf-8")
    assert batch.read_queries(j) == {"q1": "find a thing"}
    t = tmp_path / "qrels.trec"
    t.write_text("q1 0 s1 1\nq1 0 s9 0\nq2 0 s2 2\n", encoding="utf-8")
    assert batch.read_qrels(t) == {"q1": {"s1": 1, "s9": 0}, "q2": {"s2": 2}}
    tsv = tmp_path / "qrels.tsv"
    tsv.write_text("query-id\tcorpus-id\tscore\nq1\ts1\t1\n", encoding="utf-8")
    assert batch.read_qrels(tsv) == {"q1": {"s1": 1}}


def test_bad_input_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}\n', encoding="utf-8")
    with pytest.raises(batch.InputError, match="duplicate id"):
        batch.read_corpus(bad)
    bad.write_text('{"name": "a"}\n', encoding="utf-8")
    with pytest.raises(batch.InputError, match="no id field"):
        batch.read_corpus(bad)
    with pytest.raises(batch.InputError, match="file not found"):
        batch.read_queries(tmp_path / "missing.jsonl")


def test_rank_then_eval_end_to_end_matches_the_shipped_path(corpus_db, tmp_path, monkeypatch, capsys):
    queries = tmp_path / "queries.jsonl"
    write_jsonl(queries, QUERIES)
    qrels = tmp_path / "qrels.csv"
    qrels.write_text("query_id,corpus_id,score\n" + "".join(
        f"{q},{d},1\n" for q, ds in QRELS.items() for d in ds), encoding="utf-8")
    out, jl = tmp_path / "ranked.csv", tmp_path / "ranked.jsonl"
    monkeypatch.setattr(cli, "Embedder", TextEmbedder)
    assert cli.main(["rank", "--queries", str(queries), "--db", str(corpus_db), "--top", "4",
                     "--out", str(out), "--jsonl", str(jl)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["rank"] for r in rows if r["query_id"] == "q1"] == ["1", "2", "3", "4"]
    lines = [json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines()]
    s1 = next(x for x in lines if x["corpus_id"] == "s1")
    assert s1["path"] == "cfg/loader.py" and s1["start_line"] == 1
    assert next(x for x in lines if x["corpus_id"] == "s2")["path"] == "corpus"

    capsys.readouterr()
    assert cli.main(["eval", "--qrels", str(qrels), "--rankings", str(out)]) == 0
    printed = capsys.readouterr().out
    index = load(corpus_db)
    try:
        shipped = workflows.evaluate_shipped(index, {q["id"]: q["text"] for q in QUERIES},
                                             QRELS, TextEmbedder())
        run = {q: [s for s, _ in hits] for q, hits in
               batch.rank(index, {q["id"]: q["text"] for q in QUERIES}, TextEmbedder(),
                          top_k=100).items()}
    finally:
        index.close()
    ours = batch.evaluate(run, QRELS)
    assert ours["ndcg_at_10"] == pytest.approx(shipped.scores.ndcg_at_10, abs=1e-9)
    assert f"NDCG@10   {batch.evaluate({r['query_id']: [x['corpus_id'] for x in rows if x['query_id'] == r['query_id']] for r in rows}, QRELS)['ndcg_at_10']:.5f}" in printed


def test_generic_corpus_is_not_hubness_corrected(corpus_db):
    index = load(corpus_db)
    try:
        assert index.corpus_kind == "corpus" and workflows.hubness_default(index) is False
    finally:
        index.close()


def test_eval_needs_both_files(capsys):
    assert cli.main(["eval", "--qrels", "x.tsv"]) == 2
    assert "needs both" in capsys.readouterr().out


def test_recall_at_100():
    from tools.score_csv import recall
    assert recall({"q": ["a", "b"]}, {"q": {"a": 1, "c": 1}}) == 0.5
