from __future__ import annotations

from pipeline.query_proc import process
from retrieval.index import build, load
from retrieval.rerank import RerankConfig, interaction_score, rerank
from tests.test_vector_keys import TextEmbedder

CORPUS = {f"f{i}.py::fn{i}#1": f"def fn{i}(path, n):\n    '''load config file {i}'''\n"
                               f"    return read_config(path)[{i}] * n\n" for i in range(30)}


def test_rerank_reads_all_candidates_in_one_query(tmp_path):
    db = tmp_path / "r.db"
    build(CORPUS, db, version="v1", views=("code",), embedder=TextEmbedder())
    index = load(db)
    try:
        query = process("load the config file from a path")
        shortlist = [(sid, 1.0 - i / 100) for i, sid in enumerate(sorted(CORPUS))]
        cfg = RerankConfig()
        expected = sorted(((sid, 0.0) for sid in CORPUS), key=lambda kv: kv[0])
        reads: list[str] = []
        index.conn.set_trace_callback(lambda sql: reads.append(sql)
                                      if "enriched" in sql and "SELECT" in sql.upper() else None)
        ranked = rerank(index, query, shortlist, cfg, top_k=30)
        index.conn.set_trace_callback(None)
        assert len(reads) == 1, f"{len(reads)} reads of candidate text for one rerank"
        for sid, score in ranked:
            rank = next(i for i, (s, _) in enumerate(shortlist) if s == sid)
            base = cfg.w_base / (1.0 + rank / 10.0)
            from retrieval.rerank import structural_score
            alone = structural_score(base + interaction_score(index, query, sid, cfg),
                                     index, query, sid, cfg)
            assert abs(score - alone) < 1e-12
        assert len(expected) == len(ranked)
    finally:
        index.close()
