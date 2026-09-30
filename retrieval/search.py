from __future__ import annotations

import time
from dataclasses import dataclass, field

from pipeline.query_proc import ProcessedQuery, QueryConfig, process
from retrieval import dense, sparse
from retrieval.embed import Embedder, EmbedderUnavailable
from retrieval.fuse import rrf
from retrieval.index import Index
from retrieval.rerank import RerankConfig, rerank


@dataclass(frozen=True)
class SearchConfig:
    query: QueryConfig = field(default_factory=QueryConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    use_dense: bool = True
    use_sparse: bool = True
    shortlist: int = 100
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    collapse_versions: bool = False
    prefer_newest: bool = False
    hubness: bool = False
    view_weights: dict[str, float] | None = None
    embed_raw_query: bool = True

    @classmethod
    def lexical_only(cls) -> "SearchConfig":
        return cls(use_dense=False, rerank=RerankConfig.off())

    @classmethod
    def dense_only(cls) -> "SearchConfig":
        return cls(use_sparse=False, rerank=RerankConfig.off())

    @classmethod
    def track_a(cls) -> "SearchConfig":
        return cls(use_sparse=False, rerank=RerankConfig.off())


@dataclass
class SearchResult:
    query: str
    processed: ProcessedQuery
    hits: list[tuple[str, float]]
    timings_ms: dict[str, float]
    stage_counts: dict[str, int]
    query_vector: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    versions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return [snippet_id for snippet_id, _ in self.hits]


def search(index: Index, query_text: str, config: SearchConfig | None = None,
           top_k: int = 10, embedder: Embedder | None = None) -> SearchResult:
    cfg = config or SearchConfig()
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}
    notes: list[str] = []
    query_vector: list[float] = []

    mark = time.perf_counter()
    processed = process(query_text, cfg.query)
    timings["preprocess"] = (time.perf_counter() - mark) * 1000

    lists: list[list[tuple[str, float]]] = []
    weights: list[float] = []

    if cfg.use_dense:
        mark = time.perf_counter()
        if not index.has_vectors:
            notes.append("index has no embeddings; semantic arm skipped")
        else:
            try:
                emb = embedder or Embedder()
                if cfg.embed_raw_query:
                    text = processed.embedding_text or query_text
                else:
                    text = processed.text
                query_vector = emb.encode_one(text)
                bias = None
                if cfg.hubness:
                    from retrieval import hubness
                    table = hubness.load_table()
                    if table is None:
                        raise FileNotFoundError(f"hubness is on but {hubness.TABLE} is missing")
                    view = hubness.table_view(index, table)
                    bias = {view: hubness.index_bias(index, table, view=view)}
                hits = dense.search_vector(index, query_vector, cfg.shortlist,
                                           cfg.view_weights, bias=bias)
                counts["dense"] = len(hits)
                lists.append(hits)
                weights.append(cfg.dense_weight)
            except EmbedderUnavailable as exc:
                notes.append(f"semantic arm unavailable: {exc}")
        timings["dense"] = (time.perf_counter() - mark) * 1000

    if cfg.use_sparse:
        mark = time.perf_counter()
        hits = sparse.search(index, processed.fts_terms(), cfg.shortlist)
        timings["sparse"] = (time.perf_counter() - mark) * 1000
        counts["sparse"] = len(hits)
        lists.append(hits)
        weights.append(cfg.sparse_weight)

    mark = time.perf_counter()
    if len(lists) > 1:
        fused = rrf(lists, k=cfg.rrf_k, weights=weights)
    elif lists:
        fused = lists[0]
    else:
        fused = []
    fused = fused[:cfg.shortlist]
    timings["fuse"] = (time.perf_counter() - mark) * 1000
    counts["shortlist"] = len(fused)

    mark = time.perf_counter()
    depth = top_k * max(1, len(index.loaded_versions)) if cfg.collapse_versions else top_k
    final = rerank(index, processed, fused, cfg.rerank, top_k=depth)
    timings["rerank"] = (time.perf_counter() - mark) * 1000

    versions: dict[str, list[str]] = {}
    if cfg.collapse_versions:
        from retrieval.versions import collapse_versions
        final, versions = collapse_versions(index, final, top_k=top_k,
                                            prefer_newest=cfg.prefer_newest)

    counts["returned"] = len(final)
    timings["total"] = sum(v for k, v in timings.items() if k != "total")

    return SearchResult(query=query_text, processed=processed, hits=final,
                        timings_ms=timings, stage_counts=counts,
                        query_vector=query_vector, notes=notes, versions=versions)
