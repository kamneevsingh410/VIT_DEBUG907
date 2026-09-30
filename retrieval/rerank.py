from __future__ import annotations

from dataclasses import dataclass

from pipeline.query_proc import ProcessedQuery
from pipeline.tokens import split_identifier
from retrieval.index import Index, doc_tokens

CATEGORY_AFFINITY: dict[str, set[str]] = {
    "algorithmic": {"algorithm", "data_structure"},
    "identifier_led": {"utility", "class_method", "algorithm", "io", "data_structure"},
    "structural": {"class_method", "utility"},
    "behavioural": {"utility", "io", "algorithm"},
}


@dataclass(frozen=True)
class RerankConfig:

    enabled: bool = True
    w_coverage: float = 0.50
    w_idf_coverage: float = 1.00
    w_phrase: float = 0.35
    w_proximity: float = 0.25
    w_head: float = 0.30
    structural: bool = True
    boost_category: float = 0.15
    boost_identifier: float = 0.10
    length_penalty_cap: float = 0.10
    length_penalty_scale: float = 2000.0
    w_base: float = 1.0

    @classmethod
    def off(cls) -> "RerankConfig":
        return cls(enabled=False, structural=False)


def _proximity(doc_tokens_list: list[str], query_terms: set[str]) -> float:
    positions = [(i, t) for i, t in enumerate(doc_tokens_list) if t in query_terms]
    if len(positions) < 2:
        return 1.0 if positions else 0.0
    distinct = {t for _, t in positions}
    if len(distinct) < 2:
        return 0.5
    best = None
    seen: dict[str, int] = {}
    left = 0
    for right, (pos, term) in enumerate(positions):
        seen[term] = seen.get(term, 0) + 1
        while len(seen) == len(distinct):
            span = pos - positions[left][0] + 1
            best = span if best is None else min(best, span)
            lterm = positions[left][1]
            seen[lterm] -= 1
            if not seen[lterm]:
                del seen[lterm]
            left += 1
    if not best:
        return 0.0
    ideal = len(distinct)
    return ideal / max(ideal, best)


def _phrase_hits(doc_tokens_list: list[str], query_tokens: list[str]) -> int:
    if len(query_tokens) < 2:
        return 0
    wanted = {(a, b) for a, b in zip(query_tokens, query_tokens[1:])}
    if not wanted:
        return 0
    return sum(1 for pair in zip(doc_tokens_list, doc_tokens_list[1:]) if pair in wanted)


def interaction_score(index: Index, query: ProcessedQuery, snippet_id: str,
                      config: RerankConfig, enriched: str | None = None) -> float:
    if enriched is None:
        enriched = index.enriched(snippet_id)
    tokens = doc_tokens(enriched)
    if not tokens:
        return 0.0
    token_set = set(tokens)

    q_terms = [t for t in dict.fromkeys(query.tokens) if len(t) > 1]
    if not q_terms:
        return 0.0
    q_set = set(q_terms)

    matched = q_set & token_set
    coverage = len(matched) / len(q_set)

    idf_total = sum(index.idf(t) for t in q_set) or 1.0
    idf_matched = sum(index.idf(t) for t in matched)
    idf_coverage = idf_matched / idf_total

    phrase = _phrase_hits(tokens, q_terms)
    phrase_score = min(1.0, phrase / 3.0)

    proximity = _proximity(tokens, matched)

    head = set(doc_tokens("\n".join(enriched.splitlines()[:6])))
    head_score = len(q_set & head) / len(q_set)

    return (config.w_coverage * coverage
            + config.w_idf_coverage * idf_coverage
            + config.w_phrase * phrase_score
            + config.w_proximity * proximity
            + config.w_head * head_score)


def structural_score(base: float, index: Index, query: ProcessedQuery,
                     snippet_id: str, config: RerankConfig) -> float:
    if not config.structural:
        return base
    boost = 0.0

    if getattr(index, "corpus_kind", "") != "repo":
        expected = CATEGORY_AFFINITY.get(query.category, set())
        if index.categories.get(snippet_id) in expected:
            boost += config.boost_category

    idents = index.identifiers.get(snippet_id, set())
    if idents:
        query_idents = {i.lower() for i in query.identifiers}
        for ident in query_idents:
            if ident in idents:
                boost += config.boost_identifier
                break
        else:
            parts = set()
            for ident in idents:
                parts.update(split_identifier(ident))
            if parts and len(parts & set(query.tokens)) >= 2:
                boost += config.boost_identifier * 0.5

    length = index.lengths.get(snippet_id, 0)
    penalty = min(length / config.length_penalty_scale, 1.0) * config.length_penalty_cap

    return base + boost - penalty


def rerank(index: Index, query: ProcessedQuery, shortlist: list[tuple[str, float]],
           config: RerankConfig | None = None, top_k: int = 10) -> list[tuple[str, float]]:
    cfg = config or RerankConfig()
    if not cfg.enabled:
        return shortlist[:top_k]

    scored: list[tuple[str, float]] = []
    texts = index.enriched_many([sid for sid, _ in shortlist])
    for rank, (snippet_id, _fused) in enumerate(shortlist):
        base = cfg.w_base / (1.0 + rank / 10.0)
        pair = interaction_score(index, query, snippet_id, cfg, enriched=texts[snippet_id])
        final = structural_score(base + pair, index, query, snippet_id, cfg)
        scored.append((snippet_id, final))

    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored[:top_k]
