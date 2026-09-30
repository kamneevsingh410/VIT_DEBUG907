from __future__ import annotations

import csv
import json
from pathlib import Path

ID_KEYS = ("id", "_id", "corpus_id", "corpus-id", "doc_id", "query_id", "query-id", "qid")
TEXT_KEYS = ("text", "code", "content", "body", "query", "question")
PATH_KEYS = ("path", "title", "file")


class InputError(ValueError):
    pass


def read_records(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise InputError(f"file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson") or (suffix != ".json" and text.lstrip().startswith("{")):
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise InputError(f"{path} line {n}: not JSON ({exc.msg})") from None
        return rows
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [{"id": k, "text": v} for k, v in data.items()]
        return list(data)
    dialect = "excel-tab" if suffix == ".tsv" or "\t" in text.splitlines()[0] else "excel"
    return list(csv.DictReader(text.splitlines(), dialect=dialect))


def _field(row: dict, keys: tuple[str, ...], what: str, where: str) -> str:
    for k in keys:
        if row.get(k) not in (None, ""):
            return str(row[k])
    raise InputError(f"{where}: no {what} field (expected one of {', '.join(keys)})")


def read_corpus(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    corpus, paths = {}, {}
    for n, row in enumerate(read_records(path), 1):
        where = f"{path} record {n}"
        sid = _field(row, ID_KEYS, "id", where)
        if sid in corpus:
            raise InputError(f"{where}: duplicate id {sid!r}")
        corpus[sid] = _field(row, TEXT_KEYS, "text", where)
        p = next((str(row[k]) for k in PATH_KEYS if row.get(k)), None)
        if p:
            paths[sid] = p
    if not corpus:
        raise InputError(f"{path}: no records")
    return corpus, paths


def read_queries(path: Path) -> dict[str, str]:
    queries = {}
    for n, row in enumerate(read_records(path), 1):
        where = f"{path} record {n}"
        qid = _field(row, ("query_id", "query-id", "qid", "id", "_id"), "id", where)
        queries[qid] = _field(row, ("text", "query", "question"), "text", where)
    if not queries:
        raise InputError(f"{path}: no queries")
    return queries


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    path = Path(path)
    first = path.read_text(encoding="utf-8-sig").splitlines()[:1]
    qrels: dict[str, dict[str, int]] = {}
    if first and len(first[0].split()) == 4 and not any(k in first[0] for k in ("query", "qid,")):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                q, _, d, r = line.split()
                qrels.setdefault(q, {})[d] = int(float(r))
        return qrels
    for n, row in enumerate(read_records(path), 1):
        where = f"{path} record {n}"
        q = _field(row, ("query_id", "query-id", "qid"), "query id", where)
        d = _field(row, ("corpus_id", "corpus-id", "doc_id", "docid"), "corpus id", where)
        r = next((row[k] for k in ("score", "relevance", "rel") if row.get(k) not in (None, "")), 1)
        qrels.setdefault(q, {})[d] = int(float(r))
    return qrels


def rank(index, queries: dict[str, str], embedder, top_k: int = 10,
         hubness: bool | None = None, show_progress: bool = False) -> dict[str, list[tuple[str, float]]]:
    from retrieval import workflows
    embedder.encode(list(queries.values()), show_progress=show_progress)
    return {qid: workflows.shipped_search(index, text, embedder, top_k=top_k,
                                          hubness=hubness).hits
            for qid, text in queries.items()}


def write_rankings_csv(run: dict[str, list[tuple[str, float]]], path: Path) -> int:
    rows = 0
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "corpus_id", "rank", "score"])
        for qid, hits in run.items():
            for r, (sid, score) in enumerate(hits, 1):
                writer.writerow([qid, sid, r, f"{score:.6f}"])
                rows += 1
    return rows


def write_rankings_jsonl(run: dict[str, list[tuple[str, float]]], index, path: Path) -> int:
    from retrieval import workflows
    rows = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for qid, hits in run.items():
            for r, (sid, score) in enumerate(hits, 1):
                info = workflows.describe_hit(index, sid)
                handle.write(json.dumps({"query_id": qid, "rank": r, "corpus_id": sid,
                                         "score": round(score, 6), "path": info.path,
                                         "start_line": info.start, "end_line": info.end,
                                         "name": info.name}) + "\n")
                rows += 1
    return rows


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict:
    from tools.score_csv import recall, score
    return {**score(run, qrels), "recall_at_100": recall(run, qrels, 100)}
