from __future__ import annotations

import numpy as np
import pytest

from retrieval import hubness


def unit(rows):
    m = np.asarray(rows, dtype="float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


DOCS = unit([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]])
REFS = unit([[1, 0.05, 0], [0.95, 0.1, 0], [1, 0, 0.02], [0, 0.2, 1]])


def test_hub_score_is_mean_cosine_to_the_k_nearest_references():
    hub = hubness.hub_scores(DOCS, REFS, k=2)
    sims = DOCS @ REFS.T
    expected = np.sort(sims, axis=1)[:, -2:].mean(axis=1)
    assert np.allclose(hub, expected, atol=1e-6)
    assert hub[0] > hub[2]


def test_own_answers_are_excluded_from_a_documents_hub_score():
    with_own = hubness.hub_scores(DOCS, REFS, k=1)
    without = hubness.hub_scores(DOCS, REFS, k=1, exclude={0: {0, 1, 2}})
    assert without[0] < with_own[0]
    assert np.allclose(without[1:], with_own[1:])


def test_doc_doc_hub_never_counts_a_document_as_its_own_neighbour():
    hub = hubness.doc_doc_hub(DOCS, k=1)
    sims = DOCS @ DOCS.T
    np.fill_diagonal(sims, -np.inf)
    assert np.allclose(hub, sims.max(axis=1), atol=1e-6)


def test_penalised_scores_subtract_beta_times_hub():
    q = unit([[1, 0, 0]])
    hub = np.array([0.5, 0.1, 0.0, 0.0], dtype="float32")
    scores = hubness.penalised(q, DOCS, hub, beta=0.4)
    assert np.allclose(scores, q @ DOCS.T - 0.4 * hub, atol=1e-6)


def test_extra_dimension_makes_a_plain_dot_product_equal_the_penalised_score():
    q = unit([[0.3, 0.9, 0.1]])
    hub = np.array([0.5, 0.1, 0.3, 0.2], dtype="float32")
    qx = hubness.extend_queries(q)
    dx = hubness.extend_documents(DOCS, hub, beta=0.25)
    assert np.allclose(qx @ dx.T, hubness.penalised(q, DOCS, hub, 0.25), atol=1e-6)


def test_leakage_auc_detects_a_score_that_separates_two_groups():
    separating = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
    groups = np.array([True, True, True, False, False, False])
    assert hubness.separation_auc(separating, groups) == pytest.approx(1.0)
    assert hubness.separation_auc(np.array([0.5] * 6), groups) == pytest.approx(0.5)


def test_hub_share_of_wrong_top1():
    wrong_top1 = ["a", "a", "a", "a", "a", "b", "c"]
    assert hubness.hub_share(wrong_top1, min_times=5) == pytest.approx(5 / 7)
    assert hubness.hub_share([], min_times=5) == 0.0


def test_mteb_encoder_hub_table_reproduces_the_penalised_score(monkeypatch):
    import hashlib
    from encoder import PrePostPipelineEncoder
    enc = PrePostPipelineEncoder()
    docs = ["def a():\n    return 1", "def b():\n    return 2"]
    prepared = [enc.process_document(d) for d in docs]
    vecs = {prepared[0]: DOCS[0], prepared[1]: DOCS[2], "q": unit([[1, 0.2, 0]])[0]}
    monkeypatch.setattr(enc.embedder, "encode", lambda texts, **_: [vecs.get(t, DOCS[3]) for t in texts])
    hubs = {hashlib.sha256(p.encode("utf-8")).hexdigest(): h for p, h in zip(prepared, (0.6, 0.1))}
    enc.hub_table = {"source": "train-excl", "k": 10, "beta": 0.3, "hubs": hubs}
    qx = enc.encode(["q"], prompt_type="query")
    dx = enc.encode(docs, prompt_type="document")
    assert qx.shape[1] == dx.shape[1] == 4
    expected = hubness.penalised(qx[:, :3], dx[:, :3], np.array([0.6, 0.1]), 0.3)
    assert np.allclose(enc.similarity(qx, dx), expected, atol=1e-6)
    with pytest.raises(KeyError, match="not in the hub table"):
        enc.encode(["def c(): pass"], prompt_type="document")


def test_mteb_encoder_without_a_hub_table_is_unchanged(monkeypatch):
    from encoder import PrePostPipelineEncoder
    enc = PrePostPipelineEncoder()
    monkeypatch.setattr(enc.embedder, "encode", lambda texts, **_: [DOCS[0] for _ in texts])
    assert enc.encode(["x"], prompt_type="query").shape == (1, 3)


def test_raw_documents_flag_passes_text_through_unchanged():
    from encoder import PrePostPipelineEncoder
    src = "def f(a):\n    return a  # keep me\n"
    assert PrePostPipelineEncoder(raw_documents=True).process_document(src) == src.strip()
    assert PrePostPipelineEncoder().process_document(src) == src.strip()
    assert PrePostPipelineEncoder(raw_documents=False).process_document(src) != src


def test_naive_baseline_memory_watchdog_reads_this_process():
    import os
    from tools.naive_baseline import working_set_gb
    if os.name == "nt":
        assert 0.001 < working_set_gb() < 64
    else:
        assert working_set_gb() == 0.0


def test_the_default_encoder_is_the_submitted_configuration(monkeypatch):
    from types import SimpleNamespace
    from encoder import PrePostPipelineEncoder
    enc = PrePostPipelineEncoder()
    assert enc.raw_documents is True
    apps, other = SimpleNamespace(name="AppsRetrieval"), SimpleNamespace(name="CosQA")
    table = enc.table_for(apps)
    assert table is not None and table["document_text"] == "raw"
    assert enc.table_for(other) is None and enc.table_for(None) is None
    monkeypatch.setattr(enc.embedder, "encode", lambda texts, **_: [DOCS[0] for _ in texts])
    assert enc.encode(["q"], prompt_type="query", task_metadata=apps).shape == (1, 4)
    assert enc.encode(["q"], prompt_type="query", task_metadata=other).shape == (1, 3)
    meta = enc.mteb_model_meta
    assert meta is None or "hub" in meta.revision
    assert PrePostPipelineEncoder(hub_table=None).table_for(apps) is None
