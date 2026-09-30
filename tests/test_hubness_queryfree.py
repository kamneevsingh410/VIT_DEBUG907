from __future__ import annotations

import numpy as np

from tools.hubness_queryfree import centroid_similarity, reverse_knn_counts, sources, zscore


def unit(rows):
    m = np.asarray(rows, dtype="float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


DOCS = unit([[1, 1, 0], [1, 0.2, 0], [0.2, 1, 0], [1, 0.9, 0.3]])


def test_reverse_knn_counts_how_often_a_document_is_a_neighbour():
    counts = reverse_knn_counts(DOCS, k=1)
    sims = DOCS @ DOCS.T
    np.fill_diagonal(sims, -np.inf)
    expected = np.bincount(sims.argmax(axis=1), minlength=4)
    assert np.array_equal(counts, expected) and counts.sum() == 4
    assert counts.argmax() == 0


def test_centroid_similarity_and_zscore():
    c = centroid_similarity(DOCS)
    assert c.argmax() in (0, 3)
    z = zscore(c)
    assert abs(float(z.mean())) < 1e-6 and abs(float(z.std()) - 1) < 1e-5
    assert np.all(zscore(np.ones(3)) == 0)


def test_sources_are_standardised_and_cover_every_kind():
    s = sources(DOCS, ks=(1, 2))
    assert set(s) == {"centroid", "doc-mean-k1", "doc-mean-k2", "rknn-k1", "rknn-k2"}
    for z in s.values():
        assert abs(float(z.mean())) < 1e-5
