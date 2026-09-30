from __future__ import annotations

import numpy as np

from tools.adaptive_eval import margins, rrf, segments

STATEMENT = ("Given n integers, print the largest sum of any contiguous run of them, "
             "or 0 when all are negative.\n\n-----Input-----\nThe first line holds n.\n\n"
             "-----Output-----\nOne integer.")


def test_segments_split_description_from_io_sections():
    desc, io = segments(STATEMENT)
    assert desc.startswith("Given n integers") and "-----Input-----" not in desc
    assert io.startswith("-----Input-----") and "-----Output-----" in io


def test_no_sections_means_no_second_pass():
    assert segments("reverse a linked list in place") is None
    assert segments("-----Input----- only a marker at the start and nothing else here") is None


def test_margin_is_top1_minus_top2():
    assert np.allclose(margins(np.array([[0.1, 0.5, 0.45], [0.9, 0.2, 0.1]])), [0.05, 0.7])


def test_rrf_rewards_agreement():
    a = np.array([[0.9, 0.8, 0.1]])
    b = np.array([[0.1, 0.9, 0.2]])
    fused = rrf(a, b)
    assert fused[0].argmax() == 1
