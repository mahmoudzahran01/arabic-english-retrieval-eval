"""Rank fusion (RRF), the language gate used by the hybrid systems, and the direction macro-average."""

import math

import numpy as np

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval import metrics as M


def _run(pids):
    """A stored first-stage run for one query: (ranked pids, scores)."""
    return {"q": (list(pids), [0.0] * len(pids))}


def test_rrf_scores_match_the_formula():
    # a: rank 1 in L1, rank 3 in L2 -> 1/61 + 1/63 ; b: rank 2 and rank 1 -> 1/62 + 1/61 ; c only in L2 at rank 2 -> 1/62
    fused = M.rrf([["a", "b"], ["b", "c", "a"]], k=60)
    assert fused == ["b", "a", "c"]
    assert 1 / 61 + 1 / 62 > 1 / 61 + 1 / 63 > 1 / 62


def test_rrf_respects_top_and_breaks_ties_by_id():
    fused = M.rrf([["x", "y", "z"], ["z", "y", "x"]], k=60, top=2)
    assert len(fused) == 2
    # x and z: 1/61 + 1/63 = 0.032266 each; y: 2/62 = 0.032258 -> the extremes win by convexity, then the id breaks the x/z tie
    assert fused == ["x", "z"]
    assert M.rrf([["b"], ["a"]]) == ["a", "b"]  # exact tie -> alphabetical order, deterministic


def test_gate_keeps_the_dense_run_for_cross_lingual_slices():
    bm25, dense = _run(["n1", "n2", "g"]), _run(["g", "n1", "n2"])
    assert E.fuse_slice(bm25, dense, cross_lingual=True, gate_language=True) is dense


def test_gate_still_fuses_same_language_slices():
    bm25, dense = _run(["g", "n1", "n2"]), _run(["n1", "g", "n2"])
    fused = E.fuse_slice(bm25, dense, cross_lingual=False, gate_language=True)
    assert fused["q"][0][:2] == ["g", "n1"]  # g: ranks 1 and 2 beats n1: ranks 2 and 1 only via the id tie-break
    assert set(fused["q"][0]) == {"g", "n1", "n2"}


def test_without_the_gate_cross_lingual_noise_leaks_into_the_fused_list():
    # BM25 across scripts: its rank-1 is a meaningless passage ("noise"); equal-weight RRF cannot discount it
    bm25, dense = _run(["noise", "x", "y"]), _run(["g", "noise2", "z"])
    ungated = E.fuse_slice(bm25, dense, cross_lingual=True, gate_language=False)
    gated = E.fuse_slice(bm25, dense, cross_lingual=True, gate_language=True)
    assert ungated["q"][0][0] == "g" and ungated["q"][0][1] == "noise"  # noise reaches rank 2 ...
    assert gated["q"][0] == ["g", "noise2", "z"]  # ... and is absent when gated


def test_topk_from_scores_orders_and_filters():
    scores = np.array([[0.1, 0.9, 0.5, 0.0]])
    pids = np.array(["a", "b", "c", "d"])
    assert E._topk_from_scores(scores, pids, k=3)[0][0] == ["b", "c", "a"]
    ranked, vals = E._topk_from_scores(scores, pids, k=4, positive_only=True)[0]
    assert ranked == ["b", "c", "a"] and math.isclose(vals[0], 0.9)  # zero-score passages are dropped for BM25


def test_direction_macro_averages_slices_then_directions():
    slices = {
        "tydi_ar": {"mrr@10": 0.9, "n": 10},
        "arcd_ar": {"mrr@10": 0.5, "n": 10},
        "xquad_ar2en": {"mrr@10": 0.4, "n": 10},
        "xquad_en2ar": {"mrr@10": 0.8, "n": 10},
        "tydi_en": {"mrr@10": 0.9, "n": 10},
    }
    d = E.direction_macro(slices)
    assert math.isclose(d["ar->ar"]["mrr@10"], 0.7) and math.isclose(d["en->en"]["mrr@10"], 0.9)
    assert math.isclose(d["macro"]["mrr@10"], (0.7 + 0.9 + 0.4 + 0.8) / 4)


def test_result_path_is_valid_on_every_os():
    p = E.result_path("e5-small>>rr_ft_bce_s13", "test", "json")
    assert ">" not in p.name and p.name == "e5-small__rr_ft_bce_s13.test.json"
