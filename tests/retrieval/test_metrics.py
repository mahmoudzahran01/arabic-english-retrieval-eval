"""Hand-checked examples for metrics.py. Run: python -m pytest tests/retrieval -q"""

import math

import numpy as np

from arretrieval.retrieval import metrics as M


def test_hit_mrr_ndcg_single_relevant():
    ranked = ["a", "b", "c", "d"]
    assert M.hit_at_k(ranked, {"b"}, 1) == 0.0
    assert M.hit_at_k(ranked, {"b"}, 5) == 1.0
    assert M.mrr_at_k(ranked, {"b"}) == 0.5
    assert math.isclose(M.ndcg_at_k(ranked, {"b"}), 1 / math.log2(3))  # 0.6309...
    assert M.mrr_at_k(ranked, {"zzz"}) == 0.0
    assert M.ndcg_at_k(ranked, {"a"}) == 1.0


def test_ndcg_two_relevant_hand_computed():
    # relevant at ranks 1 and 3; DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # ideal puts both first: 1 + 1/log2(3) = 1.6309  ->  nDCG = 1.5 / 1.6309 = 0.91972
    ranked = ["r1", "x", "r2", "y"]
    assert math.isclose(M.ndcg_at_k(ranked, {"r1", "r2"}), 1.5 / (1 + 1 / math.log2(3)))
    assert math.isclose(M.ndcg_at_k(ranked, {"r1", "r2"}), 0.91972, abs_tol=1e-4)


def test_mrr_cutoff():
    ranked = [f"d{i}" for i in range(20)]
    assert M.mrr_at_k(ranked, {"d10"}, 10) == 0.0  # rank 11 is outside the cutoff
    assert M.mrr_at_k(ranked, {"d9"}, 10) == 0.1  # rank 10


def test_evaluate_run_macro():
    run = {"q1": ["a", "b", "c"], "q2": ["x", "y"], "q3": []}
    qrels = {"q1": {"b"}, "q2": {"x"}, "q3": {"z"}}
    pq = M.evaluate_run(run, qrels)
    s = M.summarize(pq)
    assert s["n"] == 3
    assert math.isclose(s["hit@1"], 1 / 3)
    assert math.isclose(s["mrr@10"], (0.5 + 1.0 + 0.0) / 3)
    assert math.isclose(s["hit@10"], 2 / 3)


def test_rrf_hand_computed():
    # k=60: a is rank1 in L1 and rank2 in L2 -> 1/61 + 1/62 ; b rank2 in L1, rank1 in L2 -> 1/62 + 1/61 (tie -> id order)
    fused = M.rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused[:2] == ["a", "b"]
    assert fused[2:] == ["c", "d"] or fused[2:] == ["d", "c"]
    # c only in L1 at rank 3 (1/63) ; d only in L2 at rank 3 (1/63) -> tie broken by id
    assert fused[2:] == ["c", "d"]


def test_paired_bootstrap_detects_gain_and_null():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    res = M.paired_bootstrap(a, a + 0.1)
    assert res["ci_lo"] > 0 and res["p_le_0"] == 0.0
    res0 = M.paired_bootstrap(a, a.copy())
    assert res0["delta"] == 0.0
