"""Agreement and uncertainty statistics used to score LLM judges: every expected value is derived by hand in a comment."""

import math

import numpy as np

from arretrieval.rageval import stats as S


def test_kappa_hand_checked():
    # truth: 25 pos / 25 neg ; judge: says pos for 20 of the 25 true pos and for 10 of the 25 true neg
    #   TP=20 FN=5 FP=10 TN=15 ; po = 35/50 = 0.7 ; p_pos(truth)=0.5, p_pos(judge)=30/50=0.6
    #   pe = 0.5*0.6 + 0.5*0.4 = 0.5 ; kappa = (0.7-0.5)/(1-0.5) = 0.4
    t = [1] * 25 + [0] * 25
    p = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert math.isclose(S.cohen_kappa(t, p), 0.4)
    assert S.confusion(t, p).tolist() == [[15, 10], [5, 20]]


def test_raw_agreement_misleads():
    # 95 of 100 answers are good; a judge that says "good" to everything agrees 95% of the time but kappa = 0
    t = [1] * 95 + [0] * 5
    p = [1] * 100
    assert math.isclose(S.raw_agreement(t, p), 0.95)
    assert math.isclose(S.cohen_kappa(t, p), 0.0, abs_tol=1e-12)


def test_prf():
    r = S.prf([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])  # TP=2 FP=1 FN=1
    assert math.isclose(r["precision"], 2 / 3) and math.isclose(r["recall"], 2 / 3) and math.isclose(r["f1"], 2 / 3)


def test_perfect_and_inverse_kappa():
    assert S.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert math.isclose(S.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]), -1.0)


def test_wilson_bounds():
    lo, hi = S.wilson(8, 10)  # known value: 0.490 - 0.943
    assert abs(lo - 0.490) < 0.005 and abs(hi - 0.943) < 0.005


def test_bootstrap_covers_truth_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rng.binomial(1, 0.7, 30).astype(float)
    large = rng.binomial(1, 0.7, 600).astype(float)
    p1, lo1, hi1 = S.bootstrap_ci(small, n_boot=2000)
    p2, lo2, hi2 = S.bootstrap_ci(large, n_boot=2000)
    assert (hi1 - lo1) > 2.5 * (hi2 - lo2)  # width ~ 1/sqrt(n): 20x more data -> ~4.5x narrower
    assert lo2 < 0.7 < hi2


def test_paired_bootstrap_detects_and_rejects():
    a = np.array([0] * 50 + [1] * 50, dtype=float)
    b = a.copy()
    b[:20] = 1  # B fixes 20 failures, breaks none -> diff 0.20
    r = S.paired_bootstrap(a, b, n_boot=4000)
    assert math.isclose(r["diff"], 0.2) and r["lo"] > 0 and r["p"] < 0.01
    c = a.copy()
    c[:3] = 1  # 3 wins out of 100 -> not distinguishable from 0
    r2 = S.paired_bootstrap(a, c, n_boot=4000)
    assert r2["lo"] <= 0.01


def test_mcnemar_exact():
    a = np.array([0] * 8 + [1] * 2)
    b = np.array([1] * 8 + [1] * 2)  # 8 discordant pairs all in B's favour: p = 2*(0.5^8) = 0.0078
    r = S.mcnemar_exact(a, b)
    assert r["n01"] == 8 and r["n10"] == 0 and math.isclose(r["p"], 2 * 0.5**8)


def test_cluster_bootstrap_resamples_whole_clusters():
    # 4 clusters of 5 identical outcomes: every resample is a mix of whole clusters, so the CI is wider than the i.i.d. one
    vals = np.array([1.0] * 5 + [0.0] * 5 + [1.0] * 5 + [0.0] * 5)
    clusters = ["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5
    p, lo_c, hi_c = S.bootstrap_ci(vals, n_boot=2000, clusters=clusters)
    _, lo_i, hi_i = S.bootstrap_ci(vals, n_boot=2000)
    assert p == 0.5 and (hi_c - lo_c) > (hi_i - lo_i)


def test_prf_handles_no_positive_predictions_and_all_correct():
    r = S.prf([1, 1, 0], [0, 0, 0])
    assert r["recall"] == 0.0 and np.isnan(r["precision"]) and r["tp"] == 0
    assert S.prf([1, 0], [1, 0])["f1"] == 1.0


def test_kappa_is_nan_without_label_variation_in_a_disagreeing_pair():
    assert np.isnan(S.cohen_kappa([], []))
    assert S.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0  # identical constant raters: perfect agreement by convention


def test_min_detectable_effect_shrinks_with_n():
    assert S.min_detectable_effect(180) > S.min_detectable_effect(720)
    assert math.isclose(S.min_detectable_effect(180, p=0.7, rho=0.5), 0.096, abs_tol=0.002)  # the "9.6 points" quoted in the A/B report


def test_wilson_interval_of_a_perfect_score_is_one_sided():
    lo, hi = S.wilson(20, 20)  # 20/20 abstentions only supports "at least ~84%": the lower bound quoted in the results
    assert abs(lo - 0.839) < 0.005 and abs(hi - 1.0) < 1e-9
