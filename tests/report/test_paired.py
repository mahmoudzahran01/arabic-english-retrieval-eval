"""Paired MTEB(ara) statistics: sign test, bootstrap over tasks, parsing of mteb_paired.json, task-type means."""

from __future__ import annotations

import json

import pytest
from report.tabcommon import MTEB, V2

from report import paired as pd_


def test_sign_test_exact_values():
    assert pd_.sign_test_p(9, 9) == 1.0
    assert pd_.sign_test_p(0, 0) == 1.0
    assert pd_.sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert pd_.sign_test_p(1, 5) == pytest.approx(2 * (1 + 6) / 64)


def test_paired_stats_constant_gain_and_ties():
    scores = {f"t{i}": (60.0 + 1.0, 60.0) for i in range(8)} | {"tie": (50.0, 50.0)}
    st = pd_.paired_stats(scores, n_boot=500)
    assert st["wins"] == 8 and st["losses"] == 0 and st["ties"] == 1 and st["n"] == 9
    assert st["mean_diff"] == pytest.approx(8 / 9) and st["ci_lo"] <= st["mean_diff"] <= st["ci_hi"]
    assert st["sign_p"] == pytest.approx(2 / 256)


def test_paired_stats_is_deterministic():
    scores = {f"t{i}": (50.0 + (i % 5) - 2, 50.0) for i in range(12)}
    assert pd_.paired_stats(scores, n_boot=300) == pd_.paired_stats(scores, n_boot=300)
    assert pd_.paired_stats({}) == {"n": 0}


def test_parse_dict_keyed_list_and_subsets():
    entry = {
        "new": "a",
        "ref": "b",
        "per_task": [{"task": "T1", "new": 70, "ref": 60, "diff": 10}, {"task": "T2", "new": 50, "ref": 55}],
        "subsets": {"ar-en cross-lingual subsets": {"per_task": {"T1": [1, 0.5]}}},
    }
    for raw in ({"a vs b": entry}, [{**entry, "label": "a vs b"}]):
        (c,) = pd_.parse(raw)
        assert c.label == "a vs b" and c.scores == {"T1": (70.0, 60.0), "T2": (50.0, 55.0)}
        assert c.subsets["ar-en cross-lingual subsets"] == {"T1": (1.0, 0.5)}
    assert pd_.parse({"x": {"per_task": {"T": {"new": 2, "ref": 1}}}})[0].scores == {"T": (2.0, 1.0)}
    assert pd_.parse("garbage") == [] and pd_.load(V2 / "does_not_exist.json") == []


def test_type_means_reproduce_the_leaderboard_columns():
    lb = json.loads((MTEB / "leaderboard.json").read_text())
    by = {r["model_name"]: r for r in lb["per_task"]}
    for summary in lb["summary"]:
        tm = pd_.type_means({t: v for t, v in by[summary["model_name"]].items() if t != "model_name"})
        for col in (
            "Mean (Task)",
            "Mean (TaskType)",
            "Retrieval",
            "Reranking",
            "STS",
            "PairClassification",
            "Classification",
            "Clustering",
            "BitextMining",
        ):
            assert tm[col] == pytest.approx(summary[col], abs=0.01), (summary["model_name"], col)


def test_committed_paired_file_is_consistent_with_recomputation():
    comps = pd_.load(V2 / "mteb_paired.json")
    assert comps, "results/retrieval_v2/mteb_paired.json is committed"
    for c in comps:
        st = pd_.paired_stats(c.scores)
        pre = c.precomputed
        assert st["mean_diff"] == pytest.approx(pre["mean_diff"], abs=1e-6)
        assert (st["wins"], st["losses"], st["ties"]) == (pre["wins"], pre["losses"], pre["ties"])
        assert st["sign_p"] == pytest.approx(pre["sign_test_p_two_sided"])
        lo, hi = pre["ci95_task_bootstrap"]
        assert abs(st["ci_lo"] - lo) < 0.15 and abs(st["ci_hi"] - hi) < 0.15  # different RNG streams, same interval up to bootstrap noise


def test_from_leaderboard_builds_pairs_for_known_models():
    lb = json.loads((MTEB / "leaderboard.json").read_text())
    comps = pd_.from_leaderboard(lb, [("bge-m3 vs mE5-base", "bge-m3", "multilingual-e5-base"), ("missing", "nope", "bge-m3")])
    assert [c.label for c in comps] == ["bge-m3 vs mE5-base"] and len(comps[0].scores) == 19 and comps[0].source == "leaderboard"


def test_subset_helpers_and_extractor(tmp_path):
    from report import extract_mteb_subsets as ex

    def write(model, rev, task, rows):
        d = tmp_path / model / rev
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{task}.json").write_text(json.dumps({"scores": {"test": [{"hf_subset": s, "main_score": v} for s, v in rows]}}))

    for model, off in (("A__new", 0.1), ("B__ref", 0.0)):
        write(model, "r1", "MLQARetrieval", [("ara-ara", 0.6 + off), ("ara-eng", 0.5 + off), ("eng-ara", 0.4 + off)])
        write(model, "r1", "XQuADRetrieval", [("ar", 0.9 + off)])
    out = ex.extract([tmp_path])
    assert set(out) == {"A__new", "B__ref"} and set(out["A__new"]) == {"MLQARetrieval", "XQuADRetrieval"}
    m = out["A__new"]["MLQARetrieval"]
    assert m["cross"] == pytest.approx(55.0) and m["mono"] == pytest.approx(70.0) and m["all"] == pytest.approx(60.0)
    assert out["A__new"]["XQuADRetrieval"]["cross"] is None
    sc = pd_.subset_comparison(out, "A__new", "B__ref", "cross")
    assert sc == {"MLQARetrieval": (pytest.approx(55.0), pytest.approx(45.0))}  # tasks without cross subsets are skipped
    assert pd_.subset_mean(out, "A__new", "mono") == pytest.approx(85.0) and pd_.subset_mean(out, "nope", "mono") is None
    assert pd_.load_subsets(tmp_path / "missing.json") == {}
