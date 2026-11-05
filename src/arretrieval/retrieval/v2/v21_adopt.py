"""v2.1 adoption decision (PRE-REGISTERED in PROGRESS.md, 2026-11-05 ~08:50) - judged ONCE on the mean of the two confirmation seeds (18, 19). Standard library only (host python).

(i)   TEST ar->en >= v2 mean - 1 sd and TEST macro >= v2 mean - 1 sd          (v2 = final_m3 + w_m3b_a40_s14..s17, sample sd)
(ii)  MTEB(ara) NamaaMrTydiReranking >= bge-m3's score - 1.0
(iii) MTEB(ara) Mean (Task) >= 72.29
Reported, not gates: Mean(Task) vs bge-m3 with the 19-task bootstrap CI / sign test, the ar-en cross-lingual-subset mean vs e5-large, MIRACL-HN, SemRel24, per-seed values.
Writes results/v21_adoption.json.   Run: PYTHONPATH=lab/retrieval/v2 python3 lab/retrieval/v2/v21_adopt.py
"""

from __future__ import annotations

import json
import os
import statistics as st

try:
    from . import mteb_board, mteb_paired
except ImportError:  # run as a plain script on the Spark with PYTHONPATH=lab/retrieval/v2
    import mteb_board
    import mteb_paired

RES = os.path.expanduser("~/projects/qubit-ai-lab/runs/retrieval/v2/results")
V2_SEEDS = ["final_m3", "w_m3b_a40_s14", "w_m3b_a40_s15", "w_m3b_a40_s16", "w_m3b_a40_s17"]
NEW = ["v21_s18", "v21_s19"]
MTEB_MEAN_THRESHOLD, NAMAA_GAP = 72.29, 1.0
BGE, E5L = "BAAI__bge-m3", "intfloat__multilingual-e5-large"


def test(name: str, key: str) -> float:
    return 100 * json.load(open(f"{RES}/{name}.test.json"))["directions"][key]["mrr@10"]


def main() -> None:
    ref = {k: [test(n, k) for n in V2_SEEDS] for k in ("ar->en", "macro")}
    thr = {k: st.mean(v) - st.stdev(v) for k, v in ref.items()}
    new_test = {k: [test(n, k) for n in NEW] for k in ("ar->en", "macro")}
    boards = {n: mteb_board.board(n) for n in NEW + [BGE, E5L]}
    namaa_bge = boards[BGE]["per_task"]["NamaaMrTydiReranking"]
    namaa_new = [boards[n]["per_task"]["NamaaMrTydiReranking"] for n in NEW]
    mean_new = [boards[n]["Mean(Task)"] for n in NEW]
    crit = {
        "i_test_ar_en": st.mean(new_test["ar->en"]) >= thr["ar->en"],
        "i_test_macro": st.mean(new_test["macro"]) >= thr["macro"],
        "ii_namaa": st.mean(namaa_new) >= namaa_bge - NAMAA_GAP,
        "iii_mteb_mean": st.mean(mean_new) >= MTEB_MEAN_THRESHOLD,
    }
    out = {
        "decision": "ADOPT v2.1" if all(crit.values()) else "v2 STAYS (v2.1 = negative result)",
        "criteria": crit,
        "thresholds": {
            "test_ar_en": thr["ar->en"],
            "test_macro": thr["macro"],
            "namaa": namaa_bge - NAMAA_GAP,
            "mteb_mean": MTEB_MEAN_THRESHOLD,
            "v2_ar_en_mean_sd": [st.mean(ref["ar->en"]), st.stdev(ref["ar->en"])],
            "v2_macro_mean_sd": [st.mean(ref["macro"]), st.stdev(ref["macro"])],
        },
        "v21": {
            "seeds": NEW,
            "test_ar_en": new_test["ar->en"],
            "test_macro": new_test["macro"],
            "test_ar_en_mean": st.mean(new_test["ar->en"]),
            "test_macro_mean": st.mean(new_test["macro"]),
            "namaa": namaa_new,
            "namaa_mean": st.mean(namaa_new),
            "mteb_mean_task": mean_new,
            "mteb_mean_task_mean": st.mean(mean_new),
        },
        "reported_not_gates": {
            "namaa_bge_m3": namaa_bge,
            "ar_en_cross_lingual_subsets": {n: boards[n]["ar-en cross-lingual subsets"] for n in NEW + [BGE, E5L]},
            "miracl_hn": {n: boards[n]["per_task"]["MIRACLRetrievalHardNegatives.v2"] for n in NEW + [BGE]},
            "semrel24": {n: boards[n]["per_task"]["SemRel24STS"] for n in NEW + [BGE]},
            "paired_vs_bge_m3": {n: {k: v for k, v in mteb_paired.compare(n, BGE).items() if k != "per_task"} for n in NEW},
        },
    }
    json.dump(out, open(f"{RES}/v21_adoption.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("decision", "criteria", "thresholds", "v21")}, indent=1))


if __name__ == "__main__":
    main()
