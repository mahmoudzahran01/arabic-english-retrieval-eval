"""MTEB(ara, v1) board for selected models straight from the result caches (CPU, standard library) - same definitions as mteb-arabic/scripts/make_leaderboard.py:
per-task score = main_score averaged over subsets and splits x100; Mean (Task); Mean (TaskType); 'ar-only subsets' / 'ar-en cross-lingual subsets' means (per-task mean over the
mono / cross subsets, then mean over tasks).  Adds the v2 models to the 10-model local leaderboard WITHOUT touching the MTEB agent's files.
Writes runs/retrieval/v2/results/mteb_board.json.   usage: python3 mteb_board.py MODEL_DIR_NAME [...]
"""

from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.expanduser("~/projects/mteb-arabic")
OUT = os.path.expanduser("~/projects/qubit-ai-lab/runs/retrieval/v2/results/mteb_board.json")
TYPES = {
    "Retrieval": [
        "MIRACLRetrievalHardNegatives.v2",
        "XQuADRetrieval",
        "MLQARetrieval",
        "BelebeleRetrieval",
        "XPQARetrieval",
        "WebFAQRetrieval",
    ],
    "Reranking": ["NamaaMrTydiReranking"],
    "STS": ["STS17", "STS22.v2", "SemRel24STS"],
    "PairClassification": ["XNLI", "ArEntail"],
    "Classification": [
        "AJGT.v2",
        "TweetEmotionClassification.v2",
        "TweetSarcasmClassification.v2",
        "HotelReviewSentimentClassification.v2",
    ],
    "Clustering": ["SIB200ClusteringS2S.v2"],
    "BitextMining": ["FloresBitextMining", "Tatoeba"],
}
CROSS = {
    "MLQARetrieval": {"ara-eng", "eng-ara"},
    "BelebeleRetrieval": {"arb_Arab-eng_Latn", "eng_Latn-arb_Arab"},
    "XPQARetrieval": {"ara-eng", "eng-ara"},
    "STS17": {"en-ar"},
    "FloresBitextMining": {"arb_Arab-eng_Latn", "eng_Latn-arb_Arab"},
    "Tatoeba": {"ara-eng"},
}
PARAMS_M = {
    "BAAI__bge-m3": 568,
    "intfloat__multilingual-e5-large": 560,
    "intfloat__multilingual-e5-base": 278,
    "intfloat__multilingual-e5-small": 118,
    "mzahran1__arabic-english-e5-small-retrieval": 118,
    "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2": 118,
    "sentence-transformers__LaBSE": 471,
    "final_m3": 568,
    "w_m3b_a40_s16": 568,
    "w_m3b_a40_s17": 568,
    "final_base": 278,
    "final_small": 118,
}
ALL = [t for ts in TYPES.values() for t in ts]


def rows(name: str):
    out = {}
    for cache in ("results_cache_v2", "results_cache", "results_cache_bs32"):
        for d in glob.glob(f"{ROOT}/{cache}/results/{name}/*/"):
            for t in ALL:
                p = os.path.join(d, t + ".json")
                if t not in out and os.path.exists(p):
                    r = json.load(open(p))
                    out[t] = [(s.get("hf_subset"), s["main_score"]) for sp in r["scores"].values() for s in sp if "main_score" in s]
    return out


def board(name: str) -> dict | None:
    r = rows(name)
    if len(r) < len(ALL):
        return None
    per = {t: 100 * sum(v for _, v in r[t]) / len(r[t]) for t in ALL}
    tm = {ty: sum(per[t] for t in ts) / len(ts) for ty, ts in TYPES.items()}

    def sub_mean(cross: bool):
        vals = []
        for t in ALL:
            sel = [v for s, v in r[t] if (s in CROSS.get(t, set())) == cross]
            if sel:
                vals.append(100 * sum(sel) / len(sel))
        return sum(vals) / len(vals)

    return {
        "model": name,
        "params_M": PARAMS_M.get(name),
        "Mean(Task)": sum(per.values()) / len(per),
        "Mean(TaskType)": sum(tm.values()) / len(tm),
        **{f"type:{k}": v for k, v in tm.items()},
        "ar-only subsets": sub_mean(False),
        "ar-en cross-lingual subsets": sub_mean(True),
        "per_task": per,
    }


def main() -> None:
    names = sys.argv[1:] or [
        "BAAI__bge-m3",
        "intfloat__multilingual-e5-large",
        "intfloat__multilingual-e5-base",
        "intfloat__multilingual-e5-small",
        "mzahran1__arabic-english-e5-small-retrieval",
        "final_m3",
        "w_m3b_a40_s16",
        "w_m3b_a40_s17",
        "final_base",
        "final_small",
    ]
    res = [b for b in (board(n) for n in names) if b]
    json.dump(res, open(OUT, "w"), indent=1)
    print(
        "| model | params | Mean (Task) | Mean (TaskType) | Retrieval | Reranking | STS | PairClf | Classif. | Clustering | Bitext | ar-only subsets | ar-en cross-lingual subsets |\n|---|---:|"
        + "---:|" * 11
    )
    for b in sorted(res, key=lambda b: -b["Mean(Task)"]):
        print(
            f"| {b['model']} | {b['params_M'] or '?'}M | {b['Mean(Task)']:.2f} | {b['Mean(TaskType)']:.2f} | "
            + " | ".join(f"{b['type:' + k]:.1f}" for k in TYPES)
            + f" | {b['ar-only subsets']:.1f} | {b['ar-en cross-lingual subsets']:.1f} |"
        )


if __name__ == "__main__":
    main()
