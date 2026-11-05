#!/usr/bin/env python3
"""MTEB(ara, v1) per-task table for v2 models vs reference models, read straight from the two result caches (mteb-arabic/results_cache and results_cache_v2).
Per-task score = main_score averaged over the task's subsets and splits (as the official leaderboard does) x100; Mean(Task) = mean over the 19 tasks.
Read-only; does not write into the mteb-arabic directory.   usage: ara_table.py MODEL_NAME [MODEL_NAME ...] [--ref BAAI__bge-m3]"""
import glob
import json
import os
import sys

ROOT = os.path.expanduser("~/projects/mteb-arabic")
TASKS = ["MIRACLRetrievalHardNegatives.v2", "XQuADRetrieval", "MLQARetrieval", "BelebeleRetrieval", "XPQARetrieval", "WebFAQRetrieval", "NamaaMrTydiReranking", "STS17", "STS22.v2", "SemRel24STS",
         "XNLI", "ArEntail", "AJGT.v2", "TweetEmotionClassification.v2", "TweetSarcasmClassification.v2", "HotelReviewSentimentClassification.v2", "SIB200ClusteringS2S.v2", "FloresBitextMining", "Tatoeba"]


def model_scores(name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for cache in ("results_cache_v2", "results_cache", "results_cache_bs32"):
        for d in glob.glob(f"{ROOT}/{cache}/results/{name}/*/"):
            for t in TASKS:
                p = os.path.join(d, t + ".json")
                if t in out or not os.path.exists(p):
                    continue
                r = json.load(open(p))
                vals = [s["main_score"] for sp in r["scores"].values() for s in sp if "main_score" in s]
                if vals:
                    out[t] = 100 * sum(vals) / len(vals)
    return out


def main() -> None:
    args = sys.argv[1:]
    ref = "BAAI__bge-m3"
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        args = args[:i] + args[i + 2:]
    models = [ref] + args
    sc = {m: model_scores(m) for m in models}
    print("| task | " + " | ".join(m.replace("__", "/") for m in models) + " | " + " | ".join(f"d vs {ref.replace('__', '/')} ({m})" for m in models[1:]) + " |")
    print("|---|" + "---:|" * (2 * len(models) - 1))
    for t in TASKS:
        row = [sc[m].get(t) for m in models]
        d = ["" if row[i] is None or row[0] is None else f"{row[i] - row[0]:+.1f}" for i in range(1, len(models))]
        print(f"| {t} | " + " | ".join("-" if v is None else f"{v:.1f}" for v in row) + " | " + " | ".join(d) + " |")
    means = {m: (sum(sc[m].values()) / len(sc[m]) if sc[m] else float('nan'), len(sc[m])) for m in models}
    print("| **Mean (Task)** | " + " | ".join(f"{means[m][0]:.1f} ({means[m][1]} tasks)" for m in models) + " | " + " | ".join(f"{means[m][0] - means[ref][0]:+.2f}" for m in models[1:]) + " |")


if __name__ == "__main__":
    main()
