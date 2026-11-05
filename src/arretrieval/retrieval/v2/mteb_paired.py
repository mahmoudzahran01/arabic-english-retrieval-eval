"""MTEB(ara, v1) uncertainty between two models, from the result caches only (CPU, standard library; no GPU, no network).

Per comparison (new vs reference): per-task win / loss / tie counts (a tie is |difference| < 0.005 points), a percentile-bootstrap 95% CI of the Mean(Task) difference
obtained by resampling the 19 TASKS with replacement (10,000 resamples, seed 13), and an exact two-sided paired sign test over the non-tied tasks.
Per-task score = main_score averaged over the task's subsets and splits (as the leaderboard does) x100.   Writes runs/retrieval/v2/results/mteb_paired.json.
usage: python3 mteb_paired.py NEW:REF [NEW:REF ...]          names are result-cache directory names, e.g.  final_m3:BAAI__bge-m3  final_base:intfloat__multilingual-e5-base
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import sys

ROOT = os.path.expanduser("~/projects/mteb-arabic")
OUT = os.path.expanduser("~/projects/qubit-ai-lab/runs/retrieval/v2/results/mteb_paired.json")
TASKS = [
    "MIRACLRetrievalHardNegatives.v2",
    "XQuADRetrieval",
    "MLQARetrieval",
    "BelebeleRetrieval",
    "XPQARetrieval",
    "WebFAQRetrieval",
    "NamaaMrTydiReranking",
    "STS17",
    "STS22.v2",
    "SemRel24STS",
    "XNLI",
    "ArEntail",
    "AJGT.v2",
    "TweetEmotionClassification.v2",
    "TweetSarcasmClassification.v2",
    "HotelReviewSentimentClassification.v2",
    "SIB200ClusteringS2S.v2",
    "FloresBitextMining",
    "Tatoeba",
]
N_BOOT, SEED, TIE = 10_000, 13, 0.005


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


def sign_test_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(new: str, ref: str) -> dict | None:
    a, b = model_scores(new), model_scores(ref)
    tasks = [t for t in TASKS if t in a and t in b]
    if len(tasks) < len(TASKS):
        return {
            "new": new,
            "ref": ref,
            "error": f"incomplete: {len(tasks)}/{len(TASKS)} common tasks",
            "missing_new": [t for t in TASKS if t not in a],
            "missing_ref": [t for t in TASKS if t not in b],
        }
    d = [a[t] - b[t] for t in tasks]
    wins, losses = sum(x > TIE for x in d), sum(x < -TIE for x in d)
    rng = random.Random(SEED)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(N_BOOT))
    return {
        "new": new,
        "ref": ref,
        "n_tasks": len(tasks),
        "mean_new": sum(a[t] for t in tasks) / len(tasks),
        "mean_ref": sum(b[t] for t in tasks) / len(tasks),
        "mean_diff": sum(d) / len(d),
        "ci95_task_bootstrap": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]],
        "boot_p_diff_gt_0": sum(x > 0 for x in boots) / N_BOOT,
        "wins": wins,
        "losses": losses,
        "ties": len(d) - wins - losses,
        "sign_test_p_two_sided": sign_test_p(wins, losses),
        "per_task": [{"task": t, "new": a[t], "ref": b[t], "diff": a[t] - b[t]} for t in tasks],
    }


def main() -> None:
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    for spec in sys.argv[1:]:
        new, _, ref = spec.partition(":")
        r = compare(new, ref)
        res[f"{new} vs {ref}"] = r
        if "error" in r:
            print(f"{new} vs {ref}: {r['error']}")
            continue
        lo, hi = r["ci95_task_bootstrap"]
        print(
            f"{new} vs {ref}: Mean(Task) {r['mean_new']:.3f} vs {r['mean_ref']:.3f}  diff {r['mean_diff']:+.3f} (95% task-bootstrap CI {lo:+.3f}..{hi:+.3f})  "
            f"wins/losses/ties {r['wins']}/{r['losses']}/{r['ties']}  sign test p={r['sign_test_p_two_sided']:.3f}"
        )
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
