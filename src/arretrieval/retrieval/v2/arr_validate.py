"""Validity gate of the v2.1 proxy component "ar_rerank" (pre-registered: Spearman rho >= 0.70 between the proxy and the known MTEB NamaaMrTydiReranking score over >= 12 existing models).
Standard library only (host python).  Reads results/arr/NAME.json (ar_rerank_proxy.py) and the MTEB result caches; writes results/arr/_gate.json.
"""

from __future__ import annotations

import glob
import json
import os

ROOT = os.path.expanduser("~/projects/mteb-arabic")
RES = os.path.expanduser("~/projects/qubit-ai-lab/runs/retrieval/v2/results")
THRESHOLD, MIN_MODELS = 0.70, 12
# arr model name -> MTEB result-cache directory name holding its NamaaMrTydiReranking score
MAP = {
    "bge-m3": "BAAI__bge-m3",
    "e5-base": "intfloat__multilingual-e5-base",
    "e5-small": "intfloat__multilingual-e5-small",
    "v1": "mzahran1__arabic-english-e5-small-retrieval",
    "w_m3b_a40": "final_m3",
    "wm_a75": "final_m3_wa75_s13",
    "w_m3b_a30": "alpha_w_m3b_a30",
    "wm_a50": "alpha_wm_a50",
    "w_m3b_a60": "alpha_w_m3b_a60",
    "m3_b_s13": "alpha_m3_b_s13",
    "final_base": "final_base",
    "final_small": "final_small",
    "w_m3b_a40_s16": "w_m3b_a40_s16",
    "w_m3b_a40_s17": "w_m3b_a40_s17",
}


def namaa(dirname: str):
    for cache in ("results_cache_v2", "results_cache"):
        for d in glob.glob(f"{ROOT}/{cache}/results/{dirname}/*/"):
            p = d + "NamaaMrTydiReranking.json"
            if os.path.exists(p):
                r = json.load(open(p))
                v = [s["main_score"] for sp in r["scores"].values() for s in sp if "main_score" in s]
                return 100 * sum(v) / len(v)
    return None


def ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def main() -> None:
    pairs = []
    for name, d in MAP.items():
        p = f"{RES}/arr/{name}.json"
        nm = namaa(d)
        if os.path.exists(p) and nm is not None:
            pairs.append((name, 100 * json.load(open(p))["ap"], nm))
    rho = spearman([x[1] for x in pairs], [x[2] for x in pairs]) if len(pairs) >= 3 else float("nan")
    out = {
        "rho": rho,
        "n_models": len(pairs),
        "threshold": THRESHOLD,
        "min_models": MIN_MODELS,
        "passed": bool(len(pairs) >= MIN_MODELS and rho >= THRESHOLD),
        "pairs": [{"model": n, "ar_rerank_ap": a, "namaa_ap": b} for n, a, b in pairs],
    }
    json.dump(out, open(f"{RES}/arr/_gate.json", "w"), indent=1)
    for n, a, b in sorted(pairs, key=lambda x: x[2]):
        print(f"{n:16s} ar_rerank {a:6.2f}   Namaa {b:6.2f}")
    print(f"Spearman rho = {rho:.3f} over {len(pairs)} models -> gate {'PASSED' if out['passed'] else 'FAILED (component 7 dropped)'}")


if __name__ == "__main__":
    main()
