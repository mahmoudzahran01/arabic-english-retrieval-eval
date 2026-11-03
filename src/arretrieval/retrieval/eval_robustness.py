"""Orthographic-noise robustness (Arabic queries only).

Real users type Arabic quickly: hamza is dropped (أ/إ/آ -> ا), ta marbuta becomes ha (ة -> ه), alef maqsura becomes ya (ى -> ي).
Wikipedia-derived test queries are cleanly spelled, so they under-state this problem. We corrupt ONLY the queries with a
seeded character-level process and compare systems on ar->ar test slices. Passages are left untouched.

Run: python -m arretrieval.retrieval.eval_robustness --dense e5-small bge-m3 [--model-dir NAME ...]
"""

from __future__ import annotations

import argparse
import json
import random

from . import evalkit as E
from .eval_baselines import DENSE
from .paths import MODELS, RESULTS

SLICES = ("tydi_ar", "arcd_ar", "xquad_ar2ar")
SWAPS = {"أ": ("ا", 0.6), "إ": ("ا", 0.6), "آ": ("ا", 0.6), "ة": ("ه", 0.4), "ى": ("ي", 0.5)}


def perturb(text: str, rng: random.Random) -> str:
    return "".join(SWAPS[c][0] if c in SWAPS and rng.random() < SWAPS[c][1] else c for c in text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", nargs="*", default=["e5-small", "bge-m3"])
    ap.add_argument("--model-dir", nargs="*", default=[], help="fine-tuned model dir names under runs/retrieval/models")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    passages, queries = E.load_passages(), E.load_queries()
    rng = random.Random(args.seed)
    qs = queries[(queries.split == "test") & queries.slice.isin(SLICES)].copy()
    clean = qs.copy()
    qs["text"] = [perturb(t, rng) for t in qs.text]
    changed = float((qs.text.values != clean.text.values).mean())
    out = {"fraction_of_queries_changed": round(changed, 3), "slices": list(SLICES), "systems": {}}

    def evaluate(name: str, retriever) -> None:
        r = {}
        for tag, q in (("clean", clean), ("noisy", qs)):
            runs = {sl: retriever.rank(g.reset_index(drop=True), E.pool_corpus(passages, "test", "ar")) for sl, g in q.groupby("slice")}
            res = E.evaluate_runs(runs, queries, "test")
            r[tag] = {sl: round(res["slices"][sl]["mrr@10"], 4) for sl in SLICES}
            r[tag]["macro"] = round(sum(r[tag].values()) / len(SLICES), 4)
        r["delta_macro"] = round(r["noisy"]["macro"] - r["clean"]["macro"], 4)
        out["systems"][name] = r
        print(f"{name:28s} clean {r['clean']['macro']:.3f}  noisy {r['noisy']['macro']:.3f}  delta {r['delta_macro']:+.3f}", flush=True)

    for mode in ("raw", "norm", "stem"):
        evaluate(f"bm25-{mode}", E.BM25(mode))
    for tag in args.dense:
        hf, qp, pp = DENSE[tag]
        evaluate(tag, E.Dense(hf, tag, qp, pp, cache=True))
    for name in args.model_dir:
        evaluate(name, E.Dense(str(MODELS / name), name, "query: ", "passage: ", cache=False, bs=128))
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
