"""Describe a raw mining file and (optionally) the LLM false-negative judgments.

Outputs runs/retrieval/results/mining_<model>.json with:
  * how often the first-stage retriever ranks the gold first / in top 10 / 30 on its own TRAIN pool
  * distribution of the score gap (gold - candidate) by candidate rank
  * how many candidates each margin / title / duplicate filter removes
  * (if fn_judge.jsonl exists) judge 'yes' rate on non-gold candidates by rank band and by score gap, plus the judge's
    agreement with the labelled gold passages (calibration)
Run: python -m arretrieval.retrieval.analyze_mining --model e5-small
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import evalkit as E
from .negatives import NegPolicy, raw_path, select
from .paths import RESULTS


def judged_table() -> dict | None:
    p = RESULTS / "fn_judge.jsonl"
    if not p.exists() or p.stat().st_size == 0:
        return None
    j = pd.read_json(p, lines=True)
    j["yes"] = j.answer.str.startswith("yes")
    j = j[j.answer != "error"]
    out = {"n_judged": len(j), "gold_agreement(yes rate on labelled gold)": round(float(j[j.kind == "gold"].yes.mean()), 3)}
    neg = j[j.kind == "neg"]
    out["by_band"] = {b: {"n": int(len(g)), "yes_rate": round(float(g.yes.mean()), 3)} for b, g in neg.groupby("band")}
    out["by_lang"] = {l: {"n": int(len(g)), "yes_rate": round(float(g.yes.mean()), 3)} for l, g in neg.groupby("lang")}
    bins = [-9, 0.0, 0.02, 0.05, 0.1, 9]
    neg = neg.assign(gap_bin=pd.cut(neg.gap, bins))
    out["by_score_gap(gold - candidate)"] = {
        str(b): {"n": int(len(g)), "yes_rate": round(float(g.yes.mean()), 3)} for b, g in neg.groupby("gap_bin", observed=True)
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small")
    args = ap.parse_args()
    raw = pd.read_parquet(raw_path(args.model))
    passages = E.load_passages()
    res: dict = {"model": args.model, "n_queries": len(raw)}
    res["gold_rank_on_train_pool"] = {
        "top1": float((raw.pos_rank == 1).mean()),
        "top10": float((raw.pos_rank <= 10).mean()),
        "top30": float((raw.pos_rank <= 30).mean()),
        "beyond_100": float((raw.pos_rank > 100).mean()),
    }
    by_rank = {r: [] for r in (1, 2, 3, 5, 10, 20, 30)}
    for r in raw.itertuples():
        negs = [(p, s) for p, s in zip(r.cand_pids, r.cand_scores) if p != r.pos_pid]
        for rk in by_rank:
            by_rank[rk].append(r.pos_score - negs[rk - 1][1])
    res["score_gap_by_negative_rank"] = {
        f"rank{rk}": {
            "median": round(float(np.median(v)), 4),
            "p10": round(float(np.quantile(v, 0.1)), 4),
            "p90": round(float(np.quantile(v, 0.9)), 4),
            "frac_beating_gold": round(float(np.mean(np.array(v) < 0)), 3),
        }
        for rk, v in by_rank.items()
    }
    filters = {}
    for label, pol in {
        "no filter (ranks 1-30)": NegPolicy(n_neg=3, use_margin=False),
        "margin 0.00": NegPolicy(n_neg=3, margin=0.0),
        "margin 0.02": NegPolicy(n_neg=3, margin=0.02),
        "margin 0.05": NegPolicy(n_neg=3, margin=0.05),
        "margin 0.02 + drop same article": NegPolicy(n_neg=3, margin=0.02, same_title=True),
        "skip top-9 (ranks 10-30)": NegPolicy(n_neg=3, rank_lo=10, use_margin=False),
    }.items():
        negs, st = select(raw, passages, pol)
        filters[label] = {k: v for k, v in st.items() if k != "policy"}
    res["filters(counts summed over all train queries; each query has its top-100 candidates, n_neg=3 kept)"] = filters
    res["judge"] = judged_table()
    (RESULTS / f"mining_{args.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
