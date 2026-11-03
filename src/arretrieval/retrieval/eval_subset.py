"""Score already-evaluated systems (stored test runs) on the query subset used for an expensive reranker run.

Run: python -m arretrieval.retrieval.eval_subset --n 150 --systems e5-small "e5-small>>rr_ft_bce_s13" ...
Writes runs/retrieval/results/subset_<n>.scores.json  {system: {direction: {mrr@10, hit@1, hit@10, ndcg@10}}}
"""

from __future__ import annotations

import argparse
import json

from . import evalkit as E
from .paths import RESULTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    args = ap.parse_args()
    keep = set(json.loads((RESULTS / f"subset_{args.n}.json").read_text()))
    queries = E.load_queries()
    queries = queries[queries.qid.isin(keep)]
    out = {}
    for name in args.systems:
        try:
            runs = E.load_runs(name, "test")
        except FileNotFoundError:
            continue
        runs = {sl: {q: v for q, v in r.items() if q in keep} for sl, r in runs.items()}
        res = E.evaluate_runs(runs, queries, "test")
        out[name] = {
            d: {k: round(v, 4) for k, v in m.items() if k in ("mrr@10", "hit@1", "hit@10", "ndcg@10")} for d, m in res["directions"].items()
        }
        print(f"{name:55s} {E.fmt_row(res)}")
    (RESULTS / f"subset_{args.n}.scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
