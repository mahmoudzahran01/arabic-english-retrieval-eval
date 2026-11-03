"""Stage 5b: retrieve(top-N) -> rerank -> top-10 metrics, from a stored first-stage run.

Run: python -m arretrieval.retrieval.eval_rerank --first-stage e5-small \
        --reranker cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --name rr_mmarco_off_the_shelf
The result name is  <first-stage>>><name>  (e.g. e5-small>>rr_mmarco_off_the_shelf).
"""

from __future__ import annotations

import argparse
import json
import time

from . import evalkit as E
from .paths import RESULTS
from .rerank import Reranker, rerank_slice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-stage", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument(
        "--sample-per-slice",
        type=int,
        default=0,
        help="rerank only a seeded random sample of N queries per slice (for expensive rerankers); the sample is saved to "
        "results/subset_<N>.json so other systems can be scored on exactly the same queries with eval_subset.py",
    )
    args = ap.parse_args()
    passages, queries = E.load_passages(), E.load_queries()
    text = dict(zip(passages.pid, passages.text))
    rr = Reranker(args.reranker, max_len=args.max_len, bs=args.bs)
    for split in args.splits:
        first = E.load_runs(args.first_stage, split)
        t0 = time.time()
        out, kept = {}, []
        for sl, q, _ in E.eval_units(queries, passages, split):
            if sl in first:
                if args.sample_per_slice:
                    q = q.sample(min(args.sample_per_slice, len(q)), random_state=13).reset_index(drop=True)
                    kept += list(q.qid)
                out[sl] = rerank_slice(rr, q, text, first[sl], args.top_n)
        if args.sample_per_slice:
            (RESULTS / f"subset_{args.sample_per_slice}.json").write_text(json.dumps(sorted(kept)))
            queries = queries[queries.qid.isin(set(kept))]
        res = E.evaluate_runs(out, queries, split)
        name = f"{args.first_stage}>>{args.name}"
        E.save_result(
            name,
            split,
            res,
            runs=out if split == "test" else None,
            extra={
                "first_stage": args.first_stage,
                "reranker": args.reranker,
                "top_n": args.top_n,
                "rerank_seconds": round(time.time() - t0, 1),
            },
        )
        first = {sl: {q: v for q, v in r.items() if q in set(queries.qid)} for sl, r in first.items()} if args.sample_per_slice else first
        base = E.evaluate_runs(first, queries, split)
        print(
            f"[{split}] {name}  rerank {time.time() - t0:.0f}s\n  first-stage mrr@10: {E.fmt_row(base)}\n  reranked    mrr@10: {E.fmt_row(res)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
