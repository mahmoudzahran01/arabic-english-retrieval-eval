"""Retrieve top-50 with a v2 first stage -> v1's fine-tuned reranker (rr_ft_bce_s13) -> test metrics (same protocol as v1 eval_rerank.py).

Needs the first stage's stored test run (results/{name}.test.runs.json, written by train_v2 --test / eval_v2 --splits test).
Run (GPU): python -m arretrieval.retrieval.v2.rerank_v2 --first-stage NAME [NAME ...]
"""

from __future__ import annotations

import argparse
import json
import time

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval.paths import MODELS
from arretrieval.retrieval.rerank import Reranker, rerank_slice

from . import eval_v2 as EV
from .common_v2 import V2RES, load_queries_v2

RERANKER = str(MODELS / "rr_ft_bce_s13")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-stage", nargs="+", required=True)
    ap.add_argument("--reranker", default=RERANKER)
    ap.add_argument("--tag", default="rr_ft_bce_s13")
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()
    passages, queries = E.load_passages(), load_queries_v2()
    text = dict(zip(passages.pid, passages.text))
    rr = Reranker(args.reranker, max_len=256, bs=128)
    for fs in args.first_stage:
        raw = json.loads((V2RES / f"{fs}.test.runs.json").read_text())
        first = {sl: {q: (p, [0.0] * len(p)) for q, p in r.items()} for sl, r in raw.items()}
        t0 = time.time()
        out = {sl: rerank_slice(rr, q, text, first[sl], args.top_n) for sl, q, _ in E.eval_units(queries, passages, "test") if sl in first}
        res = E.evaluate_runs(out, queries, "test")
        name = f"{fs}>>{args.tag}"
        EV.save(name, "test", res, runs=out, extra={"first_stage": fs, "reranker": args.reranker, "top_n": args.top_n})
        print(
            f"[test] {name} {time.time() - t0:.0f}s reranked mrr@10: {E.fmt_row(res)} macro={res['directions']['macro']['mrr@10']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
