"""Hybrid = Reciprocal Rank Fusion of a stored BM25 run and a stored dense run (any system with saved runs).

--gate-language: only fuse when query language == passage language; cross-lingual slices fall back to the dense run.
Why: BM25 has no shared vocabulary across scripts, so its "top hit" for an ar->en query is noise (digits, Latin names).
RRF is rank-based and cannot tell that BM25's rank-1 is meaningless, so the noise leaks into the fused list.

Run: python -m arretrieval.retrieval.eval_hybrid --dense e5-small --bm25 bm25-stem --gate-language
"""

from __future__ import annotations

import argparse

from . import evalkit as E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--bm25", default="bm25-stem")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--gate-language", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    queries = E.load_queries()
    d, b = E.load_runs(args.dense, args.split), E.load_runs(args.bm25, args.split)
    qlang = queries.drop_duplicates("slice").set_index("slice")[["qlang", "plang"]]
    out = {}
    for sl in d:
        cross = qlang.loc[sl, "qlang"] != qlang.loc[sl, "plang"]
        out[sl] = E.fuse_slice(b[sl], d[sl], cross_lingual=cross, gate_language=args.gate_language, k=args.k)
    name = args.name or f"hybrid{'-gated' if args.gate_language else ''}-{args.dense}+{args.bm25}"
    res = E.evaluate_runs(out, queries, args.split)
    E.save_result(name, args.split, res, runs=out)
    print(f"[{args.split}] {name}  mrr@10: {E.fmt_row(res)}")


if __name__ == "__main__":
    main()
