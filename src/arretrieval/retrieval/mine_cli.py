"""Mine raw hard-negative candidates for all train queries with a baseline model.

Run: python -m arretrieval.retrieval.mine_cli --model e5-small   (or bge-m3)
"""

from __future__ import annotations

import argparse

from . import evalkit as E
from .eval_baselines import DENSE
from .negatives import mine_raw, raw_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(DENSE))
    ap.add_argument("--tag", default=None, help="output tag (default: the model name); e.g. e5-small-xl after cross-lingual queries exist")
    ap.add_argument(
        "--mono-only",
        action="store_true",
        help="mine only the monolingual train queries even when queries.parquet (with translations) exists; "
        "the false-negative audit and the monolingual ablations use this file",
    )
    args = ap.parse_args()
    hf, qp, pp = DENSE[args.model]
    model = E.Dense(hf, args.model, qp, pp, cache=True, bs=16 if args.model == "bge-m3" else 64)
    raw = mine_raw(model, E.load_queries(mono_only=args.mono_only), E.load_passages())
    raw.to_parquet(raw_path(args.tag or args.model), index=False)
    print(f"{len(raw)} train queries mined with {args.model}")
    print(
        "gold rank among its train pool: top1 %.3f  top10 %.3f  top30 %.3f  (not in top100: %.3f)"
        % (
            (raw.pos_rank == 1).mean(),
            ((raw.pos_rank >= 1) & (raw.pos_rank <= 10)).mean(),
            ((raw.pos_rank >= 1) & (raw.pos_rank <= 30)).mean(),
            (raw.pos_rank > 100).mean(),
        )
    )
    gap = raw.apply(lambda r: r.pos_score - r.cand_scores[0] if r.cand_pids[0] != r.pos_pid else r.pos_score - r.cand_scores[1], axis=1)
    print("score gap  gold - best_non_gold_candidate: quantiles", gap.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist())


if __name__ == "__main__":
    main()
