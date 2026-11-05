"""v2 hard-negative mining with the BASELINE e5-small (same miner as v1) over the v2 query table.

Two candidate lists per TRAIN query, same row schema as v1's raw files (so negatives.select() is reused unchanged):
  same  : top-100 of the TRAIN pool in the gold passage's language        (what v1 used; for an ar->en query these are English passages
                                                                           ranked by the Arabic query, i.e. already cross-lingual)
  other : top-100 of the TRAIN pool in the OTHER language                 (for an ar->en query: Arabic passages = the same-language distractors
                                                                           of a mixed Arabic+English index; for en->ar: English ones)
pos_score is always the score of the gold passage (so the margin filter drops candidates that outscore the gold).

Run (GPU): python -m arretrieval.retrieval.v2.mine_v2 --part v1|xl2
  part v1 = every train query of v1 (v1 slices), part xl2 = the new train_xl2_* queries.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from arretrieval.retrieval import evalkit as E

from .common_v2 import BASES, V2MINED, data_family, load_passages_v2, load_queries_v2

TOP_K = 100


def mine(model: E.Dense, q: pd.DataFrame, passages: pd.DataFrame, mode: str) -> pd.DataFrame:
    import torch

    train_p = passages[passages.pool == "train"].reset_index(drop=True)
    if mode == "same":  # only the gold passages' language is searched (v2.1: skips encoding the other language)
        train_p = train_p[train_p.lang.isin(set(q.plang))].reset_index(drop=True)
    pe = model.encode(list(train_p.text), "p")
    idx = {p: i for i, p in enumerate(train_p.pid)}
    lang = train_p.lang.to_numpy()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pet = torch.from_numpy(pe).to(dev)
    rows = []
    for plang, g in q.groupby("plang"):
        target = plang if mode == "same" else ("en" if plang == "ar" else "ar")
        cols = np.flatnonzero(lang == target)
        qe = torch.from_numpy(model.encode(list(g.text), "q")).to(dev)
        scores = (qe @ pet[cols].T).float().cpu().numpy()
        gold = (qe * pet[[idx[p] for p in g.pos_pid]]).sum(1).float().cpu().numpy()
        pids = train_p.pid.to_numpy()[cols]
        for i, (qid, pos) in enumerate(zip(g.qid, g.pos_pid)):
            top = np.argpartition(-scores[i], TOP_K)[:TOP_K]
            top = top[np.argsort(-scores[i][top], kind="stable")]
            rows.append(
                {
                    "qid": qid,
                    "pos_pid": pos,
                    "pos_score": float(gold[i]),
                    "pos_rank": int((scores[i] > gold[i]).sum()) + 1,
                    "cand_pids": [pids[j] for j in top],
                    "cand_scores": scores[i][top].astype(float).tolist(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["v1", "xl2", "v3", "all"])
    ap.add_argument("--miner", default="", help="mine with this model dir instead of the baseline (round-2 / ANCE-style negatives)")
    ap.add_argument("--tag", default="", help="name of the miner in the raw file name (default: --base)")
    ap.add_argument("--prefix", default="e5", choices=["e5", "none"])
    ap.add_argument("--base", default="e5-small")
    ap.add_argument("--slices", default="", help="v2.1: comma list of train slices to mine (default: every query of the part)")
    ap.add_argument("--modes", default="same,other", help="v2.1: which candidate lists to mine (training only uses 'same')")
    args = ap.parse_args()
    hf, qp, pp = BASES[args.base]
    queries = load_queries_v2()
    tr = queries[queries.split == "train"]
    fam = data_family(tr.slice)
    tr = tr[fam == {"v1": "v1", "xl2": "v2", "v3": "v3"}[args.part]] if args.part != "all" else tr
    if args.miner:
        qp, pp = ("query: ", "passage: ") if args.prefix == "e5" else ("", "")
        hf = args.miner
    if args.slices:
        tr = tr[tr.slice.isin(set(args.slices.split(",")))]
    print(f"mining {len(tr)} queries ({tr.slice.value_counts().to_dict()}) with {hf}", flush=True)
    tag = args.tag or args.base
    model = E.Dense(hf, tag + "-v2mine", qp, pp, cache=not args.miner, bs=32 if args.prefix == "none" else 64)
    passages = load_passages_v2()
    for mode in args.modes.split(","):
        raw = mine(model, tr, passages, mode)
        raw.to_parquet(V2MINED / f"raw_{mode}_{tag}_{args.part}.parquet", index=False)
        print(
            f"[{mode}/{args.part}] {len(raw)} queries; gold rank in the mined pool: top1 {(raw.pos_rank == 1).mean():.3f} "
            f"top10 {(raw.pos_rank <= 10).mean():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
