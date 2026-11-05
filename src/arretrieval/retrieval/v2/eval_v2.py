"""v2 evaluation: v1 protocol (per-slice, per-direction MRR@10 / Hit / nDCG on the split's own pools) on the v2 query table,
results stored under runs/retrieval/v2/results, plus an optional mixed-index diagnostic.

CLI (GPU):  python -m arretrieval.retrieval.v2.eval_v2 --model PATH_OR_HF_ID --name NAME --prefix e5|none --splits dev [test] [--mixed]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval import metrics as M

from .common_v2 import V2RES, load_passages_v2, load_queries_v2

E.DIRECTION_OF["dev_miracl_ar"] = "miracl"  # selection-only hold-out slice (split devM): kept out of the 4-direction macros


def evaluate_devM(dense: E.Dense, name: str, queries: pd.DataFrame, passages: pd.DataFrame) -> float | None:
    """MRR@10 on the 500 held-out MIRACL-ar TRAIN queries over their own mini-corpus (pool devM). Returns None if the replay data is not built."""
    if not (queries.split == "devM").any():
        return None
    run = E.run_system(dense, queries, passages, "devM")
    res = E.evaluate_runs(run, queries, "devM")
    save(name, "devM", res)
    v = res["slices"]["dev_miracl_ar"]["mrr@10"]
    print(f"[{name}] devM (held-out MIRACL-ar train, {res['slices']['dev_miracl_ar']['n']} q) mrr@10={v:.4f}", flush=True)
    return v


def save(name: str, split: str, res: dict, runs: dict | None = None, extra: dict | None = None) -> None:
    payload = {"system": name, "split": split, "slices": res["slices"], "directions": res["directions"], **(extra or {})}
    (V2RES / f"{name}.{split}.json").write_text(json.dumps(payload, indent=1))
    np.savez_compressed(V2RES / f"{name}.{split}.perq.npz", **{f"{sl}|{m}": v for sl, d in res["perq"].items() for m, v in d.items()})
    if runs is not None:
        slim = {sl: {q: v[0] for q, v in r.items()} for sl, r in runs.items()}
        (V2RES / f"{name}.{split}.runs.json").write_text(json.dumps(slim))


def load_json(name: str, split: str) -> dict | None:
    p = V2RES / f"{name}.{split}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_perq(name: str, split: str) -> dict:
    z = np.load(V2RES / f"{name}.{split}.perq.npz")
    out: dict = {}
    for k in z.files:
        sl, m = k.split("|")
        out.setdefault(sl, {})[m] = z[k]
    return out


def evaluate(
    dense: E.Dense, name: str, splits: list[str], queries: pd.DataFrame, passages: pd.DataFrame, extra: dict | None = None
) -> dict:
    out = {}
    for split in splits:
        run = E.run_system(dense, queries, passages, split)
        res = E.evaluate_runs(run, queries, split)
        save(name, split, res, runs=run if split == "test" else None, extra=extra)
        d = res["directions"]
        if "macro" in d:
            print(f"[{name}] {split} mrr@10: {E.fmt_row(res)}  macro={d['macro']['mrr@10']:.4f}", flush=True)
        else:
            print(f"[{name}] {split} slices: " + " ".join(f"{k}={v['mrr@10']:.4f}" for k, v in res["slices"].items()), flush=True)
        out[split] = res
    return out


# ------------------------------------------------------------------------------------------------ mixed-index diagnostic
def mixed_eval(dense: E.Dense, name: str, split: str, queries: pd.DataFrame, passages: pd.DataFrame) -> dict:
    """Cross-lingual queries searched in ONE index holding the split's Arabic and English passages together.

    strict  : only the passage in the *other* language (the labelled gold) counts. For XQuAD the parallel same-language
              paragraph (which also answers the question) is then a distractor, so this measures ranking across scripts.
    lenient : XQuAD twin paragraphs both count (practical answerability); dev has no twins, so lenient == strict there.
    wrong_lang_top1 : share of queries whose top-1 passage is in the query's own language (language bias).
    """
    corpus = passages[passages.pool == split].reset_index(drop=True)
    qs = queries[queries.split == split]
    xl = qs[qs.qlang != qs.plang]
    twin = {}
    if (
        split == "test"
    ):  # XQuAD parallel paragraphs: for query (src_qid, lang L -> passage lang P) the twin is the L-language gold of the L2L slice
        mono = qs[qs.slice.isin(["xquad_ar2ar", "xquad_en2en"])]
        twin = {(r.src_qid, r.qlang): r.pos_pid for r in mono.itertuples()}
    lang_of = dict(zip(corpus.pid, corpus.lang))
    ce = dense.encode(list(corpus.text), "p")
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cet = torch.from_numpy(ce).to(dev)
    out = {}
    for sl, g in xl.groupby("slice"):
        qe = dense.encode(list(g.text), "q")
        scores = (torch.from_numpy(qe).to(dev) @ cet.T).float().cpu().numpy()
        runs = dict(zip(g.qid, E._topk_from_scores(scores, corpus.pid.to_numpy(), 100)))
        strict = {q: {p} for q, p in zip(g.qid, g.pos_pid)}
        lenient = {q: ({p, twin[(s, ql)]} if (s, ql) in twin else {p}) for q, p, s, ql in zip(g.qid, g.pos_pid, g.src_qid, g.qlang)}
        ps = M.evaluate_run({q: v[0] for q, v in runs.items()}, strict)
        pl = M.evaluate_run({q: v[0] for q, v in runs.items()}, lenient)
        own = np.array([lang_of[runs[q][0][0]] == ql for q, ql in zip(sorted(strict), g.set_index("qid").loc[sorted(strict)].qlang)])
        out[sl] = {
            "n": len(g),
            "strict_mrr@10": float(ps["mrr@10"].mean()),
            "lenient_mrr@10": float(pl["mrr@10"].mean()),
            "strict_hit@10": float(ps["hit@10"].mean()),
            "wrong_lang_top1": float(own.mean()),
        }
    (V2RES / f"{name}.{split}.mixed.json").write_text(json.dumps(out, indent=1))
    print(
        f"[{name}] {split} mixed-index: "
        + " | ".join(
            f"{k}: strict {v['strict_mrr@10']:.3f} lenient {v['lenient_mrr@10']:.3f} own-lang@1 {v['wrong_lang_top1']:.2f}"
            for k, v in out.items()
        ),
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prefix", default="e5", choices=["e5", "none"])
    ap.add_argument("--splits", nargs="+", default=["dev"])
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    qp, pp = ("query: ", "passage: ") if args.prefix == "e5" else ("", "")
    passages, queries = load_passages_v2(), load_queries_v2()
    d = E.Dense(args.model, args.name, qp, pp, max_len=512, bs=args.bs)
    evaluate(d, args.name, args.splits, queries, passages)
    if args.mixed:
        for s in args.splits:
            mixed_eval(d, args.name, s, queries, passages)


if __name__ == "__main__":
    main()
