"""Retrieval-only metrics (no LLM): hit@k, MRR@k of the gold passage, per question kind. Deterministic, cheap.

python -m arretrieval.rageval.retrieval_eval
"""

from __future__ import annotations

import json

import numpy as np

from . import retrieval as R
from .paths import DATA, RUNS
from .stats import bootstrap_ci

CONFIGS = {
    "dense_k3": dict(collection="rag_eval_xquad", k=3, use_rerank=False),
    "dense_k5": dict(collection="rag_eval_xquad", k=5, use_rerank=False),
    "dense_k10": dict(collection="rag_eval_xquad", k=10, use_rerank=False),
    "rerank10_k3": dict(collection="rag_eval_xquad", k=3, use_rerank=True, fetch=10),
    "rerank20_k3": dict(collection="rag_eval_xquad", k=3, use_rerank=True, fetch=20),
    "chunks_dense_k3": dict(collection="rag_eval_xquad_chunks", k=3, use_rerank=False),
    "chunks_dense_k5": dict(collection="rag_eval_xquad_chunks", k=5, use_rerank=False),
    "chunks_rerank_k3": dict(collection="rag_eval_xquad_chunks", k=3, use_rerank=True, fetch=15),
}


def parent(h):
    return h.get("parent", h["id"])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    wanted = a.configs.split(",")
    prev = json.load(open(RUNS / "retrieval_eval.json")) if (RUNS / "retrieval_eval.json").exists() else {}
    qs = [json.loads(l) for l in open(DATA / "eval_questions.jsonl")]
    ans = [q for q in qs if q["answerable"]]
    out = dict(prev)
    for name, cfg in CONFIGS.items():
        if name not in wanted:
            continue
        ranks = []
        for q in ans:
            hits = R.retrieve(question=q["question"], **cfg)
            pids = []
            for h in hits:  # de-duplicate chunk parents, keep order
                if parent(h) not in pids:
                    pids.append(parent(h))
            ranks.append(pids.index(q["gold_id"]) + 1 if q["gold_id"] in pids else 0)
        ranks = np.array(ranks)
        row = {"hit": float((ranks > 0).mean()), "mrr": float(np.where(ranks > 0, 1 / np.maximum(ranks, 1), 0).mean())}
        row["hit_ci"] = bootstrap_ci((ranks > 0).astype(float), n_boot=2000)[1:]
        for kind in ("mono", "cross"):
            m = np.array([q["kind"] == kind for q in ans])
            row[f"hit_{kind}"] = float((ranks[m] > 0).mean())
        for ql in ("en", "ar"):
            m = np.array([q["q_lang"] == ql for q in ans])
            row[f"hit_q{ql}"] = float((ranks[m] > 0).mean())
        out[name] = row
        print(
            f"{name:18s} hit={row['hit']:.3f} [{row['hit_ci'][0]:.2f},{row['hit_ci'][1]:.2f}] mrr={row['mrr']:.3f} "
            f"mono={row['hit_mono']:.3f} cross={row['hit_cross']:.3f} q_en={row['hit_qen']:.3f} q_ar={row['hit_qar']:.3f}",
            flush=True,
        )
        json.dump(out, open(RUNS / "retrieval_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
