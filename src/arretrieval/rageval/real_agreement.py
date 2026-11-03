"""Gemma-vs-Qwen agreement on REAL RAG answers (no planted labels): do the cheap judge and the strong judge agree?

python -m arretrieval.rageval.real_agreement --configs A_baseline,I_rerank_lang --judge correct_v1
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .paths import RUNS
from .stats import cohen_kappa, confusion


def load(cfg, model, judge):
    p = RUNS / f"judge_rag_{cfg}_{model}" / f"{judge}.jsonl"
    return {r["id"]: r["pass"] for r in (json.loads(l) for l in open(p))} if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="A_baseline,I_rerank_lang")
    ap.add_argument("--judge", default="correct_v1")
    a = ap.parse_args()
    x, y = [], []
    for c in a.configs.split(","):
        g, q = load(c, "gemma", a.judge), load(c, "qwen", a.judge)
        for i in g:
            if i in q and g[i] is not None and q[i] is not None:
                x.append(g[i])
                y.append(q[i])
    cm = confusion(x, y)
    print(f"{a.judge}: n={len(x)} raw agreement={np.mean(np.array(x) == np.array(y)):.3f} kappa(gemma,qwen)={cohen_kappa(x, y):.3f}")
    print(f"confusion rows=gemma[fail,pass] cols=qwen[fail,pass]: {cm.tolist()}")


if __name__ == "__main__":
    main()
