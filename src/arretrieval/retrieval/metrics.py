"""Ranking metrics with binary relevance, plus a paired bootstrap for system comparisons.

All functions take a ranked list of passage ids (best first) and the set of relevant ids.
Definitions (also derived in EXPLAINER.md):
  Hit@k    = 1 if any relevant id is in the top k, else 0
  MRR@k    = 1 / rank of the first relevant id if rank <= k, else 0
  nDCG@k   = DCG@k / IDCG@k,  DCG@k = sum_{i<=k} rel_i / log2(i + 1)
             IDCG@k is the DCG of the ideal ranking (all relevant ids first)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

K_HITS = (1, 5, 10)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(p in relevant for p in ranked[:k]))


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for i, p in enumerate(ranked[:k], start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1) if p in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def per_query_metrics(ranked: Sequence[str], relevant: set[str], extra_hits: Iterable[int] = ()) -> dict[str, float]:
    out = {f"hit@{k}": hit_at_k(ranked, relevant, k) for k in sorted(set(K_HITS) | set(extra_hits))}
    out["mrr@10"] = mrr_at_k(ranked, relevant, 10)
    out["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    return out


def evaluate_run(run: dict[str, Sequence[str]], qrels: dict[str, set[str]], extra_hits: Iterable[int] = ()) -> dict[str, np.ndarray]:
    """Per-query metric arrays (aligned to sorted(qrels)); a missing query counts as an empty ranking."""
    qids = sorted(qrels)
    rows = [per_query_metrics(run.get(q, []), qrels[q], extra_hits) for q in qids]
    return {m: np.array([r[m] for r in rows], dtype=np.float64) for m in rows[0]} if rows else {}


def summarize(per_query: dict[str, np.ndarray]) -> dict[str, float]:
    out = {m: float(v.mean()) for m, v in per_query.items()}
    out["n"] = int(len(next(iter(per_query.values())))) if per_query else 0
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap over queries for mean(b) - mean(a). Returns delta, 95% CI and P(delta <= 0)."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "delta": float(d.mean()),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "p_le_0": float((boots <= 0).mean()),
    }


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, top: int = 100) -> list[str]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank_in_list(d)), rank starts at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + r)
    return [p for p, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:top]
