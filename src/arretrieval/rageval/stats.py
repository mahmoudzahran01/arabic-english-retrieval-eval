"""Agreement and uncertainty statistics. Pure numpy; hand-checked in tests/rageval/test_stats.py."""

from __future__ import annotations

from math import comb
from typing import Callable, Sequence

import numpy as np


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """2x2 matrix [[TN, FP], [FN, TP]]; rows = truth (0/1), cols = prediction (0/1)."""
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    m = np.zeros((2, 2), dtype=int)
    for a in (0, 1):
        for b in (0, 1):
            m[a, b] = int(((t == a) & (p == b)).sum())
    return m


def cohen_kappa(y1: Sequence, y2: Sequence) -> float:
    """Cohen's kappa for two raters over any categorical labels. Returns nan if undefined."""
    a, b = np.asarray(y1), np.asarray(y2)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    if abs(1 - pe) < 1e-12:
        return float("nan") if po < 1 else 1.0
    return (po - pe) / (1 - pe)


def prf(y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1) -> dict:
    """Precision/recall/F1 for the chosen positive class."""
    t = np.asarray(y_true) == positive
    p = np.asarray(y_pred) == positive
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and not np.isnan(prec + rec) else (0.0 if tp + fp + fn else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "support": int(t.sum())}


def raw_agreement(y1: Sequence, y2: Sequence) -> float:
    return float((np.asarray(y1) == np.asarray(y2)).mean())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (better than normal approx at small n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    clusters: Sequence | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over items (or over clusters if given). Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        boots = np.array([stat(v[i]) for i in idx])
    else:
        c = np.asarray(clusters)
        ids = sorted(set(c.tolist()), key=str)
        groups = [np.where(c == g)[0] for g in ids]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(groups), size=len(groups))
            boots[b] = stat(v[np.concatenate([groups[i] for i in pick])])
    return (float(stat(v)), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2))))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap the mean paired difference (b - a) over questions. Returns diff, CI, two-sided p."""
    x = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value: how often the resampled mean falls on the other side of 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(x.mean()), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p)), "n": len(x)}


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact McNemar test on paired binary outcomes (b01 = a fails & b passes; b10 = reverse)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2**n)
    return {"n01": n01, "n10": n10, "p": float(p)}


def min_detectable_effect(n: int, p: float = 0.7, rho: float = 0.5, z_a: float = 1.96, z_b: float = 0.84) -> float:
    """Rough minimum detectable paired difference in a proportion at 80% power.

    Uses var(diff) = 2*p*(1-p)*(1-rho) where rho is the within-question correlation of the two systems.
    """
    return float((z_a + z_b) * np.sqrt(2 * p * (1 - p) * (1 - rho) / n))
