"""Helpers that turn the stored result json files into markdown tables (used by make_report.py).

Results are read from `paths.REPORT_RESULTS` (the committed `results/retrieval` snapshot unless `ARR_RESULTS_DIR` says otherwise).
"""

from __future__ import annotations

import json

import numpy as np

from . import evalkit as E
from .paths import REPORT_CARD, REPORT_RESULTS

RESULTS = REPORT_RESULTS

DIRS = E.DIRECTIONS


def load(name: str, split: str) -> dict | None:
    p = E.result_path(name, split, "json", RESULTS)
    return json.loads(p.read_text()) if p.exists() else None


def direction_value(res: dict, d: str, metric: str) -> float:
    return res["directions"][d][metric] if d in res["directions"] else float("nan")


def seed_names(prefix: str, split: str) -> list[str]:
    return sorted(
        p.name[: -len(f".{split}.json")]
        for p in RESULTS.glob(f"{prefix}_s*.{split}.json")
        if p.name[len(prefix) + 2 :].split(".")[0].isdigit()
    )


def _value(res: dict, d: str, metric: str) -> float:
    """d may be a direction, 'macro', or 'macro:ar->ar,en->en' (macro over the listed directions only)."""
    if d.startswith("macro:"):
        parts = [direction_value(res, x, metric) for x in d[6:].split(",")]
        return float(np.mean(parts)) if not any(np.isnan(parts)) else float("nan")
    return direction_value(res, d, metric)


def agg(names: list[str], split: str, d: str, metric: str) -> tuple[float, float, int]:
    vals = [_value(load(n, split), d, metric) for n in names if load(n, split)]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"), len(vals)


def cell(mean: float, std: float, n: int, pct: bool = True) -> str:
    if np.isnan(mean):
        return "n/a"
    scale = 100 if pct else 1
    s = f"{mean * scale:.1f}" if pct else f"{mean:.3f}"
    return f"{s} ± {std * scale:.1f}" if n > 1 and not np.isnan(std) else s


def table(
    rows: list[tuple[str, list[str]]],
    split: str,
    metric: str = "mrr@10",
    dirs=DIRS,
    macro: bool = True,
    nseeds_col: bool = False,
    macro_col: str = "macro",
) -> str:
    """rows: (label, [result names]); several names = several seeds -> mean ± std. Values x100.
    macro_col='macro:ar->ar,en->en' gives a macro over those directions only (used for DEV, where early runs lack cross-lingual slices)."""
    cols = list(dirs) + ([macro_col] if macro else [])
    head = (
        "| System | " + " | ".join(c if not c.startswith("macro:") else "mono macro" for c in cols) + (" | seeds |" if nseeds_col else " |")
    )
    sep = "|---|" + "---:|" * len(cols) + ("---:|" if nseeds_col else "")
    lines = [head, sep]
    for label, names in rows:
        cells = [cell(*agg(names, split, d, metric)) for d in cols]
        n = max((agg(names, split, d, metric)[2] for d in cols), default=0)
        lines.append(f"| {label} | " + " | ".join(cells) + (f" | {n} |" if nseeds_col else " |"))
    return "\n".join(lines)


def slice_table(names: list[tuple[str, str]], split: str, metrics=("hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")) -> str:
    """One row per (system, slice) with all five metrics (x100)."""
    lines = ["| System | Slice | n | " + " | ".join(m for m in metrics) + " |", "|---|---|---:|" + "---:|" * len(metrics)]
    for label, name in names:
        res = load(name, split)
        if not res:
            continue
        for sl, s in res["slices"].items():
            lines.append(f"| {label} | {sl} | {s['n']} | " + " | ".join(f"{s[m] * 100:.1f}" for m in metrics) + " |")
    return "\n".join(lines)


def boot_direction_delta(a: str, b: str, split: str, direction: str, metric: str = "mrr@10", n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap of the direction-macro difference (b - a): queries are resampled within each slice, the same
    indices for both systems, then slice means are averaged exactly like the reported direction number."""
    pa, pb = E.load_perq(a, split, base=RESULTS), E.load_perq(b, split, base=RESULTS)
    sls = [s for s in pa if E.DIRECTION_OF[s] == direction and s in pb]
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_boot)
    for s in sls:
        x, y = pa[s][metric], pb[s][metric]
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        diffs += (y[idx].mean(axis=1) - x[idx].mean(axis=1)) / len(sls)
    point = float(np.mean([pb[s][metric].mean() - pa[s][metric].mean() for s in sls]))
    return {"delta": point, "ci_lo": float(np.quantile(diffs, 0.025)), "ci_hi": float(np.quantile(diffs, 0.975))}


def delta_table(pairs: list[tuple[str, str, str]], split: str = "test", metric: str = "mrr@10") -> str:
    """pairs: (label, baseline name, system name) -> per-direction delta [95% CI] in points."""
    lines = ["| Comparison | " + " | ".join(DIRS) + " |", "|---|" + "---:|" * len(DIRS)]
    for label, a, b in pairs:
        cells = []
        for d in DIRS:
            try:
                r = boot_direction_delta(a, b, split, d, metric)
                sig = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else ""
                cells.append(f"{r['delta'] * 100:+.1f} [{r['ci_lo'] * 100:+.1f}, {r['ci_hi'] * 100:+.1f}]{sig}")
            except Exception:
                cells.append("n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def contamination_sensitivity(names: list[tuple[str, str]], metric: str = "mrr@10") -> str:
    """Test MRR with all queries vs. excluding queries whose gold passage has >=50% 6-gram overlap with a TRAIN passage."""
    import pandas as pd

    flagged = set(json.loads((REPORT_CARD / "test_pids_near_train.json").read_text()))
    q = pd.read_csv(REPORT_CARD / "test_query_index.csv")  # qid, slice, pos_pid (ids only, no dataset text)
    bad = set(q[q.pos_pid.isin(flagged)].qid)
    lines = [
        f"Excluded test queries (gold passage near-duplicated in train): {len(bad)} of {len(q)}.",
        "",
        "| System | macro MRR@10, all queries | macro MRR@10, excluding flagged | difference |",
        "|---|---:|---:|---:|",
    ]
    for label, name in names:
        pq = E.load_perq(name, "test", base=RESULTS)
        per_dir_all, per_dir_ex = {}, {}
        for sl, d in pq.items():
            qids = sorted(q[q.slice == sl].qid)
            keep = np.array([x not in bad for x in qids])
            per_dir_all.setdefault(E.DIRECTION_OF[sl], []).append(d[metric].mean())
            per_dir_ex.setdefault(E.DIRECTION_OF[sl], []).append(d[metric][keep].mean())
        a = np.mean([np.mean(v) for v in per_dir_all.values()])
        b = np.mean([np.mean(v) for v in per_dir_ex.values()])
        lines.append(f"| {label} | {a * 100:.2f} | {b * 100:.2f} | {(b - a) * 100:+.2f} |")
    return "\n".join(lines)
