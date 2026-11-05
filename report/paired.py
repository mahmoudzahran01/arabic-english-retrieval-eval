"""Paired statistics over MTEB(ara) tasks: how much does a score difference between two models mean with only a few tasks?

Input: `results/retrieval_v2/mteb_paired.json`, written by the v2 pipeline (runs/retrieval/v2/results/mteb_paired.json). Observed schema (one entry per comparison,
keyed by a label, a list of such entries is accepted as well):

    {"final_m3_wa75_s13 vs BAAI__bge-m3": {
        "new": "final_m3_wa75_s13", "ref": "BAAI__bge-m3", "n_tasks": 19,
        "mean_new": 72.04, "mean_ref": 71.98, "mean_diff": 0.06, "ci95_task_bootstrap": [-1.14, 1.16], "boot_p_diff_gt_0": 0.56,
        "wins": 9, "losses": 9, "ties": 1, "sign_test_p_two_sided": 1.0,
        "per_task": [{"task": "XQuADRetrieval", "new": 95.4, "ref": 94.4, "diff": 0.95}, ...],
        "subsets": {"ar-en cross-lingual subsets": {"per_task": [...same shape...]}}     # optional; also read from "cross_lingual" / "xling"
    }}

The statistics are recomputed here from `per_task` with the lab's own procedure (random.Random(13), 10,000 resamples of the tasks, sorted means, percentile indices
int(0.025 N) and int(0.975 N) - 1), so the recomputed interval of every comparison in the file equals the file's `ci95_task_bootstrap` exactly (checked in the claims) and the
same procedure also serves the subset rows and the leaderboard pairs, which the file does not carry.
`per_task` entries may also be `{task: [new, ref]}` or `{task: {"new": .., "ref": ..}}`. Scores are on the 0-100 scale.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_BOOT = 10_000
SEED = 13

# MTEB(ara, v1) task types (the column groups of results/mteb_ara/leaderboard.json)
TASK_TYPES = {
    "Retrieval": [
        "MIRACLRetrievalHardNegatives.v2",
        "XQuADRetrieval",
        "MLQARetrieval",
        "BelebeleRetrieval",
        "XPQARetrieval",
        "WebFAQRetrieval",
    ],
    "Reranking": ["NamaaMrTydiReranking"],
    "STS": ["STS17", "STS22.v2", "SemRel24STS"],
    "PairClassification": ["XNLI", "ArEntail"],
    "Classification": [
        "AJGT.v2",
        "TweetEmotionClassification.v2",
        "TweetSarcasmClassification.v2",
        "HotelReviewSentimentClassification.v2",
    ],
    "Clustering": ["SIB200ClusteringS2S.v2"],
    "BitextMining": ["FloresBitextMining", "Tatoeba"],
}


@dataclass
class Comparison:
    label: str
    new: str
    ref: str
    scores: dict[str, tuple[float, float]]  # task -> (new, ref)
    source: str = "file"
    subsets: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    precomputed: dict = field(default_factory=dict)


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test (ties are dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_stats(scores: dict[str, tuple[float, float]], n_boot: int = N_BOOT, seed: int = SEED, tie_eps: float = 0.005) -> dict:
    """Mean difference with a percentile bootstrap CI over TASKS, wins / losses / ties (a tie is |difference| < tie_eps = 0.005 points, as in the lab) and the sign test for {task: (new, ref)}.

    The bootstrap is the lab's (runs/retrieval/v2 mteb_paired.py): random.Random(seed), n_boot resamples of the tasks with replacement, sorted resample means, interval
    [boots[int(0.025 n_boot)], boots[int(0.975 n_boot) - 1]].
    """
    d = [n - r for n, r in scores.values()]
    if not d:
        return {"n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    wins, losses = sum(x > tie_eps for x in d), sum(x < -tie_eps for x in d)
    return {
        "n": len(d),
        "mean_new": sum(n for n, _ in scores.values()) / len(d),
        "mean_ref": sum(r for _, r in scores.values()) / len(d),
        "mean_diff": sum(d) / len(d),
        "ci_lo": boots[int(0.025 * n_boot)],
        "ci_hi": boots[int(0.975 * n_boot) - 1],
        "wins": wins,
        "losses": losses,
        "ties": len(d) - wins - losses,
        "sign_p": sign_test_p(wins, losses),
    }


def _per_task(raw) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if isinstance(raw, list):
        for e in raw:
            if isinstance(e, dict) and "task" in e and "new" in e and "ref" in e:
                out[e["task"]] = (float(e["new"]), float(e["ref"]))
    elif isinstance(raw, dict):
        for t, v in raw.items():
            if isinstance(v, list | tuple) and len(v) == 2:
                out[t] = (float(v[0]), float(v[1]))
            elif isinstance(v, dict) and "new" in v and "ref" in v:
                out[t] = (float(v["new"]), float(v["ref"]))
    return out


def parse(raw) -> list[Comparison]:
    entries = (
        list(raw.items())
        if isinstance(raw, dict)
        else [(e.get("label", f"comparison {i}"), e) for i, e in enumerate(raw)]
        if isinstance(raw, list)
        else []
    )
    out = []
    for label, e in entries:
        if not isinstance(e, dict):
            continue
        scores = _per_task(e.get("per_task"))
        subsets = {}
        for key in ("subsets", "cross_lingual", "xling"):
            sub = e.get(key)
            if isinstance(sub, dict):
                if key == "subsets":
                    subsets.update({n: _per_task(v.get("per_task", v) if isinstance(v, dict) else v) for n, v in sub.items()})
                else:
                    subsets["ar-en cross-lingual subsets"] = _per_task(sub.get("per_task", sub))
        pre = {k: e[k] for k in ("mean_diff", "ci95_task_bootstrap", "wins", "losses", "ties", "sign_test_p_two_sided") if k in e}
        out.append(
            Comparison(label, str(e.get("new", "")), str(e.get("ref", "")), scores, "file", {k: v for k, v in subsets.items() if v}, pre)
        )
    return out


def load(path: Path) -> list[Comparison]:
    if not path.exists():
        return []
    try:
        return parse(json.loads(path.read_text()))
    except (OSError, ValueError):
        return []


def from_leaderboard(lb: dict, pairs: list[tuple[str, str, str]]) -> list[Comparison]:
    """Comparisons between models of results/mteb_ara/leaderboard.json: pairs = [(label, new model_name suffix, ref model_name suffix)]."""
    by = {r["model_name"].split("/")[-1]: r for r in lb.get("per_task", [])}
    tasks = [t for ts in TASK_TYPES.values() for t in ts]
    out = []
    for label, new, ref in pairs:
        if new in by and ref in by:
            out.append(
                Comparison(
                    label,
                    new,
                    ref,
                    {t: (float(by[new][t]), float(by[ref][t])) for t in tasks if t in by[new] and t in by[ref]},
                    "leaderboard",
                )
            )
    return out


def type_means(scores: dict[str, float]) -> dict[str, float | None]:
    """Mean per task type, Mean (Task) and Mean (TaskType) of a {task: score} dict over MTEB(ara, v1); None where a task is missing."""
    out: dict[str, float | None] = {}
    for typ, tasks in TASK_TYPES.items():
        out[typ] = float(np.mean([scores[t] for t in tasks])) if all(t in scores for t in tasks) else None
    all_tasks = [t for ts in TASK_TYPES.values() for t in ts]
    out["Mean (Task)"] = float(np.mean([scores[t] for t in all_tasks])) if all(t in scores for t in all_tasks) else None
    types = [v for k, v in out.items() if k in TASK_TYPES]
    out["Mean (TaskType)"] = float(np.mean(types)) if all(v is not None for v in types) else None
    return out


def load_subsets(path: Path) -> dict[str, dict[str, dict[str, float | None]]]:
    """results/retrieval_v2/mteb_subsets.json written by report/extract_mteb_subsets.py: {model folder: {task: {all, cross, mono}}}."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def subset_comparison(subsets: dict, new: str, ref: str, kind: str) -> dict[str, tuple[float, float]]:
    """{task: (new, ref)} of the per-task mean over the `kind` subsets ('cross' = Arabic<->English, 'mono' = Arabic-only) for tasks that have it in both models."""
    a, b = subsets.get(new, {}), subsets.get(ref, {})
    return {t: (a[t][kind], b[t][kind]) for t in a if t in b and a[t].get(kind) is not None and b[t].get(kind) is not None}


def subset_mean(subsets: dict, model: str, kind: str) -> float | None:
    """Leaderboard-style mean over tasks of the per-task `kind` score (None if the model or the subsets are missing)."""
    vals = [v[kind] for v in subsets.get(model, {}).values() if v.get(kind) is not None]
    return float(np.mean(vals)) if vals else None
