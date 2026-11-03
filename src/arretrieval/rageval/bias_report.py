"""Markdown table of the pairwise-judge bias checks (reads runs/rageval/bias_{gemma,qwen}.json).

  python -m arretrieval.rageval.bias_report
Rates are over individual verdicts (each pair is judged in both orders, so verdicts are paired, not independent: the
Wilson intervals shown are therefore optimistic).
"""

from __future__ import annotations

import json

from .paths import RUNS
from .stats import wilson

ROWS = [
    ("quality", "correct vs wrong-number answer", "x = correct; should be 100%"),
    ("equal", "ASCII-digit answer vs same answer in Arabic-Indic digits (both correct)", "x = ASCII version; 50% if script-invariant"),
    (
        "pad_equal",
        "short correct vs same + content-free fluff (both correct)",
        "x = short; 50% if no verbosity effect, <50% = prefers long",
    ),
    ("pad_wrong", "short correct vs wrong-number answer + fluff", "x = short correct; <100% = padding rescued a wrong answer"),
    ("self", "Gemma-written vs Qwen-written correct answers", "x = Gemma-written; <50% = prefers Qwen-written"),
]


def cell(d):
    n = d["x_overall"]["n"]
    k = round(d["x_overall"]["rate"] * n)
    lo, hi = wilson(k, n)
    return f"{100 * d['x_overall']['rate']:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] (n={n})"


def main():
    res = {m: json.load(open(RUNS / f"bias_{m}.json")) for m in ("gemma", "qwen") if (RUNS / f"bias_{m}.json").exists()}
    L = [
        "| suite | what is compared | reading | "
        + " | ".join(f"{m}: P(x chosen) | {m}: picks slot A | {m}: order-consistent" for m in res)
        + " |",
        "|---|---|---|" + "---|---|---|" * len(res),
    ]
    for key, what, reading in ROWS:
        cells = []
        for m, d in res.items():
            s = d[key]
            cells += [cell(s), f"{100 * s['picks_position_A']['rate']:.0f}%", f"{100 * s['order_consistent']['rate']:.0f}%"]
        L.append(f"| {key} | {what} | {reading} | " + " | ".join(cells) + " |")
    for m, d in res.items():
        L.append(
            f"\n{m}: self-preference suite used {d['self']['n_pairs']} pairs; Qwen-written / Gemma-written length ratio = {d['self']['qwen_over_gemma_length_ratio']:.2f}."
        )
    print("\n".join(L))


if __name__ == "__main__":
    main()
