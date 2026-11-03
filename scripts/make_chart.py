"""Draw docs/img/mrr_by_system*.png (light and dark variants) from the committed result files.

Test MRR@10 (x100) of the five systems that tell the story, in each of the four retrieval directions plus the macro average.
Every value is read from results/retrieval/*.json through the same aggregation code that builds the headline table, so the
chart can never disagree with RESULTS.md.   Run: python scripts/make_chart.py   (needs matplotlib; CPU only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from arretrieval.config import ROOT  # noqa: E402
from arretrieval.retrieval import report_lib as R  # noqa: E402

OUT = ROOT / "docs" / "img"
DIRECTIONS = ["ar->ar", "en->en", "ar->en", "en->ar", "macro"]
GROUP_LABELS = ["ar → ar", "en → en", "ar → en", "en → ar", "macro (mean of 4)"]

# (legend label, result names; several names = several seeds -> mean). Order = categorical slot order.
FT = R.seed_names("e5s_hard1_xl_e1", "test")
SYSTEMS = [
    ("multilingual-e5-small, off the shelf (118M)", ["e5-small"]),
    ("bge-m3, off the shelf (568M)", ["bge-m3"]),
    ("fine-tuned e5-small (118M; mean of 3 seeds)", FT),
    ("fine-tuned e5-small + fine-tuned reranker", ["e5s_hard1_xl_e1_s13__rr_ft_bce_s13"]),
    ("gated RRF(bge-m3, BM25) + fine-tuned reranker", ["hybrid-gated-bge-m3+bm25-stem__rr_ft_bce_s13"]),
]

# Categorical slots 1-5 of the validated reference palette (scripts/validate_palette.js: all checks pass in both modes).
THEMES = {
    "light": dict(
        surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e6e5e1", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    ),
    "dark": dict(
        surface="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="#34342f", series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
    ),
}

W_IN, H_IN, DPI = 9.0, 8.1, 200
PT = DPI / 72.0  # pixels per point
THICK, PITCH, RADIUS = 11.0, 13.0, 3.0  # bar thickness, slot pitch (the 2 pt difference is the surface gap), tip radius, in points


def values() -> np.ndarray:
    """rows = systems, columns = DIRECTIONS; MRR@10 x100."""
    out = np.zeros((len(SYSTEMS), len(DIRECTIONS)))
    for i, (_, names) in enumerate(SYSTEMS):
        for j, d in enumerate(DIRECTIONS):
            mean, _, n = R.agg(names, "test", d, "mrr@10")
            if n == 0:
                sys.exit(f"missing result for {names} / {d}")
            out[i, j] = mean * 100
    return out


def rounded_tip_bar(ax, y_center: float, x_end: float, color: str) -> None:
    """Horizontal bar from x=0: square at the baseline, rounded (radius RADIUS pt) at the data end. Built in display space."""
    t = THICK * PT
    r = RADIUS * PT
    to_disp = ax.transData.transform
    x0, yc = to_disp((0, y_center))
    x1, _ = to_disp((x_end, y_center))
    top, bot = yc + t / 2, yc - t / 2
    theta = np.linspace(0, np.pi / 2, 8)
    pts = [(x0, bot), (x1 - r, bot)]
    pts += [(x1 - r + r * np.sin(a), bot + r - r * np.cos(a)) for a in theta]  # bottom-right corner
    pts += [(x1 - r + r * np.cos(a), top - r + r * np.sin(a)) for a in theta]  # top-right corner
    pts += [(x1 - r, top), (x0, top), (x0, bot)]  # last vertex is the CLOSEPOLY placeholder
    data = ax.transData.inverted().transform(np.array(pts))
    ax.add_patch(PathPatch(MplPath(data, closed=True), facecolor=color, edgecolor="none", linewidth=0, zorder=3))


def draw(mode: str, vals: np.ndarray) -> Path:
    th = THEMES[mode]
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=th["surface"])
    n_sys, n_grp = vals.shape
    group_pitch = n_sys * PITCH + 16.0  # points per group incl. the air between groups
    total_pt = n_grp * group_pitch
    # axes exactly total_pt points tall, so one y data unit == one point on the page
    ax = fig.add_axes((0.215, 0.105, 0.745, total_pt / 72.0 / H_IN), facecolor=th["surface"])
    ax.set_xlim(0, 100)
    ax.set_ylim(total_pt, 0)  # y in points, top group first
    for x in range(20, 101, 20):
        ax.axvline(x, color=th["grid"], linewidth=0.6, zorder=1)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", colors=th["text2"], labelsize=8.5, length=0, pad=6)
    fig.canvas.draw()  # fix the transforms before building display-space shapes
    for g in range(n_grp):
        top_pt = g * group_pitch
        for s in range(n_sys):
            yc = top_pt + 8.0 + s * PITCH + PITCH / 2
            x_end = vals[s, g]
            rounded_tip_bar(ax, yc, x_end, th["series"][s])
            ax.text(x_end + 1.2, yc, f"{x_end:.1f}", va="center", ha="left", fontsize=7.8, color=th["text"], zorder=4)
        ax.text(
            -2.0,
            top_pt + 8.0 + n_sys * PITCH / 2,
            GROUP_LABELS[g],
            va="center",
            ha="right",
            fontsize=10,
            color=th["text"],
            fontweight="bold" if g == n_grp - 1 else "normal",
        )
    ax.set_xlabel("MRR@10 × 100 on the test split (higher is better)", color=th["text2"], fontsize=9, labelpad=8)
    # title block and legend: text tokens for words, series colour only on the swatch
    fig.text(
        0.035,
        0.965,
        "Arabic ↔ English retrieval: test MRR@10 by system and direction",
        color=th["text"],
        fontsize=13.5,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.035,
        0.932,
        "Query language → passage language, 6,823 test queries (TyDi-QA, ARCD, XQuAD). Macro = mean of the four directions.",
        color=th["text2"],
        fontsize=9,
        va="top",
    )
    handles = [Patch(facecolor=th["series"][i], edgecolor="none", label=SYSTEMS[i][0]) for i in range(n_sys)]
    leg = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.905),
        ncol=2,
        frameon=False,
        fontsize=8.6,
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=1.6,
        labelspacing=0.55,
    )
    for t in leg.get_texts():
        t.set_color(th["text"])
    fig.text(
        0.035,
        0.010,
        "Reranked rows: embedder seed 13, single reranker run; every reranker scores the first stage's top-50. Source: results/retrieval.",
        color=th["text2"],
        fontsize=7.4,
        va="bottom",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("mrr_by_system.png" if mode == "light" else "mrr_by_system_dark.png")
    fig.savefig(path, dpi=DPI, facecolor=th["surface"])
    plt.close(fig)
    return path


def main() -> None:
    vals = values()
    for mode in THEMES:
        print("wrote", draw(mode, vals))
    print(np.round(vals, 1))


if __name__ == "__main__":
    main()
