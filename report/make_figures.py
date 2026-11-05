"""Figures of the report, drawn from the committed result files.

    python report/make_figures.py          # writes report/figures/alpha_curve.{pdf,png} if results/retrieval_v2/alpha_curve.json exists

WiSE trade-off curve (figure `alpha_curve`)
-------------------------------------------
Input: `results/retrieval_v2/alpha_curve.json`, copied from `runs/retrieval/v2/results/alpha_curve.json` of the v2 pipeline by `report/pull_results.sh`.
The loader is deliberately tolerant. It expects (and the v2 agent is asked to write):

    {
      "selected_alpha": 0.75,                                   # also accepted: meta.selected_alpha
      "points": [                                               # also accepted: "curve" / "alphas" / "results", or a dict keyed by the alpha
        {"alpha": 0.0,                                          # fine-tuned share: 0 = bge-m3 itself, 1 = the raw fine-tune (aliases: a, w, weight)
         "test":   {"ar->en": 0.63, "macro": 0.75},             # TEST MRR@10, fractions or percentages (values <= 1.5 are read as fractions)
         "guards": {"MIRACLRetrievalHardNegatives.v2": 0.79,    # MTEB main scores of the guard tasks; any key that starts with miracl / namaa / semrel
                    "NamaaMrTydiReranking": 0.79, "SemRel24STS": 0.59}},
        ...
      ]
    }

Flat points are accepted too (`{"alpha": .., "test_ar2en": .., "test_macro": .., "miracl": .., "namaa": ..}`). Series with no data are skipped and named in the
panel text; if no point can be read at all nothing is drawn and the report shows a red TODO box. The curve is POST HOC: the alpha was selected before it was
computed (see the pre-registration appendix), and the figure says so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURVE_JSON = ROOT / "results" / "retrieval_v2" / "alpha_curve.json"
FIG_DIR = ROOT / "report" / "figures"

POINT_KEYS = ("points", "curve", "alphas", "results", "data")
ALPHA_KEYS = ("alpha", "a", "w", "weight", "ft_share")
SERIES = {  # canonical name -> (label, accepted normalised key prefixes)
    "ar->en": ("TEST ar$\\to$en", ("aren", "ar2en", "testaren", "testar2en", "arabictoenglish")),
    "macro": ("TEST macro", ("macro", "testmacro")),
    "miracl": ("MIRACL-HN", ("miracl",)),
    "namaa": ("Mr.TyDi rerank", ("namaa", "mrtydi")),
    "semrel": ("SemRel24", ("semrel",)),
}


@dataclass
class Point:
    alpha: float
    values: dict[str, float] = field(default_factory=dict)  # canonical series name -> value on a 0-100 scale


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower().replace("->", "").replace("→", ""))


def _canonical(key: str) -> str | None:
    k = _norm(key)
    for name, (_, prefixes) in SERIES.items():
        if any(k.startswith(p) for p in prefixes):
            return name
    if key.lower().replace(" ", "") in ("ar->en", "ar_en", "ar2en"):
        return "ar->en"
    return None


def _flatten(obj: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out.update(_flatten(v))
        elif isinstance(v, int | float) and not isinstance(v, bool):
            name = _canonical(k)
            if name is not None:
                out[name] = float(v)
    return out


def _alpha_of(p: dict, default: float | None = None) -> float | None:
    for k in ALPHA_KEYS:
        if k in p and isinstance(p[k], int | float):
            return float(p[k])
    return default


def load_curve(path: Path = CURVE_JSON) -> tuple[list[Point], float | None]:
    """Read the curve tolerantly. Returns (points sorted by alpha, selected alpha or None); an unreadable file gives ([], None)."""
    if not path.exists():
        return [], None
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return [], None
    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    selected = None
    if isinstance(raw, dict):
        for src in (raw, meta):
            for k in ("selected_alpha", "selected", "pick_alpha"):
                if isinstance(src.get(k), int | float):
                    selected = float(src[k])
    entries: list[tuple[float | None, dict]] = []
    body = raw
    if isinstance(raw, dict):
        body = next((raw[k] for k in POINT_KEYS if k in raw), raw)
    if isinstance(body, list):
        entries = [(_alpha_of(p), p) for p in body if isinstance(p, dict)]
    elif isinstance(body, dict):
        for k, p in body.items():
            try:
                a = float(k)
            except ValueError:
                continue
            if isinstance(p, dict):
                entries.append((_alpha_of(p, a), p))
    pts = []
    for a, p in entries:
        if a is None:
            continue
        vals = _flatten({k: v for k, v in p.items() if k not in ALPHA_KEYS})
        if vals:
            pts.append(Point(a, vals))
    if pts and all(abs(v) <= 1.5 for p in pts for v in p.values.values()):
        for p in pts:
            p.values = {k: 100 * v for k, v in p.values.items()}
    return sorted(pts, key=lambda p: p.alpha), selected


def series(points: list[Point], name: str) -> tuple[list[float], list[float]]:
    xs = [p.alpha for p in points if name in p.values]
    return xs, [p.values[name] for p in points if name in p.values]


def plot(points: list[Point], selected: float | None, out_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bot) = plt.subplots(2, 1, figsize=(3.3, 4.3), sharex=True, constrained_layout=True)
    style = {
        "ar->en": ("#0b6e4f", "o", "-"),
        "macro": ("#1f4e9c", "s", "-"),
        "miracl": ("#b8860b", "o", "-"),
        "namaa": ("#8e2c8e", "s", "-"),
        "semrel": ("#777777", "^", ":"),
    }
    missing = []
    for panel, names in ((top, ("ar->en", "macro")), (bot, ("miracl", "namaa", "semrel"))):
        for n in names:
            xs, ys = series(points, n)
            if not xs:
                missing.append(SERIES[n][0])
                continue
            c, m, ls = style[n]
            panel.plot(xs, ys, color=c, marker=m, ls=ls, ms=3.5, lw=1.2, label=SERIES[n][0])
            ref = [(x, y) for x, y in zip(xs, ys, strict=True) if x == 0]
            if ref:  # alpha = 0 is bge-m3 itself
                panel.scatter([ref[0][0]], [ref[0][1]], s=40, facecolors="none", edgecolors="k", linewidths=0.9, zorder=5)
        if selected is not None:
            panel.axvline(selected, color="k", ls="--", lw=0.8)
        panel.grid(alpha=0.25, lw=0.5)
        if panel.get_legend_handles_labels()[0]:
            panel.legend(fontsize=6.5, frameon=False, loc="best")
        panel.tick_params(labelsize=7)
    top.set_ylabel("MRR@10 x100 (TEST)", fontsize=7.5)
    bot.set_ylabel("guard score x100", fontsize=7.5)
    bot.set_xlabel("fine-tuned share alpha (0 = bge-m3, 1 = raw fine-tune)", fontsize=7.5)
    note = "post hoc curve"
    if selected is not None:
        note += f"; dashed: selected alpha {selected:g}"
    top.set_title(note + "; circle: bge-m3", fontsize=6.5, loc="left")
    if missing:
        bot.text(0.02, 0.04, "no data for: " + ", ".join(missing), transform=bot.transAxes, fontsize=5.5, color="red")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


def main() -> None:
    points, selected = load_curve()
    if not points:
        print(
            f"no readable WiSE curve at {CURVE_JSON} (expected schema: see this file's docstring); no figure written, the report shows a TODO box"
        )
        return
    for p in plot(points, selected, FIG_DIR / "alpha_curve"):
        print("wrote", p)


if __name__ == "__main__":
    main()
