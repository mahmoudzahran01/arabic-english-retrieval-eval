"""Paths, constants and formatting helpers shared by the report scripts (make_tables.py, study_tables.py)."""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "retrieval_v2"
MTEB = ROOT / "results" / "mteb_ara"
DEPLOY = ROOT / "deploy" / "results"
DIRS = ("ar->ar", "en->en", "ar->en", "en->ar")
DIR_TEX = {"ar->ar": r"ar$\to$ar", "en->en": r"en$\to$en", "ar->en": r"ar$\to$en", "en->ar": r"en$\to$ar", "macro": "macro"}


def esc(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s.replace("->", r"$\to$")


def round_half_up(v: float, nd: int) -> str:
    """Decimal text of v with nd decimals, rounding half away from zero on the shortest decimal form of v (0.4275 -> 0.43, -0.425 -> -0.43)."""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pct(x: float, nd: int = 1) -> str:
    return round_half_up(100 * x, nd)


def _signed(v: float, nd: int) -> str:
    txt = round_half_up(v, nd)
    if float(txt) == 0.0:  # never print "-0.0" / "+0.0" for a rounded zero
        return f"{0:.{nd}f}"
    return txt.replace("-", "$-$") if txt.startswith("-") else "+" + txt


def spct(x: float, nd: int = 1) -> str:
    return _signed(100 * x, nd)


def sfmt(x: float, nd: int = 1) -> str:
    """Signed number already on a 0-100 scale."""
    return _signed(x, nd)


def todo(msg: str) -> str:
    return rf"\TODO{{{msg.replace('_', chr(92) + '_')}}}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def load_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def load_system(stem: str, split: str = "test") -> dict | None:
    j = load_json(V2 / f"{stem}.{split}.json")
    if j is None:
        return None
    z = V2 / f"{stem}.{split}.perq.npz"
    perq = {}
    if z.exists():
        zz = np.load(z)
        perq = {k.split("|")[0]: zz[k].astype(np.float64) for k in zz.files if k.endswith("|mrr@10")}
    return {"json": j, "perq": perq}


def dir_value(sysd: dict, d: str) -> float:
    return sysd["json"]["directions"][d]["mrr@10"]


def load_board() -> dict[str, dict]:
    """MTEB(ara) per-task scores of the scored models (results/retrieval_v2/mteb_board.json) keyed by model name."""
    return {r["model"]: r for r in (load_json(V2 / "mteb_board.json") or [])}


def dev_value(stem: str, d: str) -> float | None:
    j = load_json(V2 / f"{stem}.dev.json")
    return j["directions"][d]["mrr@10"] if j and d in j.get("directions", {}) else None


def devm_value(stem: str) -> float | None:
    j = load_json(V2 / f"{stem}.devM.json")
    return j["slices"]["dev_miracl_ar"]["mrr@10"] if j and "dev_miracl_ar" in j.get("slices", {}) else None


LATEX_UNICODE = {
    "→": r"$\to$", "≥": r"$\ge$", "≤": r"$\le$", "×": r"$\times$", "–": "--", "—": "---", "‘": "'", "’": "'",
    "“": "``", "”": "''", "≈": r"$\approx$", "±": r"$\pm$", "α": r"$\alpha$", "…": "...",
}  # fmt: skip


def latex_text(s: str) -> str:
    """Escape free text (lab-notebook quotes) for LaTeX; common unicode punctuation is mapped, anything else outside ASCII becomes '?'."""
    out = []
    for ch in s:
        if ch in LATEX_UNICODE:
            out.append(LATEX_UNICODE[ch])
        elif ord(ch) > 126:
            out.append("?")
        elif ch in "&%#_$":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out).replace("->", r"$\to$")
