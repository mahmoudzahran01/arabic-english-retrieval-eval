"""Assemble the PRE-REGISTERED validation suite (PROGRESS.md 2026-11-04 ~19:05) and apply its selection rule.  CPU only.  No MTEB(ara) and no TEST number is read.

components (x100): dev macro, devM, mMARCO-ar proxy, XNLI-ar(train) AP, cls (ArSarcasm sarcasm+sentiment probes), STS-ar(dev) Spearman
normalisation  : n_c = (x - min_pool) / max(max_pool - min_pool, 1.0), pool = every model in the table (candidates + reference rows)
score          : unweighted mean of the six n_c
eligibility    : dev ar->en >= bge-m3 dev ar->en + 5.0
pick           : highest score among eligible, selectable models; |diff| STRICTLY < 0.02 -> fewer ingredients (single-seed WiSE < soup WiSE < LoRA/freeze), then larger alpha; the exact gap is printed
Run: NO_GPU=1 python -m arretrieval.retrieval.v2.suite_select [--write --require A,B,..]        (--write stores results/suite_pick.json once every required model is present)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import report_v2 as R
from .common_v2 import V2RES

FLOOR, MARGIN_AR_EN, TIE = 1.0, 5.0, 0.02
COMPONENTS = ["dev", "devM", "proxy", "xnli", "cls", "sts"]

# name -> (family, alpha, selectable)        families order = "ingredients": 0 single-seed WiSE, 1 soup WiSE, 2 LoRA / freeze
MODELS: dict[str, tuple[str, float, bool]] = {
    "bge-m3": ("reference", 0.0, False),
    "m3_b_s13": ("reference", 1.0, False),
    "soup_m3": ("reference", 1.0, False),
    "wm_a50": ("wise", 0.5, True),
    "wm_a75": ("wise", 0.75, True),
    "w_m3b_a40": ("wise", 0.4, True),
    "w_m3b_a60": ("wise", 0.6, True),
    "soup_a40": ("soup", 0.4, True),
    "soup_a50": ("soup", 0.5, True),
    "soup_a60": ("soup", 0.6, True),
    "soup_a75": ("soup", 0.75, True),
    "m3_lo_s13": ("lora", 1.0, True),
    "w_m3_lo_s13_a75": ("lora", 0.75, True),
    "w_m3_lo_s13_a50": ("lora", 0.5, True),
    "m3_fr_s13": ("freeze", 1.0, True),
    "w_m3_fr_s13_a75": ("freeze", 0.75, True),
    "w_m3_fr_s13_a50": ("freeze", 0.5, True),
    # amendment 2 (PROGRESS.md 2026-11-04 ~21:35): ONE extra pair of alphas for the winning family only (0.30; 0.35 only if 0.30 fails eligibility)
    "w_m3b_a30": ("wise", 0.3, True),
    "w_m3b_a35": ("wise", 0.35, True),
    "soup_a30": ("soup", 0.3, True),
    "soup_a35": ("soup", 0.35, True),
    "w_m3_fr_s13_a30": ("freeze", 0.3, True),
    "w_m3_fr_s13_a35": ("freeze", 0.35, True),
}
DEV_ALIAS = {"bge-m3": "base_bge-m3"}  # the base model's extended-DEV result is stored under base_<name>
FAMILY_RANK = {"wise": 0, "soup": 1, "lora": 2, "freeze": 2}


def components(name: str) -> dict | None:
    dev, devm = R.load(DEV_ALIAS.get(name, name), "dev"), R.load(name, "devM")
    suite = V2RES / "suite" / f"{name}.json"
    proxy_all = json.loads((V2RES / "ood_dev_ar.json").read_text())
    if not (dev and devm and suite.exists() and name in proxy_all):
        return None
    s = json.loads(suite.read_text())
    return {
        "dev": dev["directions"]["macro"]["mrr@10"] * 100,
        "ar_en": dev["directions"]["ar->en"]["mrr@10"] * 100,
        "devM": devm["slices"]["dev_miracl_ar"]["mrr@10"] * 100,
        "proxy": proxy_all[name]["mrr@10"] * 100,
        "xnli": s["xnli_ap"] * 100,
        "cls": s["cls"] * 100,
        "sts": s["sts_spearman"] * 100,
        "sarcasm": s["sarcasm_acc"] * 100,
        "sentiment": s["sentiment_acc"] * 100,
    }


def table() -> list[dict]:
    rows = []
    for name, (fam, alpha, sel) in MODELS.items():
        c = components(name)
        if c:
            rows.append({"name": name, "family": fam, "alpha": alpha, "selectable": sel, **c})
    for k in COMPONENTS:
        lo, hi = min(r[k] for r in rows), max(r[k] for r in rows)
        for r in rows:
            r[f"n_{k}"] = (r[k] - lo) / max(hi - lo, FLOOR)
    for r in rows:
        r["score"] = float(np.mean([r[f"n_{k}"] for k in COMPONENTS]))
    return rows


def render(rows: list[dict], ref_ar_en: float) -> str:
    out = [
        "| model | family | alpha | dev | ar->en | devM | proxy | XNLI(tr) | cls | (sarc/sent) | STS | **score** | eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["score"]):
        el = "ref" if not r["selectable"] else ("yes" if r["ar_en"] >= ref_ar_en + MARGIN_AR_EN else "no")
        out.append(
            f"| {r['name']} | {r['family']} | {r['alpha']:.2f} | {r['dev']:.1f} | {r['ar_en']:.1f} | {r['devM']:.1f} | {r['proxy']:.1f} | {r['xnli']:.1f} | {r['cls']:.1f} | "
            f"{r['sarcasm']:.1f}/{r['sentiment']:.1f} | {r['sts']:.1f} | **{r['score']:.4f}** | {el} |"
        )
    return "\n".join(out)


def pick(rows: list[dict], ref_ar_en: float) -> tuple[dict | None, str]:
    """Highest score among the eligible, selectable models; a model whose score is within STRICTLY less than TIE of the best is tied (|diff| < 0.02) -> fewer ingredients
    (single-seed WiSE < soup WiSE < LoRA / freeze), then larger alpha.  Returns (pick, explanation of the clause that decided)."""
    ok = sorted((r for r in rows if r["selectable"] and r["ar_en"] >= ref_ar_en + MARGIN_AR_EN), key=lambda r: -r["score"])
    if not ok:
        return None, "no eligible candidate"
    best = ok[0]
    near = [r for r in ok if best["score"] - r["score"] < TIE]
    chosen = min(near, key=lambda r: (FAMILY_RANK[r["family"]], -r["alpha"]))
    if len(ok) > 1:
        gap = best["score"] - ok[1]["score"]
        txt = f"top score {best['name']} {best['score']:.4f}; runner-up {ok[1]['name']} {ok[1]['score']:.4f}; difference {gap:.4f}"
    else:
        txt = f"only eligible candidate {best['name']}"
    if len(near) == 1:
        return chosen, txt + f" -> not a tie (difference >= {TIE}) -> decided by SCORE"
    return chosen, txt + f" -> within the tie band (< {TIE}) among {[r['name'] for r in near]} -> decided by the INGREDIENTS clause"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--require", default="", help="comma list of models that must all have complete results before a pick may be written")
    ap.add_argument("--family-only", action="store_true", help="print only the winning family and its best alpha")
    a = ap.parse_args()
    rows = table()
    ref = next((r for r in rows if r["name"] == "bge-m3"), None)
    assert ref, "bge-m3 reference row incomplete"
    have = {r["name"] for r in rows}
    missing = [n for n in MODELS if n not in have]
    print(render(rows, ref["ar_en"]))
    print("\nnot in the pool (no complete results):", ", ".join(missing) or "none")
    p, why = pick(rows, ref["ar_en"])
    if p:
        print(
            f"\nPICK on the pool above: {p['name']} (family {p['family']}, alpha {p['alpha']}): score {p['score']:.4f}  dev {p['dev']:.1f}  ar->en {p['ar_en']:.1f}  devM {p['devM']:.1f}  "
            f"proxy {p['proxy']:.1f}  XNLI {p['xnli']:.1f}  cls {p['cls']:.1f}  STS {p['sts']:.1f}\nRULE: {why}"
        )
        print(f"WINNING_FAMILY={p['family']}")
    need = [n for n in a.require.split(",") if n]
    gone = [n for n in need if n not in have]
    if a.write and p and not gone:
        (V2RES / "suite_pick.json").write_text(
            json.dumps(
                {"pick": p["name"], "family": p["family"], "alpha": p["alpha"], "score": p["score"], "rule": why, "rows": rows}, indent=1
            )
        )
        print("wrote suite_pick.json")
    elif a.write:
        print("NOT WRITTEN: required models missing:", gone)


if __name__ == "__main__":
    main()
