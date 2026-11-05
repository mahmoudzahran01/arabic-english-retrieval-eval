"""v2.1 selection (PRE-REGISTERED in PROGRESS.md, 2026-11-05 ~08:50).  CPU only; no TEST and no MTEB(ara) score of any candidate is read.

Same machinery as suite_select.py (six components dev / devM / proxy / xnli / cls / sts, pool-wide min-max with a 1-point floor, unweighted mean, eligibility DEV ar->en >= bge-m3 + 5.0,
strict tie band < 0.02) with ONE documented extension: component 7 'arr' (ar_rerank_proxy.py), used only if results/arr/_gate.json says the validity gate passed.
Pool: bge-m3 + the incumbent family (w_m3b_a30 / w_m3b_a40 = v2 "stay" / wm_a50 / w_m3b_a60, raw m3_b_s13) as reference rows + 12 selectable candidates
(recipes A = +data-v3 translations, B = +Arabic hard negatives re-mined with v2-m3, AB = both; alpha 0.3 / 0.4 / 0.5 / 0.6).
Decision: the best eligible candidate goes to confirmation only if it is not the incumbent and its score >= the incumbent's + 0.02; otherwise v2 stays.
Run: NO_GPU=1 python -m arretrieval.retrieval.v2.suite_select_v21 [--write]     -> results/suite_v21_pick.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import report_v2 as R
from .common_v2 import V2RES
from .suite_select import DEV_ALIAS, FLOOR, MARGIN_AR_EN, TIE

BASE6 = ["dev", "devM", "proxy", "xnli", "cls", "sts"]
INGREDIENTS = {"stay": 0, "A": 1, "B": 1, "AB": 2}
MODELS: dict[str, tuple[str, float, bool]] = {  # name -> (recipe, alpha, selectable)
    "bge-m3": ("ref", 0.0, False),
    "m3_b_s13": ("ref", 1.0, False),
    "w_m3b_a30": ("ref", 0.3, False),
    "w_m3b_a40": ("stay", 0.4, True),
    "wm_a50": ("ref", 0.5, False),
    "w_m3b_a60": ("ref", 0.6, False),
}
for rec in ("A", "B", "AB"):
    for a in (30, 40, 50, 60):
        MODELS[f"w_m21_{rec}_a{a}"] = (rec, a / 100, True)
ARR_ALIAS = {"bge-m3": "bge-m3"}


def gate() -> dict:
    p = V2RES / "arr" / "_gate.json"
    return json.loads(p.read_text()) if p.exists() else {"passed": False, "rho": None}


def components(name: str) -> dict | None:
    dev, devm = R.load(DEV_ALIAS.get(name, name), "dev"), R.load(name, "devM")
    suite, arr = V2RES / "suite" / f"{name}.json", V2RES / "arr" / f"{name}.json"
    proxy_all = json.loads((V2RES / "ood_dev_ar.json").read_text())
    if not (dev and devm and suite.exists() and arr.exists() and name in proxy_all):
        return None
    s = json.loads(suite.read_text())
    return {
        "dev": dev["directions"]["macro"]["mrr@10"] * 100,
        "ar_en": dev["directions"]["ar->en"]["mrr@10"] * 100,
        "ar_ar": dev["directions"]["ar->ar"]["mrr@10"] * 100,
        "devM": devm["slices"]["dev_miracl_ar"]["mrr@10"] * 100,
        "proxy": proxy_all[name]["mrr@10"] * 100,
        "xnli": s["xnli_ap"] * 100,
        "cls": s["cls"] * 100,
        "sts": s["sts_spearman"] * 100,
        "arr": json.loads(arr.read_text())["ap"] * 100,
    }


def table(use_arr: bool) -> list[dict]:
    rows = []
    for name, (rec, alpha, sel) in MODELS.items():
        c = components(name)
        if c:
            rows.append({"name": name, "recipe": rec, "alpha": alpha, "selectable": sel, **c})
    for k in BASE6 + ["arr"]:
        lo, hi = min(r[k] for r in rows), max(r[k] for r in rows)
        for r in rows:
            r[f"n_{k}"] = (r[k] - lo) / max(hi - lo, FLOOR)
    for r in rows:
        r["score6"] = float(np.mean([r[f"n_{k}"] for k in BASE6]))
        r["score7"] = float(np.mean([r[f"n_{k}"] for k in BASE6 + ["arr"]]))
        r["score"] = r["score7"] if use_arr else r["score6"]
    return rows


def render(rows: list[dict], ref_ar_en: float) -> str:
    out = [
        "| model | recipe | alpha | dev | ar->en | devM | proxy | XNLI(tr) | cls | STS | ar_rerank | score (6) | score (7) | **used** | eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["score"]):
        el = "ref" if not r["selectable"] else ("yes" if r["ar_en"] >= ref_ar_en + MARGIN_AR_EN else "no")
        out.append(
            f"| {r['name']} | {r['recipe']} | {r['alpha']:.2f} | {r['dev']:.1f} | {r['ar_en']:.1f} | {r['devM']:.1f} | {r['proxy']:.1f} | {r['xnli']:.1f} | {r['cls']:.1f} | {r['sts']:.1f} | "
            f"{r['arr']:.1f} | {r['score6']:.4f} | {r['score7']:.4f} | **{r['score']:.4f}** | {el} |"
        )
    return "\n".join(out)


def decide(rows: list[dict], ref_ar_en: float) -> dict:
    ok = sorted((r for r in rows if r["selectable"] and r["ar_en"] >= ref_ar_en + MARGIN_AR_EN), key=lambda r: -r["score"])
    inc = next((r for r in rows if r["recipe"] == "stay"), None)
    if not ok or inc is None:
        return {"decision": "stay", "reason": "no eligible candidate or incumbent row missing", "pick": None}
    best = ok[0]["score"]
    near = [r for r in ok if best - r["score"] < TIE]
    chosen = min(near, key=lambda r: (INGREDIENTS[r["recipe"]], -r["alpha"]))
    gap = chosen["score"] - inc["score"]
    why = (
        f"best eligible score {ok[0]['name']} {best:.4f}; tie band < {TIE} contains {[r['name'] for r in near]} -> ingredients clause picks {chosen['name']}; incumbent w_m3b_a40 {inc['score']:.4f}; "
        f"chosen minus incumbent = {gap:+.4f}"
    )
    if chosen["recipe"] == "stay" or gap < TIE:
        return {
            "decision": "stay",
            "pick": chosen["name"],
            "reason": why + f" -> NOT >= incumbent + {TIE}: v2 stays, nothing further is spent",
        }
    return {
        "decision": "confirm",
        "pick": chosen["name"],
        "recipe": chosen["recipe"],
        "alpha": chosen["alpha"],
        "reason": why + f" -> >= incumbent + {TIE}: confirm with seeds 18 and 19",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    g = gate()
    rows = table(bool(g.get("passed")))
    ref = next(r for r in rows if r["name"] == "bge-m3")
    miss = [n for n in MODELS if n not in {r["name"] for r in rows}]
    print(
        f"component 7 (ar_rerank) validity gate: rho={g.get('rho')} n={g.get('n_models')} passed={g.get('passed')}  -> {'7' if g.get('passed') else '6'}-component score is used\n"
    )
    print(render(rows, ref["ar_en"]))
    print("\nnot in the pool (incomplete):", ", ".join(miss) or "none")
    d = decide(rows, ref["ar_en"])
    print("\nDECISION:", d["decision"], "| pick:", d["pick"], "|", d["reason"])
    if a.write and not [m for m in miss if m.startswith("w_m21_")]:
        (V2RES / "suite_v21_pick.json").write_text(json.dumps({**d, "gate": g, "rows": rows}, indent=1))
        print("wrote suite_v21_pick.json")
    elif a.write:
        print("NOT WRITTEN: candidates missing")


if __name__ == "__main__":
    main()
