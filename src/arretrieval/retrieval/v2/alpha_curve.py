"""POST-HOC blend curve of the m3_b seed-13 family (NOT used for selection; coordinator addition B, pre-registered in PROGRESS.md before it ran).

alpha = the fine-tuned share (0 = bge-m3, 1 = the raw fine-tune m3_b_s13): W = (1 - alpha) W_bge-m3 + alpha W_ft.   Scores are FRACTIONS.
Format (agreed with the report agent):
  {"selected_alpha": <alpha of the pre-registered pick>, "points": [{"alpha": 0.0, "test": {"ar->en": .., "macro": ..},
    "guards": {"MIRACLRetrievalHardNegatives.v2": .., "NamaaMrTydiReranking": .., "SemRel24STS": ..}, ...}]}
(extra keys per point: other TEST directions, devM, mMARCO-ar proxy, model names, MTEB result source.)  Existing results are reused; the queue only ran the missing TEST evals
and guard-only MTEB runs (result names alpha_<model>).  Standard library only.   Writes runs/retrieval/v2/results/alpha_curve.json and prints a markdown table.
"""

from __future__ import annotations

import glob
import json
import os

ROOT = os.path.expanduser("~/projects/mteb-arabic")
RES = os.path.expanduser("~/projects/qubit-ai-lab/runs/retrieval/v2/results")
# alpha -> (model for TEST / devM / proxy, candidate MTEB result names in order of preference)
CURVE = [
    (0.0, "base_bge-m3", ["BAAI__bge-m3"]),
    (0.3, "w_m3b_a30", ["alpha_w_m3b_a30"]),
    (0.4, "w_m3b_a40", ["final_m3_if_pick_is_a40", "alpha_w_m3b_a40"]),
    (0.5, "wm_a50", ["alpha_wm_a50"]),
    (0.6, "w_m3b_a60", ["alpha_w_m3b_a60"]),
    (0.75, "wm_a75", ["final_m3_wa75_s13"]),
    (1.0, "m3_b_s13", ["alpha_m3_b_s13"]),
]
ALIAS = {"base_bge-m3": "bge-m3"}  # devM / proxy results of the base model are stored under "bge-m3", its TEST under "base_bge-m3"
GUARDS = ["MIRACLRetrievalHardNegatives.v2", "NamaaMrTydiReranking", "SemRel24STS"]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mteb_score(names: list[str], task: str):
    for n in names:
        for cache in ("results_cache_v2", "results_cache"):
            for d in glob.glob(f"{ROOT}/{cache}/results/{n}/*/"):
                r = jload(os.path.join(d, task + ".json"))
                if r:
                    vals = [s["main_score"] for sp in r["scores"].values() for s in sp if "main_score" in s]
                    if vals:
                        return sum(vals) / len(vals), n
    return None, None


def main() -> None:
    pick_file = jload(f"{RES}/suite_pick.json") or {}
    pick = pick_file.get("pick")
    proxy = jload(f"{RES}/ood_dev_ar.json") or {}
    points = []
    for alpha, model, mteb_names in CURVE:
        mteb_names = ["final_m3" if n == "final_m3_if_pick_is_a40" and pick == "w_m3b_a40" else n for n in mteb_names]
        test, devm = jload(f"{RES}/{model}.test.json"), jload(f"{RES}/{ALIAS.get(model, model)}.devM.json")
        pt: dict = {"alpha": alpha, "model": model, "posthoc": True, "test": {}, "guards": {}, "guards_source": {}}
        if test:
            pt["test"] = {d: test["directions"][d]["mrr@10"] for d in ("ar->en", "en->ar", "ar->ar", "en->en", "macro")}
        if devm:
            pt["devM"] = devm["slices"]["dev_miracl_ar"]["mrr@10"]
        if ALIAS.get(model, model) in proxy:
            pt["proxy_mmarco_ar"] = proxy[ALIAS.get(model, model)]["mrr@10"]
        for task in GUARDS:
            v, src = mteb_score(mteb_names, task)
            pt["guards"][task], pt["guards_source"][task] = v, src
        points.append(pt)
    out = {
        "selected_alpha": pick_file.get("alpha"),
        "selected_model": pick,
        "selected_family": pick_file.get("family"),
        "note": "POST-HOC figure of the m3_b seed-13 family, never used for selection (coordinator addition B). alpha = fine-tuned share; scores are fractions.",
        "points": points,
    }
    with open(f"{RES}/alpha_curve.json", "w") as f:
        json.dump(out, f, indent=1)
    f_ = lambda v: "-" if v is None else f"{100 * v:.1f}"
    print(
        "| alpha | TEST ar->en | TEST macro | devM | proxy | MIRACL-HN ar | NamaaMrTydi | SemRel24 |\n|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for p in points:
        g = p["guards"]
        print(
            f"| {p['alpha']:.2f} | {f_(p['test'].get('ar->en'))} | {f_(p['test'].get('macro'))} | {f_(p.get('devM'))} | {f_(p.get('proxy_mmarco_ar'))} | "
            f"{f_(g[GUARDS[0]])} | {f_(g[GUARDS[1]])} | {f_(g[GUARDS[2]])} |"
        )


if __name__ == "__main__":
    main()
