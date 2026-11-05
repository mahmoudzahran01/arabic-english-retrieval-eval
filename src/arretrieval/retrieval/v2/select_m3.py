"""Apply the PRE-REGISTERED selection rule (PROGRESS.md, 2026-11-04 ~14:00) to the v2-m3 candidates. No guard and no test number is read here.

candidates: m3_b_s13 (reference), m3_r_s13 (replay), m3_p_s13 (preservation), m3_rp_s13 (both), each raw and WiSE alpha .75 / .50 against bge-m3.
qualifies  = (i) devM >= bge-m3 devM - 1.0   and (ii) dev ar->en >= 79.0 (~ TEST 72)   and (iii) dev macro >= m3_b dev macro - 1.5
choice     = highest dev macro among the qualifying; tie (within 0.2) -> the one with fewer moving parts (raw < replay < preservation < both; alpha 1 < .75 < .5).
if none qualifies: highest devM among those with dev ar->en >= 79.0 (reported as a miss).
Run: NO_GPU=1 python -m arretrieval.retrieval.v2.select_m3
"""

from __future__ import annotations

from . import report_v2 as R

BASE = ["m3_b_s13", "m3_r_s13", "m3_p_s13", "m3_rp_s13", "m3_lo_s13", "m3_fr_s13"]
ALPHAS = [("", 1.0), ("w_{}_a75", 0.75), ("w_{}_a50", 0.5)]
SPECIAL = {"m3_b_s13": {0.75: "wm_a75", 0.5: "wm_a50"}}  # the first WiSE blends were named differently
COMPLEXITY = {"m3_b_s13": 0, "m3_r_s13": 1, "m3_p_s13": 2, "m3_rp_s13": 3, "m3_lo_s13": 2, "m3_fr_s13": 2}


def main() -> None:
    ref_dev = R.load("m3_b_s13", "dev")["directions"]["macro"]["mrr@10"] * 100
    ref_devm = R.load("bge-m3", "devM")["slices"]["dev_miracl_ar"]["mrr@10"] * 100
    rows = []
    for b in BASE:
        for pat, a in ALPHAS:
            name = b if a == 1.0 else (SPECIAL.get(b, {}).get(a) or pat.format(b))
            dev, devm = R.load(name, "dev"), R.load(name, "devM")
            if not dev or not devm:
                continue
            d = dev["directions"]
            rows.append(
                dict(
                    name=name,
                    base=b,
                    alpha=a,
                    macro=d["macro"]["mrr@10"] * 100,
                    ar_en=d["ar->en"]["mrr@10"] * 100,
                    devm=devm["slices"]["dev_miracl_ar"]["mrr@10"] * 100,
                )
            )
    for r in rows:
        r["ok"] = r["devm"] >= ref_devm - 1.0 and r["ar_en"] >= 79.0 and r["macro"] >= ref_dev - 1.5
    print(
        f"reference: m3_b dev macro {ref_dev:.1f}; bge-m3 devM {ref_devm:.1f}  -> thresholds devM >= {ref_devm - 1:.1f}, ar->en >= 79.0, macro >= {ref_dev - 1.5:.1f}"
    )
    print("| candidate | alpha | dev macro | dev ar->en | devM | qualifies |\n|---|---:|---:|---:|---:|---|")
    for r in rows:
        print(f"| {r['name']} | {r['alpha']} | {r['macro']:.1f} | {r['ar_en']:.1f} | {r['devm']:.1f} | {'yes' if r['ok'] else 'no'} |")
    ok = [r for r in rows if r["ok"]]
    if ok:
        best = max(r["macro"] for r in ok)
        near = [r for r in ok if r["macro"] >= best - 0.2]
        pick = min(near, key=lambda r: (COMPLEXITY[r["base"]], -r["alpha"]))
        print(
            f"\nSELECTED: {pick['name']} (alpha {pick['alpha']}): dev macro {pick['macro']:.1f}, ar->en {pick['ar_en']:.1f}, devM {pick['devm']:.1f}"
        )
    else:
        pool = [r for r in rows if r["ar_en"] >= 79.0]
        pick = max(pool, key=lambda r: r["devm"]) if pool else None
        print("\nNONE QUALIFIES (reported as a miss); highest devM with ar->en >= 79.0:", pick and pick["name"])


if __name__ == "__main__":
    main()
