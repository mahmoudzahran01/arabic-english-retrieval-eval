"""Build runs/retrieval/v2/RESULTS_v2.md (FINAL write-up).  Every number in a table comes from the stored result files; the prose quotes numbers through helper calls.

Inputs : results/*.json (+ .perq.npz), results/{suite_pick,alpha_curve,mteb_paired,mteb_board,confirmatory,ood_dev_ar,audit_ood*,diag_xquad_style}.json, results/guard_split/_report.md, *.train.json
Run    : NO_GPU=1 python -m arretrieval.retrieval.v2.build_final_v2  [out.md]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import build_results_v2 as B
from . import report_v2 as R
from . import suite_select as S
from .common_v2 import V2, V2RES

BGE, E5B, E5S, V1S13 = "base_bge-m3", "base_e5-base", "base_e5-small", "e5s_hard1_xl_e1_s13"
V1 = [V1S13, "e5s_hard1_xl_e1_s14", "e5s_hard1_xl_e1_s15"]
M3 = ["final_m3", "w_m3b_a40_s14", "w_m3b_a40_s15"]  # development seeds 13/14/15 of the frozen recipe (WiSE 0.4 towards bge-m3)
CONF = ["w_m3b_a40_s16", "w_m3b_a40_s17"]  # confirmatory seeds
RAW = ["m3_b_s13", "m3_b_s14", "m3_b_s15"]
RAWC = ["m3_b_s16", "m3_b_s17"]
STB, STS = ["st_b_s13", "st_b_s14", "st_b_s15"], ["st_s_s13", "st_s_s14", "st_s_s15"]
DIRS = ("ar->ar", "en->en", "ar->en", "en->ar")


def jl(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def tv(name: str, d: str, split: str = "test") -> float:
    return R.load(name, split)["directions"][d]["mrr@10"] * 100


def tavg(names: list[str], d: str, split: str = "test") -> float:
    return float(np.mean([tv(n, d, split) for n in names]))


def cells(names: list[str], split: str = "test") -> str:
    rs = [R.load(n, split) for n in names]
    out = []
    for k in DIRS + ("macro",):
        v = np.array([r["directions"][k]["mrr@10"] * 100 for r in rs])
        out.append(f"{v.mean():.1f}" + (f" ± {v.std(ddof=1):.1f}" if len(v) > 1 else ""))
    return " | ".join(out)


def headline() -> str:
    rows = [
        ("bge-m3 (off the shelf)", "568M", [BGE]),
        ("multilingual-e5-base (off the shelf)", "278M", [E5B]),
        ("multilingual-e5-small (off the shelf)", "118M", [E5S]),
        ("v1 e5-small, 3 seeds", "118M", V1),
        ("**v2-m3** = bge-m3 fine-tuned + WiSE 0.4 (pick), seed 13", "568M", ["final_m3"]),
        ("v2-m3, development seeds 13/14/15", "568M", M3),
        ("v2-m3, **confirmatory** seeds 16/17", "568M", CONF),
        ("bge-m3 fine-tuned, RAW (alpha 1), seeds 13/14/15", "568M", RAW),
        ("bge-m3 fine-tuned, RAW (alpha 1), seeds 16/17", "568M", RAWC),
        ("**v2-base** = e5-base distilled from v2-m3, 3-seed soup", "278M", ["st_b_soup"]),
        ("v2-base, the 3 seeds", "278M", STB),
        ("**v2-small** = e5-small distilled from v2-m3, 3-seed soup", "118M", ["st_s_soup"]),
        ("v2-small, the 3 seeds", "118M", STS),
    ]
    lines = ["| system | params | ar->ar | en->en | ar->en | en->ar | macro | n |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {lab} | {p} | {cells(ns)} | {len(ns)} |" for lab, p, ns in rows]
    return "\n".join(lines)


def boot() -> str:
    m3_dev, m3_conf = "+".join(M3), "+".join(CONF)
    pairs = [
        ("v2-m3 (seed 13) vs bge-m3", BGE, "final_m3"),
        ("v2-m3, mean of dev seeds 13/14/15 vs bge-m3", BGE, m3_dev),
        ("v2-m3, mean of CONFIRMATORY seeds 16/17 vs bge-m3", BGE, m3_conf),
        ("v2-m3 (seed 13) vs v1 (seed 13)", V1S13, "final_m3"),
        ("v2-base (soup) vs e5-base", E5B, "st_b_soup"),
        ("v2-base (soup) vs bge-m3", BGE, "st_b_soup"),
        ("v2-base (soup) vs v1 (seed 13)", V1S13, "st_b_soup"),
        ("v2-small (soup) vs e5-small", E5S, "st_s_soup"),
        ("v2-small (soup) vs v1 (seed 13)", V1S13, "st_s_soup"),
        ("v2-small (soup) vs bge-m3", BGE, "st_s_soup"),
    ]
    return R.delta_table(pairs)


def mixed() -> str:
    return R.mixed_table(
        [
            ("bge-m3", BGE),
            ("e5-base", E5B),
            ("e5-small", E5S),
            ("v1 (seed 13)", "v1_s13"),
            ("v2-m3 (pick)", "final_m3"),
            ("bge-m3 fine-tuned RAW (seed 13)", "m3_b_s13"),
            ("v2-base (seed 13)", "st_b_s13"),
            ("v2-small (seed 13)", "st_s_s13"),
        ],
        "test",
    )


PROXY = jl(V2RES / "ood_dev_ar.json") or {}


def devrow(label: str, name: str, desc: str = "") -> str | None:
    r = R.load(name, "dev")
    if not r:
        return None
    d = r["directions"]
    dm = R.load(name if name != BGE else "bge-m3", "devM")
    pk = {BGE: "bge-m3"}.get(name, name)
    devm = f"{dm['slices']['dev_miracl_ar']['mrr@10'] * 100:.1f}" if dm else ""
    px = f"{PROXY[pk]['mrr@10'] * 100:.1f}" if pk in PROXY else ""
    return f"| `{name}` | {desc or label} | " + " | ".join(f"{d[k]['mrr@10'] * 100:.1f}" for k in DIRS + ("macro",)) + f" | {devm} | {px} |"


def dev_block(items: list[tuple[str, str]]) -> str:
    head = [
        "| run | what | ar->ar | en->en | ar->en | en->ar | macro | devM | mMARCO proxy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    return "\n".join(head + [x for x in (devrow(n, n, d) for n, d in items) if x])


M3_ITEMS = [
    (BGE, "bge-m3 off the shelf (reference)"),
    ("m3_a_s13", "bge-m3 fine-tuned itself, first run (11.9k translations)"),
    (
        "m3_b_s13",
        "**bge-m3 fine-tuned, full filtered v2 data (39,377 queries), lr 1e-5, scale 50, CachedMNRL(16), 1 epoch, seed 13 = alpha 1**",
    ),
    ("m3_b_s14", "same, seed 14"),
    ("m3_b_s15", "same, seed 15"),
    ("m3_r_s13", "+ MIRACL-ar train replay x3 (1,515 queries)"),
    ("m3_p_s13", "+ preservation loss towards frozen bge-m3 (w 0.5)"),
    ("m3_rp_s13", "+ replay and preservation"),
    ("m3_fr_s13", "frozen embeddings + bottom 12 layers"),
    ("m3_lo_s13", "LoRA r=32 (attention + MLP), lr 1e-4"),
    ("wm_a75", "WiSE 0.75 of m3_b_s13 towards bge-m3"),
    ("wm_a50", "WiSE 0.5"),
    ("w_m3b_a60", "WiSE 0.6"),
    ("w_m3b_a40", "**WiSE 0.4 (= the pick, final_m3)**"),
    ("w_m3b_a35", "WiSE 0.35"),
    ("w_m3b_a30", "WiSE 0.3"),
    ("soup_m3", "uniform soup of seeds 13/14/15 (raw)"),
    ("soup_a75", "soup, WiSE 0.75"),
    ("soup_a60", "soup, WiSE 0.6"),
    ("soup_a50", "soup, WiSE 0.5"),
    ("soup_a40", "soup, WiSE 0.4"),
    ("w_m3_r_s13_a75", "replay run, WiSE 0.75"),
    ("w_m3_p_s13_a75", "preservation run, WiSE 0.75"),
    ("w_m3_fr_s13_a75", "freeze run, WiSE 0.75"),
    ("w_m3_fr_s13_a50", "freeze run, WiSE 0.5"),
    ("w_m3_lo_s13_a75", "LoRA run, WiSE 0.75"),
    ("w_m3_lo_s13_a50", "LoRA run, WiSE 0.5"),
]
ST_ITEMS = [
    (E5B, "e5-base off the shelf"),
    ("f4_b_e1_s13", "e5-base fine-tuned on the v2 data, no distillation"),
    ("f5_b_kd_s13", "+ KD from the raw v2-m3 seed 13 (CE + KL w=1, 1 epoch)"),
    ("f5_b_kd3_s13", "+ KD w=3"),
    ("f5_b_kdonly_s13", "KD only (no CE), 1 epoch"),
    ("f5_b_kd_e2_s13", "CE + KD w=1, 2 epochs"),
    ("f5_b_kdonly_e2_s13", "KD only, 2 epochs  (**chosen recipe**)"),
    ("st_b_s13", "**FINAL recipe, teacher = final_m3**: e5-base KD-only 2 epochs, seed 13"),
    ("st_b_s14", "seed 14"),
    ("st_b_s15", "seed 15"),
    ("st_b_soup", "**3-seed soup = v2-base**"),
    (E5S, "e5-small off the shelf"),
    ("f4_s_e1_s13", "e5-small fine-tuned on the v2 data, no distillation"),
    ("f5_s_kd_s13", "+ KD from the raw v2-m3 seed 13"),
    ("st_s_s13", "**FINAL recipe, teacher = final_m3**: e5-small KD-only 2 epochs, seed 13"),
    ("st_s_s14", "seed 14"),
    ("st_s_s15", "seed 15"),
    ("st_s_soup", "**3-seed soup = v2-small**"),
]


def suite_table() -> str:
    rows = S.table()
    ref = next(r for r in rows if r["name"] == "bge-m3")
    return S.render(rows, ref["ar_en"])


def board() -> str:
    b = jl(V2RES / "mteb_board.json") or []
    order = sorted(b, key=lambda x: -x["Mean(Task)"])
    lab = {
        "final_m3": "**v2-m3 (pick)**",
        "w_m3b_a40_s16": "v2-m3 confirmatory seed 16",
        "w_m3b_a40_s17": "v2-m3 confirmatory seed 17",
        "final_base": "**v2-base**",
        "final_small": "**v2-small**",
        "BAAI__bge-m3": "BAAI/bge-m3",
        "intfloat__multilingual-e5-large": "intfloat/multilingual-e5-large",
        "intfloat__multilingual-e5-base": "intfloat/multilingual-e5-base",
        "intfloat__multilingual-e5-small": "intfloat/multilingual-e5-small",
        "mzahran1__arabic-english-e5-small-retrieval": "v1 (mzahran1/arabic-english-e5-small-retrieval)",
    }
    types = ["Retrieval", "Reranking", "STS", "PairClassification", "Classification", "Clustering", "BitextMining"]
    lines = [
        "| model | params | Mean (Task) | Mean (TaskType) | Retr. | Rerank. | STS | PairClf | Classif. | Clust. | Bitext | ar-only subsets | **ar-en cross-lingual subsets** |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in order:
        lines.append(
            f"| {lab.get(x['model'], x['model'])} | {x['params_M']}M | {x['Mean(Task)']:.2f} | {x['Mean(TaskType)']:.2f} | "
            + " | ".join(f"{x['type:' + t]:.1f}" for t in types)
            + f" | {x['ar-only subsets']:.1f} | {x['ar-en cross-lingual subsets']:.1f} |"
        )
    return "\n".join(lines)


SHARE = {
    "MIRACLRetrievalHardNegatives.v2": "TyDi/MIRACL questions (near-duplicates filtered; 9 exact copies remain in the training rows, section 8.1)",
    "XQuADRetrieval": "SQuAD-dev translations = v1 TEST source (excluded)",
    "NamaaMrTydiReranking": "Mr.TyDi test (TyDi source; v2 trains on TyDi-ar train)",
    "MLQARetrieval": "SQuAD-style Wikipedia (no training use)",
}


def per_task() -> str:
    b = {x["model"]: x for x in (jl(V2RES / "mteb_board.json") or [])}
    cols = [("BAAI__bge-m3", "bge-m3"), ("final_m3", "v2-m3"), ("w_m3b_a40_s16", "s16"), ("w_m3b_a40_s17", "s17")]
    cols2 = [
        ("intfloat__multilingual-e5-base", "e5-base"),
        ("final_base", "v2-base"),
        ("intfloat__multilingual-e5-small", "e5-small"),
        ("mzahran1__arabic-english-e5-small-retrieval", "v1"),
        ("final_small", "v2-small"),
    ]
    tasks = list(b["BAAI__bge-m3"]["per_task"])
    h1 = (
        "| task | "
        + " | ".join(c[1] for c in cols)
        + " | v2-m3 - bge-m3 | shares a source with our training data? |\n|---|"
        + "---:|" * (len(cols) + 1)
        + "---|"
    )
    l1 = [h1] + [
        f"| {t} | "
        + " | ".join(f"{b[m]['per_task'][t]:.1f}" for m, _ in cols)
        + f" | {b['final_m3']['per_task'][t] - b['BAAI__bge-m3']['per_task'][t]:+.1f} | {SHARE.get(t, '')} |"
        for t in tasks
    ]
    l1.append(
        "| **Mean (Task)** | "
        + " | ".join(f"{b[m]['Mean(Task)']:.2f}" for m, _ in cols)
        + f" | {b['final_m3']['Mean(Task)'] - b['BAAI__bge-m3']['Mean(Task)']:+.2f} | |"
    )
    h2 = (
        "| task | "
        + " | ".join(c[1] for c in cols2)
        + " | v2-base - e5-base | v2-small - e5-small | v2-small - v1 |\n|---|"
        + "---:|" * (len(cols2) + 3)
    )
    l2 = [h2] + [
        f"| {t} | "
        + " | ".join(f"{b[m]['per_task'][t]:.1f}" for m, _ in cols2)
        + f" | {b['final_base']['per_task'][t] - b['intfloat__multilingual-e5-base']['per_task'][t]:+.1f} | {b['final_small']['per_task'][t] - b['intfloat__multilingual-e5-small']['per_task'][t]:+.1f} | "
        f"{b['final_small']['per_task'][t] - b['mzahran1__arabic-english-e5-small-retrieval']['per_task'][t]:+.1f} |"
        for t in tasks
    ]
    l2.append(
        "| **Mean (Task)** | "
        + " | ".join(f"{b[m]['Mean(Task)']:.2f}" for m, _ in cols2)
        + f" | {b['final_base']['Mean(Task)'] - b['intfloat__multilingual-e5-base']['Mean(Task)']:+.2f} | "
        f"{b['final_small']['Mean(Task)'] - b['intfloat__multilingual-e5-small']['Mean(Task)']:+.2f} | {b['final_small']['Mean(Task)'] - b['mzahran1__arabic-english-e5-small-retrieval']['Mean(Task)']:+.2f} |"
    )
    return "\n".join(l1) + "\n\n" + "\n".join(l2)


def paired() -> str:
    d = jl(V2RES / "mteb_paired.json") or {}
    lines = [
        "| new | reference | Mean(Task) new / ref | difference | 95% CI (resampling the 19 tasks, 10k) | wins / losses / ties | exact sign test p |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for k, r in d.items():
        if not r or "error" in r:
            continue
        lo, hi = r["ci95_task_bootstrap"]
        lines.append(
            f"| `{r['new']}` | `{r['ref']}` | {r['mean_new']:.2f} / {r['mean_ref']:.2f} | {r['mean_diff']:+.2f} | [{lo:+.2f}, {hi:+.2f}]{'' if lo <= 0 <= hi else ' *'} | {r['wins']} / {r['losses']} / {r['ties']} | {r['sign_test_p_two_sided']:.3f} |"
        )
    return "\n".join(lines)


def alpha() -> str:
    a = jl(V2RES / "alpha_curve.json")
    g = ["MIRACLRetrievalHardNegatives.v2", "NamaaMrTydiReranking", "SemRel24STS"]
    lines = [
        "| alpha (fine-tuned share) | TEST ar->en | TEST macro | devM | mMARCO proxy | MIRACL-HN ar | NamaaMrTydi | SemRel24STS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    f = lambda v: "-" if v is None else f"{100 * v:.1f}"
    for p in a["points"]:
        mark = " **(pick)**" if abs(p["alpha"] - (a["selected_alpha"] or -1)) < 1e-9 else ""
        lines.append(
            f"| {p['alpha']:.2f}{mark} | {f(p['test'].get('ar->en'))} | {f(p['test'].get('macro'))} | {f(p.get('devM'))} | {f(p.get('proxy_mmarco_ar'))} | "
            + " | ".join(f(p["guards"][t]) for t in g)
            + " |"
        )
    return "\n".join(lines)


def rerank() -> str:
    rows = [
        ("bge-m3", BGE),
        ("e5-base", E5B),
        ("e5-small", E5S),
        ("v2-m3 (pick)", "final_m3"),
        ("v2-base (soup)", "st_b_soup"),
        ("v2-small (soup)", "st_s_soup"),
    ]
    lines = [
        "| first stage | first stage ar->en | + reranker ar->en | first stage macro | + reranker macro | + reranker en->ar | + reranker ar->ar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lab, n in rows:
        a, b = R.load(n, "test"), R.load(f"{n}>>rr_ft_bce_s13", "test")
        if a and b:
            lines.append(
                f"| {lab} | {a['directions']['ar->en']['mrr@10'] * 100:.1f} | {b['directions']['ar->en']['mrr@10'] * 100:.1f} | {a['directions']['macro']['mrr@10'] * 100:.1f} | "
                f"{b['directions']['macro']['mrr@10'] * 100:.1f} | {b['directions']['en->ar']['mrr@10'] * 100:.1f} | {b['directions']['ar->ar']['mrr@10'] * 100:.1f} |"
            )
    return "\n".join(lines)


def cost() -> str:
    t = jl(V2RES / "mteb_eval_time.json") or {"hours": float("nan"), "n_models": 0}
    secs, n = 0.0, 0
    for p in V2RES.glob("*.train.json"):
        secs += float(json.loads(p.read_text()).get("train_seconds", 0))
        n += 1
    return (
        f"**GPU time (wall-clock on a shared GB10, so an upper bound on exclusive time):** {n} training runs = {secs / 3600:.1f} h of training; the 10 MTEB(ara) scorings of v2 and alpha-curve models = {t['hours']:.1f} h "
        f"of summed task evaluation_time; the gated job queue (drive scripts 2-17: training, merges, DEV/TEST/devM/mixed/proxy/suite evaluations, MTEB scorings, teacher embeddings, guard split, reranker) adds up to about 19 h of step wall-clock "
        f"(it includes short gate waits), plus about 3 h of ungated ablation jobs before the gate existed (estimate) - **roughly 22 h of GPU wall-clock in total**"
    )


def trail() -> str:
    rows = [
        (
            "2026-11-04 03:59",
            "Translation set FROZEN at 11,875 translations (ablation data); later extended to the full 22,313 and filtered (05:23 contamination filter built before the final runs)",
            "PROGRESS.md 03:59 / 05:23",
        ),
        (
            "2026-11-04 05:40",
            "Guards (MIRACL-HN.v2 ar, NamaaMrTydi, SemRel24) reported on final candidates only, never used for selection; mitigations judged on dev + a non-MTEB(ara) proxy (mMARCO-ar subsample)",
            "PROGRESS.md 05:40",
        ),
        (
            "2026-11-04 ~11:05",
            "WiSE rule fixed BEFORE any guard eval: dev macro within 0.5 of raw AND proxy within 1.0 of bge-m3 -> alpha 0.75 (the first v2-m3 candidate, final_m3_wa75_s13)",
            "PROGRESS.md 11:05",
        ),
        (
            "2026-11-04 ~14:00",
            "PRE-REGISTERED rule for the replay / preservation levers: qualifies if devM >= bge-m3 devM - 1.0, dev ar->en >= 79.0, dev macro >= m3_b - 1.5; highest dev macro among the qualifying",
            "PROGRESS.md 14:00 (applied 17:39: qualifiers wm_a75, wm_a50, w_m3_p_a75/a50 -> wm_a75)",
        ),
        (
            "2026-11-04 17:46",
            "Coordinator re-prioritisation: score final_m3_wa75_s13 on MTEB(ara) first (Mean(Task) 72.04 vs 71.98 = a tie), then parameter-limited runs",
            "PROGRESS.md 17:46",
        ),
        (
            "2026-11-04 ~19:05",
            "PRE-REGISTRATION of the 6-component validation suite, normalisation, pool, eligibility, tie-break (XNLI train-subsample deviation flagged: mteb's XNLI scores validation too)",
            "PROGRESS.md 19:05",
        ),
        (
            "2026-11-04 ~19:20",
            "CORRECTION (before any suite number existed): Arabic STS-B validation split from Omartificial-Intelligence-Space/Arabic-stsb (stsb_multi_mt has no Arabic)",
            "PROGRESS.md 19:20",
        ),
        (
            "2026-11-04 ~21:35",
            "AMENDMENT 2 (after the LoRA crash): rerun LoRA with the fixed merge; strict tie band; alpha 0.30 (+0.35 only if 0.30 fails eligibility) for the winning family only; full-pool final select",
            "PROGRESS.md 21:35",
        ),
        (
            "2026-11-04 ~22:10",
            "PRE-REGISTRATION 3: confirmatory seeds 16/17 with the frozen recipe, student rule (soup unless its dev macro is >0.5 below its seeds' mean), post-hoc blend curve, MTEB paired statistics",
            "PROGRESS.md 22:10",
        ),
        (
            "2026-11-04 22:46:54",
            "RECIPE FROZEN: family wise, alpha 0.4, pick w_m3b_a40 (score 0.7486; runner-up soup_a40 0.7612; gap 0.0127 < 0.02 -> ingredients clause)",
            "PROGRESS.md 22:46",
        ),
        ("2026-11-04 23:13", "Confirmatory seeds 16 and 17 start (after the freeze line)", "PROGRESS.md 23:13"),
        (
            "2026-11-05 01:13 / 02:56 / 03:04",
            "Teacher re-embedded from final_m3; student picks by the registered rule: final_base = st_b_soup, final_small = st_s_soup",
            "PROGRESS.md 01:13, 02:56, 03:04",
        ),
        (
            "2026-11-05 06:38 / 07:04",
            "CORRECTION of the drive16 guard-split step (coordinator-approved): 5 stale untitled cache rows recomputed with titles; sanity check bge-m3 MIRACL-HN 79.0 vs mteb 78.9 passed",
            "PROGRESS.md 06:38, 07:04",
        ),
        (
            "2026-11-05 07:2x",
            "CORRECTION (coordinator's exact-match checks): the contamination labels missed exact copies (893/1000 exact TyDi-train copies vs 814 flagged); 9 kept training rows equal MIRACL-HN dev queries; labels recomputed, leak disclosed, not retrained",
            "PROGRESS.md 07:2x",
        ),
    ]
    return "\n".join(
        ["| when (Spark clock, +03) | what was written down | where |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    )


def f1(x: float) -> str:
    return f"{x:.1f}"


def bd(a: str, b: str, d: str) -> str:
    r = R.boot_dir(a, b, "test", d)
    return f"{r['delta'] * 100:+.1f} [{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]"


def mt() -> dict:
    return {x["model"]: x for x in (jl(V2RES / "mteb_board.json") or [])}


def pr(k: str) -> dict:
    return (jl(V2RES / "mteb_paired.json") or {})[k]


def tldr() -> str:
    b, p_m3, p_base, p_small = (
        mt(),
        pr("final_m3 vs BAAI__bge-m3"),
        pr("final_base vs intfloat__multilingual-e5-base"),
        pr("final_small vs intfloat__multilingual-e5-small"),
    )
    p_v1 = pr("final_small vs mzahran1__arabic-english-e5-small-retrieval")
    g = lambda m, t: b[m]["per_task"][t]
    G = ["MIRACLRetrievalHardNegatives.v2", "NamaaMrTydiReranking", "SemRel24STS"]
    rr = lambda n, d: R.load(f"{n}>>rr_ft_bce_s13", "test")["directions"][d]["mrr@10"] * 100
    lines = [
        f"* **Arabic query -> English passage (TEST MRR@10, the v1 benchmark).** v2-m3 (bge-m3 fine-tuned, then weight-interpolated with bge-m3: 60% original + 40% fine-tuned, WiSE-FT) **{f1(tv('final_m3', 'ar->en'))}** vs bge-m3 {f1(tv(BGE, 'ar->en'))} "
        f"(paired bootstrap {bd(BGE, 'final_m3', 'ar->en')}) and v1 {f1(tavg(V1, 'ar->en'))} (3 seeds). Development seeds 13/14/15 average {f1(tavg(M3, 'ar->en'))}; the two **confirmatory** seeds (16, 17; recipe frozen before they were trained) "
        f"{f1(tv(CONF[0], 'ar->en'))} and {f1(tv(CONF[1], 'ar->en'))}. Macro over the four directions: {f1(tv('final_m3', 'macro'))} vs bge-m3 {f1(tv(BGE, 'macro'))}.",
        f"* **The ≤300M models** (distilled from v2-m3, KD only, 3-seed soups): v2-base (278M) ar->en **{f1(tv('st_b_soup', 'ar->en'))}** (e5-base {f1(tv(E5B, 'ar->en'))}; also above bge-m3's {f1(tv(BGE, 'ar->en'))}), "
        f"v2-small (118M) **{f1(tv('st_s_soup', 'ar->en'))}** (e5-small {f1(tv(E5S, 'ar->en'))}, v1 {f1(tavg(V1, 'ar->en'))}; it does NOT reach bge-m3). Their macros: {f1(tv('st_b_soup', 'macro'))} and {f1(tv('st_s_soup', 'macro'))} (bge-m3 {f1(tv(BGE, 'macro'))}).",
        f"* **MTEB(ara, v1), Mean (Task) over 19 tasks.** v2-m3 **{p_m3['mean_new']:.2f}** vs bge-m3 {p_m3['mean_ref']:.2f}: **{p_m3['mean_diff']:+.2f}, task-bootstrap 95% CI [{p_m3['ci95_task_bootstrap'][0]:+.2f}, {p_m3['ci95_task_bootstrap'][1]:+.2f}], "
        f"{p_m3['wins']} wins / {p_m3['losses']} losses / {p_m3['ties']} tie, sign test p={p_m3['sign_test_p_two_sided']:.2f} - numerically first, statistically a tie.** The two confirmatory seeds reproduce it "
        f"({pr('w_m3b_a40_s16 vs BAAI__bge-m3')['mean_new']:.2f}, {pr('w_m3b_a40_s17 vs BAAI__bge-m3')['mean_new']:.2f}). v2-base {p_base['mean_new']:.2f} vs e5-base {p_base['mean_ref']:.2f} ({p_base['mean_diff']:+.2f}, CI [{p_base['ci95_task_bootstrap'][0]:+.2f}, "
        f"{p_base['ci95_task_bootstrap'][1]:+.2f}], {p_base['wins']}/{p_base['losses']}/{p_base['ties']}, sign test p={p_base['sign_test_p_two_sided']:.3f}); v2-small {p_small['mean_new']:.2f} vs e5-small {p_small['mean_ref']:.2f} "
        f"({p_small['mean_diff']:+.2f}, CI [{p_small['ci95_task_bootstrap'][0]:+.2f}, {p_small['ci95_task_bootstrap'][1]:+.2f}]) and vs v1 {p_v1['mean_ref']:.2f} ({p_v1['mean_diff']:+.2f}, CI [{p_v1['ci95_task_bootstrap'][0]:+.2f}, {p_v1['ci95_task_bootstrap'][1]:+.2f}] *).",
        f"* **What it cost (guard tasks, MTEB(ara) nDCG/AP x100, v2-m3 vs bge-m3):** MIRACL-HN {g('final_m3', G[0]):.1f} vs {g('BAAI__bge-m3', G[0]):.1f}, NamaaMrTydi {g('final_m3', G[1]):.1f} vs {g('BAAI__bge-m3', G[1]):.1f}, "
        f"SemRel24STS {g('final_m3', G[2]):.1f} vs {g('BAAI__bge-m3', G[2]):.1f}. The drift on the Arabic-only reranking task (NamaaMrTydi, {g('final_m3', G[1]) - g('BAAI__bge-m3', G[1]):+.1f}) is **not** fixed; "
        f"the students lose much more there (v2-base {g('final_base', G[1]):.1f} vs e5-base {g('intfloat__multilingual-e5-base', G[1]):.1f}, v2-small {g('final_small', G[1]):.1f} vs e5-small {g('intfloat__multilingual-e5-small', G[1]):.1f}).",
        f"* **With the fine-tuned cross-encoder on top (top-50 -> `rr_ft_bce_s13`)** the first-stage gap closes: ar->en {f1(rr('final_m3', 'ar->en'))} (v2-m3) vs {f1(rr(BGE, 'ar->en'))} (bge-m3), macro {f1(rr('final_m3', 'macro'))} vs {f1(rr(BGE, 'macro'))}; "
        f"for the small models the first stage still matters: v2-base {f1(rr('st_b_soup', 'ar->en'))} vs e5-base {f1(rr(E5B, 'ar->en'))}, v2-small {f1(rr('st_s_soup', 'ar->en'))} vs e5-small {f1(rr(E5S, 'ar->en'))}.",
    ]
    return "\n".join(lines)


def scorecard() -> str:
    b, p, G = mt(), pr("final_m3 vs BAAI__bge-m3"), ["MIRACLRetrievalHardNegatives.v2", "NamaaMrTydiReranking", "SemRel24STS"]
    g = lambda m, t: b[m]["per_task"][t]
    x = lambda m: b[m]["ar-en cross-lingual subsets"]
    rows = [
        (
            "1. ar->en >= bge-m3 with macro > bge-m3 (TEST)",
            f"v2-m3 {f1(tv('final_m3', 'ar->en'))} vs {f1(tv(BGE, 'ar->en'))} ({bd(BGE, 'final_m3', 'ar->en')}); macro {f1(tv('final_m3', 'macro'))} vs {f1(tv(BGE, 'macro'))}",
            "**met** (v2-m3, v2-base); v2-small misses ar->en (59.7 < 63.3) but beats e5-small and v1",
        ),
        (
            "2a. no forgetting on MIRACL-HN / NamaaMrTydi / SemRel24 (within ~1 pt of the base)",
            f"MIRACL-HN {g('final_m3', G[0]) - g('BAAI__bge-m3', G[0]):+.1f}, NamaaMrTydi {g('final_m3', G[1]) - g('BAAI__bge-m3', G[1]):+.1f}, SemRel24 {g('final_m3', G[2]) - g('BAAI__bge-m3', G[2]):+.1f}",
            "**partly**: MIRACL-HN at the edge of the 1-point tolerance, SemRel24 up, **NamaaMrTydi missed (-3.4)**",
        ),
        (
            "2b. MTEB(ara) Mean (Task) > base",
            f"{p['mean_new']:.2f} vs {p['mean_ref']:.2f} ({p['mean_diff']:+.2f}, CI [{p['ci95_task_bootstrap'][0]:+.2f}, {p['ci95_task_bootstrap'][1]:+.2f}])",
            "numerically yes, **not statistically distinguishable**; reproduced by both confirmatory seeds",
        ),
        (
            "3a. #1 on MTEB(ara) among <=300M models",
            f"v2-base {b['final_base']['Mean(Task)']:.2f} vs e5-base {b['intfloat__multilingual-e5-base']['Mean(Task)']:.2f} (the best other <=300M model of the 10-model local board)",
            "**met** for v2-base; v2-small is above e5-small and v1 but below v2-base and e5-base",
        ),
        (
            "3b. #1 on the ar-en cross-lingual subsets",
            f"v2-m3 {x('final_m3'):.1f}, bge-m3 {x('BAAI__bge-m3'):.1f}, **e5-large {x('intfloat__multilingual-e5-large'):.1f}**; v2-base {x('final_base'):.1f} vs e5-base {x('intfloat__multilingual-e5-base'):.1f}",
            "**met within <=300M** (v2-base); **NOT met overall**: multilingual-e5-large (560M) is 0.3 higher than v2-m3",
        ),
        (
            "4. contamination audit + licences documented",
            "section 2 (audit table, licence table), section 8 (exact-match leak check)",
            "**met with one disclosed exception**: 9 kept training rows equal MIRACL-HN dev queries exactly (section 8); the model licence is TBD (user decision: MIT or CC BY-SA 4.0)",
        ),
        (
            "5. full-pipeline numbers with the fine-tuned reranker",
            "section 9",
            "**reported**; with the reranker v2-m3 and bge-m3 are level",
        ),
    ]
    return "\n".join(["| bar item | measured | status |", "|---|---|---|"] + [f"| {a} | {b_} | {c} |" for a, b_, c in rows])


def seed_table() -> str:
    b = mt()
    rows = [
        ("development seed 13 (= final_m3)", "final_m3", "final_m3", "m3_b_s13"),
        ("development seed 14", "w_m3b_a40_s14", None, "m3_b_s14"),
        ("development seed 15", "w_m3b_a40_s15", None, "m3_b_s15"),
        ("**confirmatory seed 16**", "w_m3b_a40_s16", "w_m3b_a40_s16", "m3_b_s16"),
        ("**confirmatory seed 17**", "w_m3b_a40_s17", "w_m3b_a40_s17", "m3_b_s17"),
    ]
    lines = [
        "| role | model (WiSE 0.4) | TEST ar->en | en->ar | ar->ar | en->en | macro | MTEB(ara) Mean (Task) | minus bge-m3 | same seed, RAW fine-tune: TEST ar->en / macro |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lab, n, mk, raw in rows:
        r = R.load(n, "test")["directions"]
        m = f"{b[mk]['Mean(Task)']:.2f}" if mk and mk in b else "not scored"
        dd = f"{b[mk]['Mean(Task)'] - b['BAAI__bge-m3']['Mean(Task)']:+.2f}" if mk and mk in b else ""
        lines.append(
            f"| {lab} | `{n}` | {r['ar->en']['mrr@10'] * 100:.1f} | {r['en->ar']['mrr@10'] * 100:.1f} | {r['ar->ar']['mrr@10'] * 100:.1f} | {r['en->en']['mrr@10'] * 100:.1f} | {r['macro']['mrr@10'] * 100:.1f} | {m} | {dd} | "
            f"{tv(raw, 'ar->en'):.1f} / {tv(raw, 'macro'):.1f} |"
        )
    return "\n".join(lines)


def leak_block() -> str:
    d = jl(V2RES / "leak_exact.json")
    if not d:
        return "(leak check not run)"
    names = {"miracl_hn_v2_ar_dev": "MIRACL-HN.v2 ar dev", "namaa_mrtydi_test": "NamaaMrTydi test"}
    lines = [
        "| evaluation set | queries (distinct after normalisation) | KEPT training rows that equal one of its queries | distinct questions | training slice | translated copies of those rows (via `src_qid`) in the kept training data |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for k, t in d["tasks"].items():
        lines.append(
            f"| {names.get(k, k)} | {t['eval_queries']} ({t['eval_queries_normalised_distinct']}) | {t['exact_matches_in_kept_training_rows']} | {t['distinct_questions']} | "
            f"{', '.join(f'{a} {b}' for a, b in t['by_slice'].items()) or '-'} | {t['translated_copies_of_those_rows_in_kept_training_rows']} |"
        )
    leaked = set(d["tasks"]["miracl_hn_v2_ar_dev"]["leaked_eval_qids"])
    res = {}
    gdir = V2RES / "guard_split"
    for p in sorted(gdir.glob("*.json")):
        if p.stem.startswith("_"):
            continue
        sc = json.loads(p.read_text())["miracl"]
        allv = np.mean(list(sc.values())) * 100
        rest = [v for q, v in sc.items() if q not in leaked]
        res[p.stem] = (allv, float(np.mean(rest)) * 100, len(sc) - len(rest))
    lines += [
        "",
        f"MIRACL-HN.v2 ar (our nDCG@10 over the full corpus, with titles) with and without the {len(leaked)} evaluation queries that have an exact copy in the kept training data:",
        "",
        "| model | all queries | without the leaked queries | difference |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {m} | {a:.1f} | {r:.1f} | {r - a:+.1f} |" for m, (a, r, n) in res.items()]
    return "\n".join(lines)


def guard_ci() -> str:
    """Per-query paired differences on the CLEAN / CONTAMINATED subsets of the guard tasks (from the cached per-query scores and the corrected labels)."""
    gdir = V2RES / "guard_split"
    lab = jl(gdir / "_labels.json")
    if not lab:
        return "(not run)"
    rng = np.random.RandomState(13)
    rows = [
        "| task | subset | n | v2-m3 minus bge-m3 [95% CI] | v2-base minus e5-base | v2-small minus e5-small |",
        "|---|---|---:|---|---|---|",
    ]
    cache: dict[str, dict] = {}
    get = lambda m: cache.setdefault(m, json.loads((gdir / f"{m}.json").read_text()))  # noqa: E731
    for key, tname in (("miracl", "MIRACL-HN.v2 ar"), ("namaa", "NamaaMrTydi")):
        for subset in ("clean", "contaminated"):
            qs = [q for q, v in lab[key].items() if (not v) == (subset == "clean")]
            cells = []
            for a, b_ in (("final_m3", "bge-m3"), ("final_base", "e5-base"), ("final_small", "e5-small")):
                A, B_ = get(a)[key], get(b_)[key]
                d = np.array([A[q] - B_[q] for q in qs if q in A and q in B_]) * 100
                boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
                cells.append(f"{d.mean():+.1f} [{np.quantile(boots, 0.025):+.1f}, {np.quantile(boots, 0.975):+.1f}]")
            rows.append(f"| {tname} | {subset} | {len(qs)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else V2 / "RESULTS_v2.md")
    tpl = (V2 / "RESULTS_v2.final.template.md").read_text()
    gs = V2RES / "guard_split" / "_report.md"
    reg = dict(
        GUARD_CI=guard_ci,
        LEAK=leak_block,
        SEEDS=seed_table,
        TLDR=tldr,
        SCORECARD=scorecard,
        HEADLINE=headline,
        BOOT=boot,
        MIXED=mixed,
        DEV_P=B.dev_table,
        DEV_M3=lambda: dev_block(M3_ITEMS),
        DEV_ST=lambda: dev_block(ST_ITEMS),
        SUITE=suite_table,
        BOARD=board,
        PER_TASK=per_task,
        PAIRED=paired,
        ALPHA=alpha,
        RERANK=rerank,
        COST=cost,
        TRAIL=trail,
        AUDIT=B.audit_table,
        STYLE=B.style_table,
        GUARD_SPLIT=lambda: gs.read_text() if gs.exists() else "(not run)",
    )
    for k, f in reg.items():
        tpl = tpl.replace("{{" + k + "}}", f())
    out.write_text(tpl)
    print("wrote", out, len(tpl.splitlines()), "lines")


if __name__ == "__main__":
    main()
