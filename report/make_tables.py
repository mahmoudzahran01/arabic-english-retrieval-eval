"""Write every table and every number of the technical report from the committed result files.

    python report/make_tables.py            # writes report/tables/*.tex

Inputs (all committed):  results/retrieval_v2/   v2 test/dev results + per-query MRR arrays, manifests, contamination split
                         results/mteb_ara/       MTEB(ara, v1) leaderboard + guard-task JSONs
                         deploy/results/         ONNX/int8 quality and latency measurements
Nothing in report/main.tex is typed by hand: numbers reach the text through \\newcommand macros in tables/numbers.tex, and anything
that is still pending is printed as a red \\TODO{...} so that it cannot be mistaken for a result.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from report.study_tables import CONF_DEFAULTS, CURVE_DEFAULTS, PAIRED_DEFAULTS, STUDY_DEFAULTS, StudyTables  # noqa: E402
from report.tabcommon import (  # noqa: E402,F401
    DEPLOY,
    DIR_TEX,
    DIRS,
    MTEB,
    V2,
    dir_value,
    esc,
    load_json,
    load_system,
    pct,
    spct,
    thousands,
    todo,
)

from deploy import summarize as DS  # noqa: E402

OUT = ROOT / "report" / "tables"
DS_THREADS_LOW, DS_THREADS_HIGH = 2, 8

# key -> (name in the table, result file stem)
SYSTEMS = {
    "e5s": ("mE5-small", "base_e5-small"),
    "v1": ("v1 (e5-small FT)", "v1_s13"),
    "s_nokd": ("e5-small FT, no KD", "f4_s_e1_s13"),
    "v2s": ("v2-small (KD)", "f5_s_kd_s13"),
    "b_nokd": ("e5-base FT, no KD", "f4_b_e1_s13"),
    "v2b": ("v2-base (KD)", "f5_b_kd_s13"),
    "m3": ("bge-m3", "base_bge-m3"),
    "v2m_raw": ("v2-m3 raw FT", "m3_b_s13"),
    "v2m": ("v2-m3 (WiSE 0.75)", "wm_a75"),
}
GROUPS = [["e5s", "v1", "s_nokd", "v2s"], ["b_nokd", "v2b"], ["m3", "v2m_raw", "v2m"]]
DELTAS = [  # label, A (baseline), B (new)
    ("v1 vs mE5-small", "e5s", "v1"),
    ("v2-small vs v1", "v1", "v2s"),
    ("KD effect, small", "s_nokd", "v2s"),
    ("KD effect, base", "b_nokd", "v2b"),
    ("v2-m3 vs bge-m3", "m3", "v2m"),
    ("WiSE cost (raw to 0.75)", "v2m_raw", "v2m"),
    ("v2-base vs bge-m3", "m3", "v2b"),
]
DEPLOY_VARIANTS = {"torch-fp32": "Torch", "onnx-fp32": "Onnx", "onnx-int8": "Dyn", "onnx-int8-pc": "Pc", "onnx-w8": "Wo"}
DEPLOY_MODELS = {"v2-small": "Small", "v2-base": "Base"}
PENDING_DEFAULTS = [  # macros used in main.tex whose values come from runs that may not have finished
    "NumThreadsLow",
    "NumThreadsHigh",
    "NumSparkCpus",
    "NumOpset",
    *[f"Num{kind}{m}{v}" for kind in ("Size", "QLat") for m in DEPLOY_MODELS.values() for v in DEPLOY_VARIANTS.values()],
    *[f"NumDelta{v}{m}{suffix}" for v in ("Onnx", "Dyn", "Pc", "Wo") for m in DEPLOY_MODELS.values() for suffix in ("", "ArEn")],
]
GUARD_MODELS = [("mE5-small", "e5-small"), ("v1", "v1"), ("mE5-base", "e5-base"), ("bge-m3", "bge-m3"), ("v2-m3 (WiSE 0.75)", "m3w75")]
GUARD_TASKS = [("MIRACLRetrievalHardNegatives.v2", "MIRACL-HN"), ("NamaaMrTydiReranking", "Mr.TyDi rerank"), ("SemRel24STS", "SemRel24")]


class Tables(StudyTables):
    def __init__(self) -> None:
        self.dir_of = DS.default_direction_map()
        self.sys = {k: load_system(stem) for k, (_, stem) in SYSTEMS.items()}
        self.nums: dict[str, str] = {}
        self.ci_cache: dict[str, dict] = {}

    def num(self, name: str, value: str) -> None:
        self.nums[name] = value

    # -------------------------------------------------------------------------------------------- main table
    def ci_single(self, key: str) -> dict | None:
        s = self.sys[key]
        if s is None or not s["perq"]:
            return None
        if key not in self.ci_cache:
            self.ci_cache[key] = DS.macro_bootstrap(s["perq"], None, self.dir_of)
        return self.ci_cache[key]

    def tab_main(self) -> str:
        best = {d: max(dir_value(s, d) for s in self.sys.values() if s) for d in (*DIRS, "macro")}
        rows = []
        for gi, group in enumerate(GROUPS):
            if gi:
                rows.append(r"\midrule")
            for k in group:
                s = self.sys[k]
                name = SYSTEMS[k][0]
                if s is None:
                    rows.append(f"{esc(name)} & \\multicolumn{{6}}{{c}}{{{todo('result missing')}}} \\\\")
                    continue
                cells = []
                for d in (*DIRS, "macro"):
                    v = dir_value(s, d)
                    c = pct(v)
                    cells.append(rf"\textbf{{{c}}}" if abs(v - best[d]) < 5e-5 else c)
                ci = self.ci_single(k)
                cells.append(f"[{pct(ci['macro']['lo'])}, {pct(ci['macro']['hi'])}]" if ci else todo("CI"))
                rows.append(f"{esc(name)} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
        rows.append(rf"mE5-base (not fine-tuned) & \multicolumn{{6}}{{c}}{{{todo('baseline not yet scored on the TEST split')}}} \\")
        head = " & ".join(DIR_TEX[d] for d in (*DIRS, "macro")) + r" & 95\% CI (macro)"
        return (
            "\\begin{tabular}{@{}l"
            + "r" * 5
            + "c@{}}\n\\toprule\nModel & "
            + head
            + " \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def tab_deltas(self) -> str:
        rows = []
        for label, a, b in DELTAS:
            sa, sb = self.sys[a], self.sys[b]
            if not sa or not sb or not sa["perq"] or not sb["perq"]:
                rows.append(f"{esc(label)} & \\multicolumn{{4}}{{c}}{{{todo('result missing')}}} \\\\")
                continue
            ci = DS.macro_bootstrap(sa["perq"], sb["perq"], self.dir_of)
            cells = []
            for d in ("ar->en", "macro"):
                delta = dir_value(sb, d) - dir_value(sa, d)
                cells += [spct(delta), f"[{spct(ci[d]['lo'])}, {spct(ci[d]['hi'])}]"]
            rows.append(f"{esc(label)} & " + " & ".join(cells) + r" \\")
            self.num("Delta" + re.sub(r"[^A-Za-z]", "", label.title()), spct(dir_value(sb, "ar->en") - dir_value(sa, "ar->en")))
        return (
            "\\begin{tabular}{@{}lrcrc@{}}\n\\toprule\n & \\multicolumn{2}{c}{$\\Delta$ "
            + DIR_TEX["ar->en"]
            + "} & \\multicolumn{2}{c}{$\\Delta$ macro} \\\\\n"
            "Comparison (B vs A) & pts & 95\\% CI & pts & 95\\% CI \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # -------------------------------------------------------------------------------------------- ablations (dev)
    def tab_ablation(self) -> str:
        def row(label: str, stem: str) -> str:
            j = load_json(V2 / f"{stem}.dev.json")
            if j is None:
                return f"{esc(label)} & \\multicolumn{{3}}{{c}}{{{todo('missing')}}} \\\\"
            dm = load_json(V2 / f"{stem}.devM.json") or (load_json(V2 / "bge-m3.devM.json") if stem == "base_bge-m3" else None)
            m = j["directions"]
            devm = pct(dm["slices"]["dev_miracl_ar"]["mrr@10"]) if dm else "--"
            return f"{esc(label)} & {pct(m['ar->en']['mrr@10'])} & {pct(m['macro']['mrr@10'])} & {devm} \\\\"

        blocks = [
            (
                "e5-base student (teacher: raw FT bge-m3)",
                [("no KD", "f4_b_e1_s13"), ("KD $w{=}1$", "f5_b_kd_s13"), ("KD $w{=}3$", "f5_b_kd3_s13"), ("KD only", "f5_b_kdonly_s13")],
            ),
            (
                "bge-m3 fine-tuned itself",
                [
                    ("bge-m3 (no FT)", "base_bge-m3"),
                    ("raw FT", "m3_b_s13"),
                    ("WiSE $\\alpha{=}0.75$", "wm_a75"),
                    ("WiSE $\\alpha{=}0.50$", "wm_a50"),
                ],
            ),
            (
                "bge-m3 + MIRACL-ar replay / preservation",
                [("replay", "m3_r_s13"), ("preservation", "m3_p_s13"), ("preservation, WiSE 0.75", "w_m3_p_s13_a75")],
            ),
        ]
        out = [
            "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nDEV selection set & "
            + DIR_TEX["ar->en"]
            + " & macro & held-out MIRACL-ar \\\\\n\\midrule"
        ]
        for bi, (title, items) in enumerate(blocks):
            if bi:
                out.append(r"\midrule")
            out.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
            out += [row(lbl, stem) for lbl, stem in items]
        out.append("\\bottomrule\n\\end{tabular}\n")
        bge = load_json(V2 / "bge-m3.devM.json")
        if bge:
            self.num("NumDevMBge", pct(bge["slices"]["dev_miracl_ar"]["mrr@10"]))
        return "\n".join(out)

    # -------------------------------------------------------------------------------------------- seeds / data volume
    def tab_seeds(self) -> str:
        rows, vals = [], []
        for seed in (13, 14, 15):
            s = load_system(f"m3_b_s{seed}")
            if s:
                vals.append((dir_value(s, "ar->en"), dir_value(s, "macro")))
                rows.append(f"v2-m3 raw FT, seed {seed} & {pct(vals[-1][0])} & {pct(vals[-1][1])} \\\\")
            else:
                rows.append(f"v2-m3 raw FT, seed {seed} & \\multicolumn{{2}}{{c}}{{{todo('run pending')}}} \\\\")
        if len(vals) == 3:
            a = np.array(vals)
            rows.append(r"\midrule")
            rows.append(
                f"mean $\\pm$ s.d. (3 seeds) & {pct(a[:, 0].mean())} $\\pm$ {pct(a[:, 0].std(ddof=1))} & {pct(a[:, 1].mean())} $\\pm$ {pct(a[:, 1].std(ddof=1))} \\\\"
            )
            self.num("NumSeedRangeArEn", pct(a[:, 0].max() - a[:, 0].min()))
            self.num("NumSeedRangeMacro", pct(a[:, 1].max() - a[:, 1].min()))
            self.num("NumSeedSdArEn", pct(a[:, 0].std(ddof=1)))
        rows.append(r"\midrule")
        rows.append(f"v2-m3 WiSE 0.75, seeds 14 and 15 & \\multicolumn{{2}}{{c}}{{{todo('blends pending')}}} \\\\")
        rows.append(f"v2-base / v2-small, seeds 14 and 15 & \\multicolumn{{2}}{{c}}{{{todo('runs pending')}}} \\\\")
        return (
            "\\begin{tabular}{@{}lrr@{}}\n\\toprule\nTEST & "
            + DIR_TEX["ar->en"]
            + " & macro \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def tab_volume(self) -> str:
        pairs = [("e5-small", "p3_s_e1_s13", "f4_s_e1_s13"), ("e5-base", "p3_b_e1_s13", "f4_b_e1_s13"), ("bge-m3", "m3_a_s13", "m3_b_s13")]
        rows = []
        for name, small, full in pairs:
            a, b = load_system(small), load_system(full)
            if not a or not b:
                rows.append(f"{esc(name)} & \\multicolumn{{4}}{{c}}{{{todo('missing')}}} \\\\")
                continue
            rows.append(
                f"{esc(name)} & {pct(dir_value(a, 'ar->en'))} & {pct(dir_value(b, 'ar->en'))} & {pct(dir_value(a, 'macro'))} & {pct(dir_value(b, 'macro'))} \\\\"
            )
        return (
            "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n & \\multicolumn{2}{c}{"
            + DIR_TEX["ar->en"]
            + "} & \\multicolumn{2}{c}{macro} \\\\\n"
            "Student & partial & full & partial & full \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # -------------------------------------------------------------------------------------------- MTEB(ara)
    def tab_mteb(self) -> str:
        lb = load_json(MTEB / "leaderboard.json")
        if lb is None:
            return todo("leaderboard.json missing")
        cols = [
            ("Mean (Task)", "Task"),
            ("Mean (TaskType)", "Type"),
            ("Retrieval", "Retr."),
            ("Reranking", "Rerank"),
            ("STS", "STS"),
            ("Classification", "Cls."),
            ("Mean ar-only subsets", "ar"),
            ("Mean ar-en cross-lingual subsets", "ar-en"),
        ]
        rows = []
        for r in lb["summary"]:
            name = r["model_name"].split("/")[-1]
            rows.append(esc(name) + " & " + " & ".join(f"{r[c]:.1f}" for c, _ in cols) + r" \\")
        pend = " & ".join([todo("pending")] + [""] * (len(cols) - 1))
        extra = self.mteb_extra_rows()  # v2 models with per-task scores in mteb_paired.json
        for name in ("v2-m3", "v2-base", "v2-small"):
            if not any(r.startswith(name) for r in extra):
                extra.append(f"{name} & {pend} \\\\")
        top = lb["summary"][0]
        self.num("NumMtebTasks", str(top["n_tasks"]))
        self.num("NumMtebModels", str(len(lb["summary"])))
        for r in lb["summary"]:
            if r["model_name"].endswith("bge-m3"):
                self.num("NumMtebBgeTask", f"{r['Mean (Task)']:.1f}")
            if r["model_name"].startswith("mzahran1/"):
                self.num("NumMtebVOneTask", f"{r['Mean (Task)']:.1f}")
        return (
            "\\begin{tabular}{@{}l"
            + "r" * len(cols)
            + "@{}}\n\\toprule\nModel & "
            + " & ".join(h for _, h in cols)
            + " \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\midrule\n"
            + "\n".join(extra)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # -------------------------------------------------------------------------------------------- guards
    def guard_scores(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for _, folder in GUARD_MODELS:
            out[folder] = {}
            for task, _ in GUARD_TASKS:
                j = load_json(MTEB / "guards" / folder / f"{task}.json")
                if j is not None:
                    split = next(iter(j["scores"]))
                    sub = next(iter(j["scores"][split]))
                    out[folder][task] = j["scores"][split][sub][j["main_score"]]
        return out

    def tab_guards(self) -> str:
        g = self.guard_scores()
        rows = []
        for label, folder in GUARD_MODELS:
            cells = [pct(g[folder][t]) if t in g[folder] else todo("missing") for t, _ in GUARD_TASKS]
            rows.append(f"{esc(label)} & " + " & ".join(cells) + r" \\")
        if all(t in g["m3w75"] and t in g["bge-m3"] for t, _ in GUARD_TASKS):
            rows.append(r"\midrule")
            rows.append(r"v2-m3 $-$ bge-m3 & " + " & ".join(spct(g["m3w75"][t] - g["bge-m3"][t]) for t, _ in GUARD_TASKS) + r" \\")
            self.num("NumGuardMiraclGap", spct(g["m3w75"][GUARD_TASKS[0][0]] - g["bge-m3"][GUARD_TASKS[0][0]]))
            self.num("NumGuardNamaaGap", spct(g["m3w75"][GUARD_TASKS[1][0]] - g["bge-m3"][GUARD_TASKS[1][0]]))
            self.num("NumGuardSemRelGap", spct(g["m3w75"][GUARD_TASKS[2][0]] - g["bge-m3"][GUARD_TASKS[2][0]]))
        # the guards were re-run by the v2 pipeline (fp16, batch 32); how far are they from the benchmark's own leaderboard run?
        lb = load_json(MTEB / "leaderboard.json")
        if lb:
            by_model = {r["model_name"].split("/")[-1]: r for r in lb["per_task"]}
            lb_name = {
                "e5-small": "multilingual-e5-small",
                "e5-base": "multilingual-e5-base",
                "bge-m3": "bge-m3",
                "v1": "arabic-english-e5-small-retrieval",
            }
            diffs = [
                abs(100 * g[f][t] - by_model[lb_name[f]][t])
                for f in lb_name
                for t, _ in GUARD_TASKS
                if t in g.get(f, {}) and t in by_model.get(lb_name[f], {})
            ]
            if diffs:
                self.num("NumGuardVsLeaderboardMax", f"{max(diffs):.1f}")
        short = {"MIRACL-HN": "Miracl", "Mr.TyDi rerank": "Namaa", "SemRel24": "SemRel"}
        names = {"e5-small": "ESmall", "v1": "VOne", "e5-base": "EBase", "bge-m3": "Bge", "m3w75": "VTwoM"}
        for folder, nm in names.items():
            for t, tshort in GUARD_TASKS:
                if t in g.get(folder, {}):
                    self.num(f"NumGuard{nm}{short[tshort]}", pct(g[folder][t]))
        head = " & ".join(esc(s) for _, s in GUARD_TASKS)
        return (
            "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nModel & "
            + head
            + " \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def clean_split(self) -> tuple[str, dict]:
        labels = load_json(V2 / "guard_split" / "_labels.json")
        if labels is None:
            return todo("guard_split missing"), {}
        models = [
            ("bge-m3", "bge-m3"),
            ("mE5-base", "e5-base"),
            ("v1", "v1"),
            ("v2-m3 raw FT", "m3_b_s13"),
            ("v2-m3 (WiSE 0.75)", "wm_a75"),
        ]
        rows, stats = [], {}
        for label, stem in models:
            j = load_json(V2 / "guard_split" / f"{stem}.json")
            if j is None:
                rows.append(f"{esc(label)} & \\multicolumn{{5}}{{c}}{{{todo('missing')}}} \\\\")
                continue
            cells = []
            for task in ("miracl", "namaa"):
                lab = labels[task]
                ids = list(j[task])
                v = np.array([j[task][i] for i in ids])
                clean = np.array([lab[i] == "" for i in ids])
                cells += [pct(v.mean()), pct(v[clean].mean()), pct(v[~clean].mean()) if (~clean).any() else "--"]
                n_clean, n_contam = int(clean.sum()), int((~clean).sum())
                stats[(stem, task)] = (v.mean(), v[clean].mean(), n_clean, n_contam)
                stats[(task, "n")] = (n_clean, n_contam)
            rows.append(f"{esc(label)} & " + " & ".join(cells) + r" \\")
        n = {t: stats.get((t, "n")) for t in ("miracl", "namaa")}
        head2 = rf"\multicolumn{{3}}{{c}}{{MIRACL-HN (n={n['miracl'][0]}+{n['miracl'][1]})}} & \multicolumn{{3}}{{c}}{{Mr.TyDi rerank (n={n['namaa'][0]}+{n['namaa'][1]})}} \\"
        tab = (
            "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & " + head2 + "\n"
            "Model & all & clean & contam. & all & clean & contam. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        )
        return tab, stats

    def numbers_contamination(self, stats: dict) -> None:
        labels = load_json(V2 / "guard_split" / "_labels.json")
        audit = load_json(V2 / "data" / "audit_ood.json")
        filt = load_json(V2 / "data" / "audit_ood_filtered.json")
        if audit:
            a = audit["MIRACLRetrievalHardNegatives.v2/ar/dev"]
            self.num("NumMiraclQueries", thousands(a["queries"]))
            self.num("NumMiraclNearDup", thousands(a["queries_near_duplicate_of_a_train_query"]))
            self.num("NumMiraclNearDupPct", f"{100 * a['queries_near_duplicate_of_a_train_query'] / a['queries']:.0f}")
        if filt:
            self.num(
                "NumMiraclNearDupAfter", str(filt["MIRACLRetrievalHardNegatives.v2/ar/dev"]["queries_near_duplicate_of_a_train_query"])
            )
        if labels:
            m, nm = labels["miracl"], labels["namaa"]
            self.num("NumMiraclContamFull", thousands(sum(v != "" for v in m.values())))
            self.num("NumMiraclCleanN", thousands(sum(v == "" for v in m.values())))
            self.num("NumNamaaContam", str(sum(v != "" for v in nm.values())))
            self.num("NumNamaaQueries", thousands(len(nm)))
        # honest flag: the contamination-split run measured absolute scores below the MTEB protocol's (passage titles missing)
        g = self.guard_scores()
        bge_all = stats.get(("bge-m3", "miracl"), (None,))[0]
        official = g.get("bge-m3", {}).get("MIRACLRetrievalHardNegatives.v2")
        if bge_all is not None and official is not None and abs(bge_all - official) > 0.01:
            self.num(
                "TodoCleanProtocol",
                todo(f"split scores are {pct(official - bge_all)} pts below the MTEB protocol for bge-m3 (titles missing); rerun pending"),
            )
        else:
            self.num("TodoCleanProtocol", "")
        clean = {k: v for k, v in stats.items() if k[1] == "miracl" and k[0] in ("bge-m3", "wm_a75")}
        if len(clean) == 2:
            self.num("NumCleanGapMiracl", spct(clean[("wm_a75", "miracl")][1] - clean[("bge-m3", "miracl")][1]))
            self.num("NumAllGapSplitMiracl", spct(clean[("wm_a75", "miracl")][0] - clean[("bge-m3", "miracl")][0]))

    # -------------------------------------------------------------------------------------------- data
    def tab_data(self) -> str:
        ex = load_json(V2 / "data" / "exclude_summary.json")
        if ex is None:
            return todo("exclude_summary.json missing")
        by = ex["by_slice"]
        v2_slices = [s for s in by if "_xl3_" not in s and not s.endswith("_squad_en_x") and s != "train_miracl_ar"]
        fam = {
            "Human questions (TyDi, ARCD, SQuAD)": ["train_tydi_ar", "train_tydi_en", "train_arcd_ar", "train_squad_en"],
            "Translated, v1": ["train_xl_ar2en", "train_xl_en2ar"],
            "Translated, new in v2": ["train_xl2_ar2en", "train_xl2_en2ar"],
        }
        rows, tot_n, tot_d = [], 0, 0
        for label, sl in fam.items():
            n, d = sum(by[s]["n"] for s in sl), sum(by[s]["dropped"] for s in sl)
            tot_n, tot_d = tot_n + n, tot_d + d
            rows.append(f"{esc(label)} & {thousands(n)} & {thousands(d)} & {thousands(n - d)} \\\\")
        assert set(v2_slices) == {s for sl in fam.values() for s in sl}
        rows += [r"\midrule", f"Total (v2 recipe) & {thousands(tot_n)} & {thousands(tot_d)} & {thousands(tot_n - tot_d)} \\\\"]
        self.num("NumTrainTotal", thousands(tot_n))
        self.num("NumTrainDropped", thousands(tot_d))
        self.num("NumTrainKept", thousands(tot_n - tot_d))
        self.num("NumXlOld", thousands(sum(by[s]["n"] for s in fam["Translated, v1"])))
        self.num("NumXlNew", thousands(sum(by[s]["n"] for s in fam["Translated, new in v2"])))
        self.num("NumEvalTasksAudited", str(len(ex["eval_tasks"])))
        return (
            "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nTraining queries & built & dropped & kept \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def numbers_licences(self) -> None:
        m1 = load_json(V2 / "data" / "manifest_v1.json")
        if m1:
            lic = m1["licenses"]
            self.num("LicTydi", esc(lic["tydi"].split(" (")[0]))
            self.num("LicSquad", esc(lic["squad"].split(" (")[0]))
            self.num("LicArcd", esc(lic["arcd"].split(" (")[0]))
            self.num("LicXquad", esc(lic["xquad"].split(" (")[0]))
            pools = m1["passages_by_pool_lang"]
            self.num("NumTestPassages", thousands(pools["test/ar"] + pools["test/en"]))
            qs = m1["queries_by_slice"]
            self.num("NumTestQueries", thousands(sum(v for k, v in qs.items() if k.startswith("test/"))))

    # -------------------------------------------------------------------------------------------- headline numbers
    def numbers_headline(self) -> None:
        names = {
            "e5s": "ESmall",
            "v1": "VOne",
            "v2s": "VTwoS",
            "v2b": "VTwoB",
            "m3": "Bge",
            "v2m_raw": "VTwoMRaw",
            "v2m": "VTwoM",
            "b_nokd": "BNoKd",
            "s_nokd": "SNoKd",
        }
        for k, nm in names.items():
            s = self.sys[k]
            if s is None:
                continue
            for d, dn in (("ar->en", "ArEn"), ("en->ar", "EnAr"), ("ar->ar", "ArAr"), ("en->en", "EnEn"), ("macro", "Macro")):
                self.num(f"Num{nm}{dn}", pct(dir_value(s, d)))
            ci = self.ci_single(k)
            if ci:
                self.num(f"Num{nm}MacroLo", pct(ci["macro"]["lo"]))
                self.num(f"Num{nm}MacroHi", pct(ci["macro"]["hi"]))
        if self.sys["m3"] and self.sys["v2m"]:
            self.num("NumGainArEn", spct(dir_value(self.sys["v2m"], "ar->en") - dir_value(self.sys["m3"], "ar->en")))
            self.num("NumGainEnAr", spct(dir_value(self.sys["v2m"], "en->ar") - dir_value(self.sys["m3"], "en->ar")))
        if self.sys["m3"] and self.sys["v1"]:
            self.num("NumGapVOneArEn", pct(dir_value(self.sys["m3"], "ar->en") - dir_value(self.sys["v1"], "ar->en")))
        # mixed-index diagnostic: how often the top-1 of a cross-lingual query is in the query's own language
        for k, nm in (("m3", "Bge"), ("v2m", "VTwoM")):
            j = load_json(V2 / f"{SYSTEMS[k][1]}.test.mixed.json")
            if j:
                self.num(f"Num{nm}OwnLangEnAr", pct(j["xquad_en2ar"]["wrong_lang_top1"], 0))
                self.num(f"Num{nm}OwnLangArEn", pct(j["xquad_ar2en"]["wrong_lang_top1"], 0))
        wm = load_json(V2 / "models" / "wm_a75" / "merge.json")
        if wm:
            self.num("NumWiseAlpha", f"{wm['alpha']:.2f}")
        for key, stem in (("M", "m3_b_s13"), ("B", "f5_b_kd_s13"), ("S", "f5_s_kd_s13")):
            a = (load_json(V2 / "models" / stem / "lab_retrieval.json") or {}).get("args")
            if a:
                self.num(f"NumLr{key}", f"{a['lr']:g}".replace("e-0", "e{-}").replace("e-", "e{-}"))
                self.num(f"NumBs{key}", str(a["bs"]))
                self.num(f"NumScale{key}", f"{a['scale']:g}")
                self.num(f"NumMaxLen{key}", str(a["max_len"]))
                self.num(f"NumTemp{key}", f"{a['t_scale']:g}")
        m = load_json(V2 / "models" / "m3_b_s13" / "lab_retrieval.json")
        if m:
            self.num("NumMiniBatch", str(m["args"]["mini_batch"]))

    def numbers_protocol(self) -> None:
        """Constants of the protocol that live in the lab notebook; read from the committed protocol.json (value + source)."""
        pr = load_json(V2 / "data" / "protocol.json") or {}
        names = {
            "contamination_jaccard_threshold": "NumJaccard",
            "wise_selection_dev_macro_tolerance": "NumWiseTolDev",
            "wise_selection_proxy_tolerance": "NumWiseTolProxy",
            "guard_bar_miracl_points": "NumGuardBar",
            "guard_bar_ar2en_test": "NumGuardBarArEn",
            "bootstrap_resamples": "NumBootstrap",
            "ci_level_percent": "NumCiLevel",
        }
        for key, macro in names.items():
            if key in pr:
                v = pr[key]["value"]
                self.num(macro, thousands(v) if isinstance(v, int) else f"{v:g}")
        # recipe constants stored with the models
        a = (load_json(V2 / "models" / "m3_b_s13" / "lab_retrieval.json") or {}).get("args", {})
        if a:
            self.num("NumMargin", f"{a['margin']:g}")
            self.num("NumRankLo", str(a["rank_lo"]))
            self.num("NumRankHi", str(a["rank_hi"]))
        k = (load_json(V2 / "models" / "f5_b_kd_s13" / "lab_retrieval.json") or {}).get("args", {})
        if k:
            self.num("NumKdWeight", f"{k['kd']:g}")

    def numbers_style(self) -> None:
        """Machine-translated XQuAD ar->en queries versus the human ones (same passages): a constant offset for every model?"""
        d = load_json(V2 / "data" / "diag_xquad_style.json")
        if not d:
            return
        off = [v["gemma"]["mrr@10"] - v["human"]["mrr@10"] for v in d.values()]
        self.num("NumStyleOffsetMin", pct(min(off)))
        self.num("NumStyleOffsetMax", pct(max(off)))
        self.num("NumStyleN", thousands(next(iter(d.values()))["human"]["n"]))

    # -------------------------------------------------------------------------------------------- deployment
    def tab_deploy(self) -> str:
        quality, bench = DS.load_quality(DEPLOY), DS.load_bench(DEPLOY)
        models = [m for m in ("v2-small", "v2-base", "v2-m3") if any(k[0] == m for k in quality)]
        if not models:
            return todo("deployment measurements pending")
        rows = []
        for mi, model in enumerate(models):
            if mi:
                rows.append(r"\midrule")
            qrows = {r["variant"]: r for r in DS.quality_rows(quality, model, self.dir_of)}
            for variant in DEPLOY_VARIANTS:
                if variant not in qrows:
                    continue
                r = qrows[variant]
                b8 = bench.get((model, variant, 8))
                b2 = bench.get((model, variant, 2))
                size = f"{b8['size_mb']:.0f}" if b8 else todo("size")
                rss = f"{b8['peak_rss_mb']:.0f}" if b8 else todo("RSS")
                q1 = f"{b2['workloads']['query_b1']['p50_ms']:.0f}" if b2 else todo("lat")
                p32 = f"{b8['workloads']['passage_b32']['throughput_sent_per_s']:.0f}" if b8 else todo("thr")
                mrr = pct(r["mrr"]["macro"])
                dm = "ref." if "delta" not in r else f"{spct(r['delta']['macro'], 2)}"
                da = "ref." if "delta" not in r else f"{spct(r['delta']['ar->en'], 2)}"
                rows.append(
                    f"{esc(model)} {esc(DS.VARIANT_LABEL[variant].replace(' (reference)', '').replace(' (dynamic)', ''))} & {size} & {rss} & {q1} & {p32} & {mrr} & {dm} & {da} \\\\"
                )
        self.deploy_numbers(quality, bench)
        return (
            "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\nVariant (CPU) & MiB & RSS & q@1 ms & p/s & MRR & $\\Delta$ & $\\Delta$ "
            + DIR_TEX["ar->en"]
            + " \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def deploy_numbers(self, quality: dict, bench: dict) -> None:
        threads = sorted({t for (_, _, t) in bench})
        if threads:
            self.num("NumThreadsLow", str(threads[0]))
            self.num("NumThreadsHigh", str(threads[-1]))
        for b in bench.values():
            m = re.search(r"\((\d+) logical", b.get("platform", ""))
            if m:
                self.num("NumSparkCpus", m.group(1))
                break
        for p in sorted((DEPLOY / "export").glob("*.json")):
            ex = load_json(p) or {}
            if "opset" in ex:
                self.num("NumOpset", str(ex["opset"]))
                break
        for model, nm in DEPLOY_MODELS.items():
            q = {r["variant"]: r for r in DS.quality_rows(quality, model, self.dir_of)}
            for v, vn in DEPLOY_VARIANTS.items():
                if v in q and "delta" in q[v]:
                    self.num(f"NumDelta{vn}{nm}", spct(q[v]["delta"]["macro"], 2))
                    self.num(f"NumDelta{vn}{nm}ArEn", spct(q[v]["delta"]["ar->en"], 2))
                b = bench.get((model, v, DS_THREADS_HIGH))
                if b:
                    self.num(f"NumSize{nm}{vn}", f"{b['size_mb']:.0f}")
                b2 = bench.get((model, v, DS_THREADS_LOW))
                if b2:
                    self.num(f"NumQLat{nm}{vn}", f"{b2['workloads']['query_b1']['p50_ms']:.0f}")

    # -------------------------------------------------------------------------------------------- qualitative claims
    def check_claims(self) -> list[tuple[str, bool | None]]:
        """The prose makes directional statements; each is re-checked against the numbers every time the tables are rebuilt.
        True = still holds, False = the text must be revised, None = inputs missing."""
        S = self.sys

        def ci(a: str, b: str, d: str) -> dict | None:
            if not (S[a] and S[b] and S[a]["perq"] and S[b]["perq"]):
                return None
            return DS.macro_bootstrap(S[a]["perq"], S[b]["perq"], self.dir_of)[d]

        def val(k: str, d: str) -> float | None:
            return dir_value(S[k], d) if S[k] else None

        out: list[tuple[str, bool | None]] = []
        c = ci("m3", "v2m", "ar->en")
        out.append(("v2-m3 beats bge-m3 on ar->en (paired CI above 0)", None if c is None else c["lo"] > 0))
        c = ci("m3", "v2m", "en->ar")
        out.append(("v2-m3 beats bge-m3 on en->ar (paired CI above 0)", None if c is None else c["lo"] > 0))
        c = ci("m3", "v2b", "ar->en")
        out.append(("v2-base vs bge-m3 on ar->en: paired CI includes 0", None if c is None else c["lo"] <= 0 <= c["hi"]))
        c = ci("m3", "v2b", "macro")
        out.append(("v2-base beats bge-m3 on macro (paired CI above 0)", None if c is None else c["lo"] > 0))
        for a, b, nm in (("b_nokd", "v2b", "base"), ("s_nokd", "v2s", "small")):
            c = ci(a, b, "ar->en")
            out.append((f"KD helps ar->en for the {nm} student (paired CI above 0)", None if c is None else c["lo"] > 0))
        a, b = val("v2m", "ar->en"), val("v2b", "ar->en")
        out.append(("the students stay below the v2-m3 teacher on ar->en", None if a is None or b is None else a > b > 0))
        c = ci("v2m_raw", "v2m", "macro")
        out.append(("WiSE 0.75 costs macro points relative to the raw fine-tune (CI below 0)", None if c is None else c["hi"] < 0))
        g = self.guard_scores()
        mi, na, se = (t for t, _ in GUARD_TASKS)
        try:
            out.append(("v2-m3 is more than the bar below bge-m3 on MIRACL-HN", g["m3w75"][mi] - g["bge-m3"][mi] < -0.01))
            out.append(("v2-m3 is below bge-m3 on Mr.TyDi rerank", g["m3w75"][na] < g["bge-m3"][na]))
            out.append(("v2-m3 is above bge-m3 on SemRel24", g["m3w75"][se] > g["bge-m3"][se]))
        except KeyError:
            out.append(("guard scores present", None))
        lb = load_json(MTEB / "leaderboard.json")
        if lb:
            names = [r["model_name"] for r in lb["summary"]]
            by = {r["model_name"].split("/")[-1]: r["Mean (Task)"] for r in lb["summary"]}
            out.append(("bge-m3 has the highest MTEB(ara) Mean (Task)", names[0].endswith("bge-m3")))
            out.append(
                (
                    "v1 scores below multilingual-e5-small on MTEB(ara) Mean (Task)",
                    by["arabic-english-e5-small-retrieval"] < by["multilingual-e5-small"],
                )
            )
        d = {
            k: load_json(V2 / f"{stem}.dev.json")
            for k, stem in (("nokd", "f4_b_e1_s13"), ("kd1", "f5_b_kd_s13"), ("kd3", "f5_b_kd3_s13"), ("kdonly", "f5_b_kdonly_s13"))
        }
        if all(d.values()):
            m = {k: v["directions"]["macro"]["mrr@10"] for k, v in d.items()}
            out.append(("dev: KD weight 3 and KD-only are at least as good as weight 1", m["kd3"] >= m["kd1"] and m["kdonly"] >= m["kd1"]))
        dm = {
            k: (load_json(V2 / f"{k}.devM.json") or {}).get("slices", {}).get("dev_miracl_ar", {}).get("mrr@10")
            for k in ("wm_a75", "m3_r_s13", "m3_p_s13", "w_m3_r_s13_a75", "w_m3_p_s13_a75")
        }
        if all(v is not None for v in dm.values()):
            out.append(
                (
                    "replay/preservation do not beat plain WiSE 0.75 on held-out MIRACL-ar by more than 0.1",
                    max(v for k, v in dm.items() if k != "wm_a75") <= dm["wm_a75"] + 0.001,
                )
            )
        cl = getattr(self, "_stats", {})
        if ("bge-m3", "miracl") in cl and ("wm_a75", "miracl") in cl:
            out.append(("v2-m3 is below bge-m3 on the clean MIRACL queries", cl[("wm_a75", "miracl")][1] < cl[("bge-m3", "miracl")][1]))
        quality, bench = DS.load_quality(DEPLOY), DS.load_bench(DEPLOY)
        for model in DEPLOY_MODELS:
            rows = {r["variant"]: r for r in DS.quality_rows(quality, model, self.dir_of)}

            def lat(v: str, model: str = model) -> float | None:
                b = bench.get((model, v, DS_THREADS_LOW))
                return b["workloads"]["query_b1"]["p50_ms"] if b else None

            def dmac(v: str, rows: dict = rows) -> float | None:
                return rows[v]["delta"]["macro"] if v in rows and "delta" in rows[v] else None

            if None not in (lat("torch-fp32"), lat("onnx-fp32")):
                out.append((f"{model}: ONNX fp32 answers a single query faster than PyTorch fp32", lat("onnx-fp32") < lat("torch-fp32")))
            if None not in (lat("onnx-int8"), lat("onnx-fp32")):
                out.append((f"{model}: dynamic int8 is faster per query than ONNX fp32", lat("onnx-int8") < lat("onnx-fp32")))
            if None not in (lat("onnx-w8"), lat("onnx-fp32")):
                out.append((f"{model}: weight-only 8-bit is slower per query than ONNX fp32", lat("onnx-w8") > lat("onnx-fp32")))
            if dmac("onnx-int8") is not None:
                out.append((f"{model}: dynamic int8 costs more than one macro point", dmac("onnx-int8") < -0.01))
            if dmac("onnx-int8-pc") is not None and dmac("onnx-int8") is not None:
                out.append((f"{model}: per-channel int8 loses less than per-tensor int8", dmac("onnx-int8-pc") > dmac("onnx-int8")))
            if dmac("onnx-w8") is not None:
                out.append((f"{model}: weight-only 8-bit stays within 0.3 macro points of fp32", abs(dmac("onnx-w8")) < 0.003))
            if dmac("onnx-fp32") is not None:
                out.append((f"{model}: ONNX fp32 is numerically equivalent to PyTorch fp32", abs(dmac("onnx-fp32")) < 0.0005))
        out.extend(self.study_claims())
        pairs = [("p3_s_e1_s13", "f4_s_e1_s13"), ("p3_b_e1_s13", "f4_b_e1_s13")]
        vol = [(load_system(a), load_system(b)) for a, b in pairs]
        m3a, m3b = load_system("m3_a_s13"), load_system("m3_b_s13")
        if all(x and y for x, y in vol) and m3a and m3b:
            out.append(
                (
                    "more translated data does not raise macro for the small students",
                    all(dir_value(y, "macro") <= dir_value(x, "macro") + 0.005 for x, y in vol),
                )
            )
            out.append(("more translated data raises macro for bge-m3", dir_value(m3b, "macro") > dir_value(m3a, "macro")))
        return out

    # -------------------------------------------------------------------------------------------- driver
    def write(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        clean_tab, stats = self.clean_split()
        tabs = {
            "tab_main": self.tab_main(),
            "tab_deltas": self.tab_deltas(),
            "tab_ablation": self.tab_ablation(),
            "tab_seeds": self.tab_seeds(),
            "tab_volume": self.tab_volume(),
            "tab_mteb": self.tab_mteb(),
            "tab_guards": self.tab_guards(),
            "tab_clean": clean_tab,
            "tab_data": self.tab_data(),
            "tab_deploy": self.tab_deploy(),
            "tab_scaling": self.tab_scaling(),
            "tab_negative": self.tab_negative(),
            "tab_mteb_paired": self.tab_mteb_paired(),
            "tab_confirmatory": self.tab_confirmatory(),
            "tab_licences": self.tab_licences(),
            "prereg": self.prereg_tex(),
        }
        has_curve = self.curve_numbers()
        self.numbers_licence_card()
        self.numbers_headline()
        self.numbers_licences()
        self.numbers_contamination(stats)
        self._stats = stats
        self.numbers_style()
        self.numbers_protocol()
        self.num("TodoCurveFigure", "" if has_curve else todo("WiSE curve: alpha_curve.json pending"))
        for name in (*PAIRED_DEFAULTS, *CURVE_DEFAULTS, *CONF_DEFAULTS, *STUDY_DEFAULTS):
            self.nums.setdefault(name, todo("pending"))
        for name in PENDING_DEFAULTS:  # anything the text uses but no result file provides yet is shown as a red TODO
            self.nums.setdefault(name, todo("pending"))
        for name, body in tabs.items():
            (OUT / f"{name}.tex").write_text(body)
        lines = ["% generated by report/make_tables.py: do not edit"]
        lines += [f"\\def\\{k}{{{v}}}" for k, v in sorted(self.nums.items())]
        (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
        claims = self.check_claims()
        (OUT / "claims_check.txt").write_text(
            "\n".join(f"{'OK     ' if ok else 'REVISE ' if ok is False else 'MISSING'} {text}" for text, ok in claims) + "\n"
        )
        bad = [t for t, ok in claims if ok is False]
        print(
            f"wrote {len(tabs)} tables and {len(self.nums)} number macros to {OUT}; claims: {sum(ok is True for _, ok in claims)} hold, {len(bad)} to revise"
        )
        for t in bad:
            print("  REVISE THE TEXT:", t)


def main() -> None:
    Tables().write()


if __name__ == "__main__":
    main()
