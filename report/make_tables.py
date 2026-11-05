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

from report.guard_tables import GUARD_DEFAULTS, GUARD_TASKS, GuardTables  # noqa: E402
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

# key -> (name in the table, result file stem); the v2 models are the final ones of the study: v2-m3 = WiSE blend of the seed-13 fine-tune (alpha from
# models/final_m3/merge.json), v2-base / v2-small = 3-seed soups of the distilled students (final_base / final_small)
SYSTEMS = {
    "e5s": ("mE5-small", "base_e5-small"),
    "v1": ("v1 (e5-small FT, seed 13)", "v1_s13"),
    "s_nokd": ("e5-small FT, no KD", "f4_s_e1_s13"),
    "v2s": ("v2-small", "st_s_soup"),
    "e5b": ("mE5-base", "base_e5-base"),
    "b_nokd": ("e5-base FT, no KD", "f4_b_e1_s13"),
    "v2b": ("v2-base", "st_b_soup"),
    "m3": ("bge-m3", "base_bge-m3"),
    "v2m_raw": ("bge-m3 raw fine-tune", "m3_b_s13"),
    "v2m": ("v2-m3", "final_m3"),
}
GROUPS = [["e5s", "v1", "s_nokd", "v2s"], ["e5b", "b_nokd", "v2b"], ["m3", "v2m_raw", "v2m"]]
DELTAS = [  # macro key, label, A (baseline), B (new)
    ("VOneVsESmall", "v1 vs mE5-small", "e5s", "v1"),
    ("VTwoSVsESmall", "v2-small vs mE5-small", "e5s", "v2s"),
    ("VTwoSVsVOne", "v2-small vs v1", "v1", "v2s"),
    ("VTwoBVsEBase", "v2-base vs mE5-base", "e5b", "v2b"),
    ("VTwoBVsBge", "v2-base vs bge-m3", "m3", "v2b"),
    ("VTwoSVsBge", "v2-small vs bge-m3", "m3", "v2s"),
    ("VTwoMVsBge", "v2-m3 vs bge-m3", "m3", "v2m"),
    ("DistSmall", "distilled vs plain FT, small", "s_nokd", "v2s"),
    ("DistBase", "distilled vs plain FT, base", "b_nokd", "v2b"),
    ("WiseCost", "WiSE blend vs raw fine-tune", "v2m_raw", "v2m"),
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


class Tables(GuardTables, StudyTables):
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
        for key, label, a, b in DELTAS:
            sa, sb = self.sys[a], self.sys[b]
            if not sa or not sb or not sa["perq"] or not sb["perq"]:
                rows.append(f"{esc(label)} & \\multicolumn{{4}}{{c}}{{{todo('result missing')}}} \\\\")
                continue
            ci = DS.macro_bootstrap(sa["perq"], sb["perq"], self.dir_of)
            cells = []
            for d, dn in (("ar->en", "ArEn"), ("macro", "Macro")):
                delta = dir_value(sb, d) - dir_value(sa, d)
                cells += [spct(delta), f"[{spct(ci[d]['lo'])}, {spct(ci[d]['hi'])}]"]
                self.num(f"NumDelta{key}{dn}", spct(delta))
                self.num(f"NumDelta{key}{dn}Lo", spct(ci[d]["lo"]))
                self.num(f"NumDelta{key}{dn}Hi", spct(ci[d]["hi"]))
            rows.append(f"{esc(label)} & " + " & ".join(cells) + r" \\")
        return (
            "\\begin{tabular}{@{}lrcrc@{}}\n\\toprule\n & \\multicolumn{2}{c}{$\\Delta$ "
            + DIR_TEX["ar->en"]
            + "} & \\multicolumn{2}{c}{$\\Delta$ macro} \\\\\n"
            "Comparison (B vs A) & pts & \\NumCiLevel\\% CI & pts & \\NumCiLevel\\% CI \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # -------------------------------------------------------------------------------------------- ablations (dev)
    def curve_alphas(self) -> dict[str, float]:
        """result stem -> alpha (fine-tuned share) of the post-hoc WiSE curve (results/retrieval_v2/alpha_curve.json)."""
        raw = load_json(V2 / "alpha_curve.json") or {}
        return {p["model"]: float(p["alpha"]) for p in raw.get("points", []) if "model" in p and "alpha" in p}

    def pick_alpha(self) -> float | None:
        a = (load_json(V2 / "models" / "final_m3" / "merge.json") or {}).get("alpha")
        return float(a) if a is not None else None

    def tab_ablation(self) -> str:
        def row(label: str, stem: str) -> str:
            j = load_json(V2 / f"{stem}.dev.json")
            if j is None:
                return f"{label} & \\multicolumn{{3}}{{c}}{{{todo('missing')}}} \\\\"
            dm = load_json(V2 / f"{stem}.devM.json") or (load_json(V2 / "bge-m3.devM.json") if stem == "base_bge-m3" else None)
            m = j["directions"]
            devm = pct(dm["slices"]["dev_miracl_ar"]["mrr@10"]) if dm else "--"
            return f"{label} & {pct(m['ar->en']['mrr@10'])} & {pct(m['macro']['mrr@10'])} & {devm} \\\\"

        alphas, pick = self.curve_alphas(), self.pick_alpha()
        wise = []
        for stem, a in sorted(alphas.items(), key=lambda kv: kv[1]):
            if a == 0:
                lab = rf"bge-m3 ($\alpha{{=}}{a:g}$)"
            elif a == 1:
                lab = rf"raw fine-tune ($\alpha{{=}}{a:g}$)"
            else:
                lab = rf"WiSE $\alpha{{=}}{a:g}$" + (" (pick)" if pick is not None and abs(a - pick) < 1e-9 else "")
            wise.append((lab, stem))
        blocks = [
            (
                "e5-base student",
                [
                    ("no KD, one epoch", "f4_b_e1_s13"),
                    ("KD from the raw fine-tune, $w{=}\\NumKdWeight$", "f5_b_kd_s13"),
                    ("KD $w{=}3$", "f5_b_kd3_s13"),
                    ("KD only", "f5_b_kdonly_s13"),
                    ("KD only, two epochs", "f5_b_kdonly_e2_s13"),
                    ("final recipe: teacher v2-m3, 3-seed soup", "st_b_soup"),
                ],
            ),
            ("bge-m3 fine-tuned, WiSE blends (seed 13)", wise),
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
        """Development seeds (13 to 15): the raw fine-tune, the WiSE pick recipe and the two distilled students; per-seed values of the pick are in the confirmatory table."""
        groups = [
            ("raw fine-tune ($\\alpha{=}1$)", ["m3_b_s13", "m3_b_s14", "m3_b_s15"], "Raw"),
            ("v2-m3 recipe (WiSE)", ["final_m3", "w_m3b_a40_s14", "w_m3b_a40_s15"], "Wise"),
            ("v2-base students", ["st_b_s13", "st_b_s14", "st_b_s15"], "StB"),
            ("v2-small students", ["st_s_s13", "st_s_s14", "st_s_s15"], "StS"),
        ]
        rows = []
        for label, stems, key in groups:
            vals = [(dir_value(s, "ar->en"), dir_value(s, "macro")) for s in (load_system(x) for x in stems) if s]
            if len(vals) < len(stems):
                rows.append(f"{label} & \\multicolumn{{2}}{{c}}{{{todo('seeds missing')}}} \\\\")
                continue
            a = np.array(vals)
            rows.append(
                f"{label} & {pct(a[:, 0].mean())} $\\pm$ {pct(a[:, 0].std(ddof=1))} & {pct(a[:, 1].mean())} $\\pm$ {pct(a[:, 1].std(ddof=1))} \\\\"
            )
            self.num(f"NumSeed{key}ArEn", pct(a[:, 0].mean()))
            self.num(f"NumSeed{key}Sd", pct(a[:, 0].std(ddof=1)))
            self.num(f"NumSeed{key}Range", pct(a[:, 0].max() - a[:, 0].min()))
        return (
            "\\begin{tabular}{@{}lrr@{}}\n\\toprule\nTEST, mean $\\pm$ s.d. over three development seeds & "
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
        """Leaderboard: the ten baseline models of results/mteb_ara/leaderboard.json plus the v2 models scored by the same pipeline (mteb_board.json)."""
        lb = load_json(MTEB / "leaderboard.json")
        if lb is None:
            return todo("leaderboard.json missing")
        board = self.board()
        cols = [  # (leaderboard key, board key, header)
            ("Mean (Task)", "Mean(Task)", "Task"),
            ("Mean (TaskType)", "Mean(TaskType)", "Type"),
            ("Retrieval", "type:Retrieval", "Retr."),
            ("Reranking", "type:Reranking", "Rerank"),
            ("STS", "type:STS", "STS"),
            ("Classification", "type:Classification", "Cls."),
            ("Mean ar-only subsets", "ar-only subsets", "ar"),
            ("Mean ar-en cross-lingual subsets", "ar-en cross-lingual subsets", "ar-en"),
        ]
        entries: list[tuple[str, bool, list[float]]] = [
            (r["model_name"].split("/")[-1], False, [r[c] for c, _, _ in cols]) for r in lb["summary"]
        ]
        v2_rows = [
            ("final_m3", "v2-m3"),
            ("w_m3b_a40_s16", "v2-m3, confirmatory seed 16"),
            ("w_m3b_a40_s17", "v2-m3, confirmatory seed 17"),
            ("final_base", "v2-base"),
            ("final_small", "v2-small"),
        ]
        for model, name in v2_rows:
            if model in board:
                entries.append((name, True, [board[model][b] for _, b, _ in cols]))
        entries.sort(key=lambda e: -e[2][0])
        best = [max(e[2][i] for e in entries) for i in range(len(cols))]
        rows = []
        for name, is_v2, vals in entries:
            cells = [rf"\textbf{{{v:.1f}}}" if abs(v - best[i]) < 0.05 else f"{v:.1f}" for i, v in enumerate(vals)]
            rows.append((r"\textbf{" + esc(name) + "}" if is_v2 else esc(name)) + " & " + " & ".join(cells) + r" \\")
        self.num("NumMtebTasks", str(lb["summary"][0]["n_tasks"]))
        self.num("NumMtebModels", str(len(lb["summary"])))
        names = {
            "BAAI/bge-m3": "Bge",
            "intfloat/multilingual-e5-large": "ELarge",
            "intfloat/multilingual-e5-base": "EBase",
            "intfloat/multilingual-e5-small": "ESmall",
            "mzahran1/arabic-english-e5-small-retrieval": "VOne",
        }
        for r in lb["summary"]:
            nm = names.get(r["model_name"])
            if nm:
                self.num(f"NumMteb{nm}Task", f"{r['Mean (Task)']:.2f}")
                self.num(f"NumMteb{nm}Type", f"{r['Mean (TaskType)']:.2f}")
                self.num(f"NumMteb{nm}Cross", f"{r['Mean ar-en cross-lingual subsets']:.1f}")
        for model, nm in (
            ("final_m3", "VTwoM"),
            ("final_base", "VTwoB"),
            ("final_small", "VTwoS"),
            ("w_m3b_a40_s16", "SixteenSeed"),
            ("w_m3b_a40_s17", "SeventeenSeed"),
        ):
            if model in board:
                self.num(f"NumMteb{nm}Task", f"{board[model]['Mean(Task)']:.2f}")
                self.num(f"NumMteb{nm}Type", f"{board[model]['Mean(TaskType)']:.2f}")
                self.num(f"NumMteb{nm}Cross", f"{board[model]['ar-en cross-lingual subsets']:.1f}")
        head = " & ".join(h for _, _, h in cols)
        return (
            "\\begin{tabular}{@{}l"
            + "r" * len(cols)
            + "@{}}\n\\toprule\nModel & "
            + head
            + " \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

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
            "e5b": "EBase",
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
        alpha = self.pick_alpha()
        if alpha is not None:
            self.num("NumWiseAlpha", f"{alpha:g}")
            self.num("NumWiseKept", f"{100 * alpha:.0f}")
            self.num("NumWiseOriginal", f"{100 * (1 - alpha):.0f}")
        first = (load_json(V2 / "models" / "wm_a75" / "merge.json") or {}).get("alpha")
        if first is not None:
            self.num("NumWiseAlphaFirst", f"{first:g}")
        for key, stem in (("M", "m3_b_s13"), ("B", "st_b_s13"), ("S", "st_s_s13")):
            a = (load_json(V2 / "models" / stem / "lab_retrieval.json") or {}).get("args")
            if a:
                self.num(f"NumLr{key}", f"{a['lr']:g}".replace("e-0", "e{-}").replace("e-", "e{-}"))
                self.num(f"NumBs{key}", str(a["bs"]))
                self.num(f"NumScale{key}", f"{a['scale']:g}")
                self.num(f"NumMaxLen{key}", str(a["max_len"]))
                self.num(f"NumTemp{key}", f"{a['t_scale']:g}")
                self.num(f"NumEpochs{key}", f"{a['epochs']:g}")
        v1 = [load_json(ROOT / "results" / "retrieval" / f"e5s_hard1_xl_e1_s{s}.test.json") for s in (13, 14, 15)]
        if all(v1):  # the published v1 recipe, three seeds (the committed v1 result snapshot)
            ar = np.array([j["directions"]["ar->en"]["mrr@10"] for j in v1])
            mac = np.array([j["directions"]["macro"]["mrr@10"] for j in v1])
            self.num("NumVOneSeedArEn", pct(ar.mean()))
            self.num("NumVOneSeedSd", pct(ar.std(ddof=1)))
            self.num("NumVOneSeedMacro", pct(mac.mean()))
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
            "reranker_depth": "NumRerankDepth",
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
        k = (load_json(V2 / "models" / "st_b_s13" / "lab_retrieval.json") or {}).get("args", {})
        if k:
            self.num("NumKdWeight", f"{k['kd']:g}")
        self.numbers_cost(pr)
        self.numbers_suite()

    def numbers_suite(self) -> None:
        """The validation-suite selection: pick, its score, the best-scoring selectable model and the gap that the tie band decided (results/retrieval_v2/suite_pick.json)."""
        sp = load_json(V2 / "suite_pick.json")
        if not sp:
            return
        rows = [r for r in sp["rows"] if r.get("selectable")]
        top = max(rows, key=lambda r: r["score"])
        pick = next(r for r in rows if r["name"] == sp["pick"])
        self.num("NumSuitePickScore", f"{pick['score']:.4f}")
        self.num("NumSuiteTopScore", f"{top['score']:.4f}")
        self.num("NumSuiteGap", f"{top['score'] - pick['score']:.4f}")
        self.num("NumSuiteTopFamily", esc(top["family"]))
        self.num("NumSuitePool", str(len(rows)))
        m = re.search(r"tie band \(< ([0-9.]+)\)", sp.get("rule", ""))
        if m:
            self.num("NumSuiteTieBand", m.group(1))

    def numbers_cost(self, pr: dict) -> None:
        """Compute used by the study: training hours summed over the committed per-run train.json files, MTEB scoring time, wall-clock estimate (lab notebook)."""
        secs = [j.get("train_seconds") for j in (load_json(p) for p in sorted(V2.glob("*.train.json"))) if j]
        secs = [x for x in secs if x]
        if secs:
            self.num("NumGpuRuns", str(len(secs)))
            self.num("NumGpuTrainHours", f"{sum(secs) / 3600:.1f}")
        mt = load_json(V2 / "mteb_eval_time.json")
        if mt:
            self.num("NumGpuMtebHours", f"{mt['hours']:.1f}")
        if "gpu_wallclock_hours" in pr:
            self.num("NumGpuWallHours", f"{pr['gpu_wallclock_hours']['value']:g}")
        conf = load_json(V2 / "confirmatory.json")
        if conf and conf.get("recipe_frozen_at"):
            self.num("NumFrozenAt", conf["recipe_frozen_at"].replace("T", " ").replace("+03:00", " (UTC+3)"))

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
        for a_, b_, d_, desc in (
            ("m3", "v2m", "ar->en", "v2-m3 beats bge-m3 on ar->en (paired CI above 0)"),
            ("m3", "v2m", "en->ar", "v2-m3 beats bge-m3 on en->ar (paired CI above 0)"),
            ("m3", "v2b", "ar->en", "v2-base beats bge-m3 on ar->en (paired CI above 0)"),
            ("m3", "v2b", "macro", "v2-base beats bge-m3 on macro (paired CI above 0)"),
            ("e5b", "v2b", "ar->en", "v2-base beats mE5-base on ar->en (paired CI above 0)"),
            ("e5s", "v2s", "ar->en", "v2-small beats mE5-small on ar->en (paired CI above 0)"),
            ("v1", "v2s", "ar->en", "v2-small beats v1 on ar->en (paired CI above 0)"),
            ("m3", "v2s", "ar->en", "v2-small is below bge-m3 on ar->en (paired CI below 0)"),
        ):
            c = ci(a_, b_, d_)
            out.append((desc, None if c is None else (c["hi"] < 0 if "below" in desc else c["lo"] > 0)))
        for a_, b_, nm in (("b_nokd", "v2b", "base"), ("s_nokd", "v2s", "small")):
            c = ci(a_, b_, "ar->en")
            out.append(
                (f"the distilled {nm} student beats its plain fine-tune on ar->en (paired CI above 0)", None if c is None else c["lo"] > 0)
            )
        a, b = val("v2m", "ar->en"), val("v2b", "ar->en")
        out.append(("the students stay below the v2-m3 teacher on ar->en", None if a is None or b is None else a > b > 0))
        c = ci("v2m_raw", "v2m", "macro")
        out.append(("the WiSE blend costs macro points relative to the raw fine-tune (CI below 0)", None if c is None else c["hi"] < 0))
        g = self.guard_scores()
        mi, na, se = (t for t, _, _ in GUARD_TASKS)
        try:
            gm, gb = g["final_m3"], g["BAAI__bge-m3"]
            out.append(("v2-m3 is about one point below bge-m3 on MIRACL-HN (between 0.5 and 1.5)", 0.5 < gb[mi] - gm[mi] < 1.5))
            out.append(("v2-m3 misses the one-point bar on Mr.TyDi rerank (more than 2 points below bge-m3)", gb[na] - gm[na] > 2.0))
            out.append(("v2-m3 is above bge-m3 on SemRel24", gm[se] > gb[se]))
            for model, ref, nm in (
                ("final_base", "intfloat__multilingual-e5-base", "v2-base"),
                ("final_small", "intfloat__multilingual-e5-small", "v2-small"),
            ):
                out.append((f"{nm} loses more than 8 points on Mr.TyDi rerank relative to its base", g[ref][na] - g[model][na] > 8.0))
        except KeyError:
            out.append(("guard scores present", None))
        lb = load_json(MTEB / "leaderboard.json")
        board = self.board()
        if lb and board:
            by = {r["model_name"].split("/")[-1]: r for r in lb["summary"]}
            base_best = max(r["Mean (Task)"] for r in lb["summary"])
            out.append(
                (
                    "v2-m3 has the highest MTEB(ara) Mean (Task) of all scored models (numerically)",
                    board["final_m3"]["Mean(Task)"] > base_best,
                )
            )
            out.append(
                (
                    "v1 scores below multilingual-e5-small on MTEB(ara) Mean (Task)",
                    by["arabic-english-e5-small-retrieval"]["Mean (Task)"] < by["multilingual-e5-small"]["Mean (Task)"],
                )
            )
            cross = {r["model_name"].split("/")[-1]: r["Mean ar-en cross-lingual subsets"] for r in lb["summary"]}
            out.append(
                (
                    "multilingual-e5-large is ahead of v2-m3 on the ar-en cross-lingual subsets (no overall first place)",
                    cross["multilingual-e5-large"] > board["final_m3"]["ar-en cross-lingual subsets"],
                )
            )
            small = [
                v for k, v in cross.items() if k in ("multilingual-e5-base", "multilingual-e5-small", "arabic-english-e5-small-retrieval")
            ]
            out.append(
                (
                    "v2-base leads the models up to 278M on the ar-en cross-lingual subsets",
                    board["final_base"]["ar-en cross-lingual subsets"] > max(small),
                )
            )
            out.append(
                (
                    "v2-small has a lower Mean (TaskType) than mE5-small",
                    board["final_small"]["Mean(TaskType)"] < by["multilingual-e5-small"]["Mean (TaskType)"],
                )
            )
            out.append(
                (
                    "v2-base has a higher Mean (TaskType) than mE5-base",
                    board["final_base"]["Mean(TaskType)"] > by["multilingual-e5-base"]["Mean (TaskType)"],
                )
            )
            seeds = [board[m]["Mean(Task)"] for m in ("w_m3b_a40_s16", "w_m3b_a40_s17") if m in board]
            out.append(
                (
                    "the confirmatory seeds reproduce the MTEB mean of seed 13 to within 0.1",
                    all(abs(x - board["final_m3"]["Mean(Task)"]) < 0.1 for x in seeds),
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
            for k in ("m3_b_s13", "m3_r_s13", "m3_p_s13", "m3_fr_s13", "m3_lo_s13")
        }
        if all(v is not None for v in dm.values()):
            out.append(
                (
                    "replay, preservation, freezing and LoRA do not beat the raw fine-tune on held-out MIRACL-ar by more than 0.3",
                    max(v for k, v in dm.items() if k != "m3_b_s13") <= dm["m3_b_s13"] + 0.003,
                )
            )
        cl = getattr(self, "_stats", {})
        if ("bge-m3", "miracl") in cl and ("final_m3", "miracl") in cl:
            out.append(("v2-m3 is below bge-m3 on the clean MIRACL queries", cl[("final_m3", "miracl")][1] < cl[("bge-m3", "miracl")][1]))
            out.append(
                (
                    "every model scores lower on the clean MIRACL queries than on the flagged ones",
                    all(v[1] < v[4] for k, v in cl.items() if k[1] == "miracl" and len(v) == 5),
                )
            )
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
        out.extend(self.contamination_claims())
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
        clean_tab, stats, clean_paired = self.clean_split()
        tabs = {
            "tab_main": self.tab_main(),
            "tab_deltas": self.tab_deltas(),
            "tab_ablation": self.tab_ablation(),
            "tab_seeds": self.tab_seeds(),
            "tab_volume": self.tab_volume(),
            "tab_mteb": self.tab_mteb(),
            "tab_guards": self.tab_guards(),
            "tab_clean": clean_tab,
            "tab_clean_paired": clean_paired,
            "tab_rerank": self.tab_rerank(),
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
        for name in (*PAIRED_DEFAULTS, *CURVE_DEFAULTS, *CONF_DEFAULTS, *STUDY_DEFAULTS, *GUARD_DEFAULTS):
            self.nums.setdefault(name, todo("pending"))
        for name in PENDING_DEFAULTS:  # anything the text uses but no result file provides yet is shown as a red TODO
            self.nums.setdefault(name, todo("pending"))
        for name, body in tabs.items():
            (OUT / f"{name}.tex").write_text(body)
        lines = ["% generated by report/make_tables.py: do not edit"]
        lines += [f"\\def\\{k}{{{v}}}" for k, v in sorted(self.nums.items())]
        (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
        claims = [(text, None if ok is None else bool(ok)) for text, ok in self.check_claims()]  # numpy bools count as bools
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
