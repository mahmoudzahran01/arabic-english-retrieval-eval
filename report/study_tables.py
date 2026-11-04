"""Tables and numbers of the empirical-study part of the report: data scaling, negative results, MTEB(ara) uncertainty, confirmatory seeds,
data card / licences, the WiSE trade-off curve and the pre-registration appendix.

`StudyTables` is a mixin of `make_tables.Tables`: it needs `self.nums`, `self.num(name, value)`, `self.dir_of` and `self.sys` from the host. Every method returns
LaTeX (or None) and registers number macros; anything whose input file does not exist yet is rendered as a red \\TODO so that nothing pending looks like a result.
"""

from __future__ import annotations

import re

import numpy as np
from report.tabcommon import (
    DEPLOY,
    DIR_TEX,
    MTEB,
    V2,
    dev_value,
    devm_value,
    dir_value,
    esc,
    latex_text,
    load_json,
    load_system,
    pct,
    sfmt,
    spct,
    thousands,
    todo,
)

from deploy import summarize as DS
from report import make_figures as MF
from report import paired as PD

# data-scaling curve: (label, result stem). New translated queries are counted from the training-set manifest of each run (data.by_slice, slices xl2_*).
SCALING = {
    "small": [
        ("none (v1 translations only)", "p1_r0"),
        ("+25% of the new translations", "p2_s_f25"),
        ("+first batch of new translations", "p2_xl2_a5k"),
        ("+50% of the new translations", "p2_s_f50"),
        ("+all translations available at the time", "p2_s_full"),
        ("full set, contamination-filtered", "f4_s_e1_s13"),
    ],
    "base": [
        ("none (v1 translations only)", "p1_base_r0"),
        ("+all translations available at the time", "p2_b_full"),
        ("full set, contamination-filtered", "f4_b_e1_s13"),
    ],
}
# negative / null results: (label, variant stem, baseline stem, group, adopted in the final recipe?)
NEGATIVE = [
    ("Other-language hard negative", "p1_oth1", "p1_r0", "small", "no"),
    ("Two same-language hard negatives", "p1_same2", "p1_r0", "small", "no"),
    ("Language-homogeneous batches", "p1_bplang", "p1_r0", "small", "no"),
    ("Symmetric InfoNCE", "p1_sym", "p1_r0", "small", "no"),
    ("KD from off-the-shelf bge-m3, $w{=}1$", "p1_kd1", "p1_r0", "small", "no"),
    ("KD from off-the-shelf bge-m3, $w{=}3$", "p1_kd3", "p1_r0", "small", "no"),
    ("KD from off-the-shelf bge-m3, KD only", "p1_kdonly", "p1_r0", "small", "no"),
    ("Two epochs", "p2_s_e2", "p2_s_full", "small", "no"),
    ("MIRACL-ar replay", "m3_r_s13", "m3_b_s13", "m3", "no"),
    ("Embedding preservation to frozen bge-m3", "m3_p_s13", "m3_b_s13", "m3", "no"),
    ("Frozen embeddings + bottom 12 layers", "m3_fr_s13", "m3_b_s13", "m3", "no"),
    ("LoRA ($r{=}32$)", "m3_lo_s13", "m3_b_s13", "m3", "no"),
]
DEPLOY_NEG = [("v2-small", "Small"), ("v2-base", "Base")]
CONFIRM_RE = re.compile(r"^(?P<recipe>.+)_s(?P<seed>16|17)\.test\.json$")
ROLE_TEXT = {
    "train": "training data (final recipe)",
    "train-exp": "training data (ablation only)",
    "eval": "evaluation / selection only",
    "translator": "translated the training queries",
    "base": "initial weights",
}
SUBSET_ORDER = ("ar-en cross-lingual subsets", "ar-only subsets")
LEADERBOARD_PAIRS = [
    ("bge-m3 vs mE5-base", "bge-m3", "multilingual-e5-base"),
    ("mE5-base vs mE5-small", "multilingual-e5-base", "multilingual-e5-small"),
    ("v1 vs mE5-small", "arabic-english-e5-small-retrieval", "multilingual-e5-small"),
    ("bge-m3 vs mE5-large", "bge-m3", "multilingual-e5-large"),
]
PAIRED_DEFAULTS = [
    f"NumPaired{g}{k}" for g in ("M", "B") for k in ("MeanDiff", "CiLo", "CiHi", "Wins", "Losses", "Ties", "SignP", "Tasks")
] + [f"NumPairedMCross{k}" for k in ("Diff", "CiLo", "CiHi", "Wins", "Losses", "Ties", "Tasks", "P")]
CURVE_DEFAULTS = [
    "NumCurvePoints",
    "NumCurveSelected",
    "NumCurveArEnSel",
    "NumCurveMiraclSel",
    "NumCurveMiraclZero",
    "NumCurveBarAlpha",
    "NumCurveBarArEn",
]
STUDY_DEFAULTS = [
    "NumScaleSmallFirstN", "NumScaleSmallFullN", "NumScaleSmallFirstGain", "NumScaleSmallFullGain", "NumScaleSmallFirstShare", "NumScaleBaseFullGain", "NumScaleFinalN",
    "NumNegBaseSmallMacro", "NumNegBaseSmallArEn", "NumNegBaseBgeMacro", "NumNegBaseBgeArEn", "NumNegBaseBgeDevM", "NumNegRows", "NumNegMaxGainMacro", "NumNegMinMacro", "NumNegMaxGainArEn",
    "NumPairedRefMeanDiff", "NumPairedRefCiLo", "NumPairedRefCiHi", "NumPairedRefSignP", "NumPairedMBestTask", "NumPairedMBestGain", "NumPairedMWorstTask", "NumPairedMWorstGain",
    "NumLicChecked", "NumLicSources", "LicGemma", "LicMiracl", "LicBge", "LicEBase",
]  # fmt: skip
CONF_DEFAULTS = ["NumConfSeeds", "NumConfArEnMean", "NumConfMacroMean", "NumConfDeltaArEn", "NumConfDeltaMacro", "NumConfSd"]


def _fmt_p(p: float) -> str:
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


class StudyTables:
    # ------------------------------------------------------------------------------------------------ data scaling
    def _n_new_translations(self, stem: str) -> int | None:
        t = load_json(V2 / f"{stem}.train.json")
        if t is None:
            return None
        by = t.get("data", {}).get("by_slice", {})
        return int(sum(v for k, v in by.items() if "_xl2_" in k))

    def tab_scaling(self) -> str:
        rows, cache = [], {}
        for group, items in SCALING.items():
            rows.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{'e5-small' if group == 'small' else 'e5-base'} student (no KD)}}}} \\")
            for label, stem in items:
                n, ar, en, mac = (
                    self._n_new_translations(stem),
                    dev_value(stem, "ar->en"),
                    dev_value(stem, "en->ar"),
                    dev_value(stem, "macro"),
                )
                cache[stem] = (n, ar, en, mac)
                if None in (n, ar, mac):
                    rows.append(f"{esc(label)} & \\multicolumn{{4}}{{c}}{{{todo('missing')}}} \\\\")
                else:
                    rows.append(f"{esc(label)} & {thousands(n)} & {pct(ar)} & {pct(en)} & {pct(mac)} \\\\")
        s0, s25, sfull = cache.get("p1_r0"), cache.get("p2_s_f25"), cache.get("p2_s_full")
        if s0 and s25 and sfull and None not in (s0[1], s25[1], sfull[1]):
            self.num("NumScaleSmallFirstN", thousands(s25[0]))
            self.num("NumScaleSmallFullN", thousands(sfull[0]))
            self.num("NumScaleSmallFirstGain", spct(s25[1] - s0[1]))
            self.num("NumScaleSmallFullGain", spct(sfull[1] - s0[1]))
            self.num("NumScaleSmallFirstShare", f"{100 * (s25[1] - s0[1]) / (sfull[1] - s0[1]):.0f}")
        b0, bfull = cache.get("p1_base_r0"), cache.get("p2_b_full")
        if b0 and bfull and None not in (b0[1], bfull[1]):
            self.num("NumScaleBaseFullGain", spct(bfull[1] - b0[1]))
        ff = cache.get("f4_s_e1_s13")
        if ff and ff[0] is not None:
            self.num("NumScaleFinalN", thousands(ff[0]))
        return (
            "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nDEV; training queries & new transl. & "
            + DIR_TEX["ar->en"]
            + " & "
            + DIR_TEX["en->ar"]
            + " & macro \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # ------------------------------------------------------------------------------------------------ negative results
    def tab_negative(self) -> str:
        rows, gains = [], []
        base_small, base_m3 = ("p1_r0", "m3_b_s13")
        for label, stem, base, group, adopted in NEGATIVE:
            dm, da = dev_value(stem, "macro"), dev_value(stem, "ar->en")
            bm, ba = dev_value(base, "macro"), dev_value(base, "ar->en")
            if None in (dm, da, bm, ba):
                rows.append(
                    f"{label} & \\multicolumn{{3}}{{c}}{{{todo('run pending' if stem.startswith('m3_lo') else 'missing')}}} & {adopted} \\\\"
                )
                continue
            vm, vb = devm_value(stem), devm_value(base)
            cell_m = sfmt(100 * (vm - vb)) if vm is not None and vb is not None and group == "m3" else "--"
            rows.append(f"{label} & {sfmt(100 * (dm - bm))} & {sfmt(100 * (da - ba))} & {cell_m} & {adopted} \\\\")
            gains.append((100 * (dm - bm), 100 * (da - ba), group, label))
        # deployment: dynamic int8 (TEST)
        quality = DS.load_quality(DEPLOY)
        dyn = []
        for model, _ in DEPLOY_NEG:
            r = {x["variant"]: x for x in DS.quality_rows(quality, model, self.dir_of)}.get("onnx-int8")
            dyn.append(r["delta"] if r and "delta" in r else None)
        if all(dyn):
            rows.append(r"\midrule")
            rows.append(
                f"Dynamic int8 export, v2-small / v2-base (TEST) & {sfmt(100 * dyn[0]['macro'])} / {sfmt(100 * dyn[1]['macro'])} & "
                f"{sfmt(100 * dyn[0]['ar->en'])} / {sfmt(100 * dyn[1]['ar->en'])} & -- & no \\\\"
            )
        bs = (dev_value(base_small, "macro"), dev_value(base_small, "ar->en"))
        bm3 = (dev_value(base_m3, "macro"), dev_value(base_m3, "ar->en"), devm_value(base_m3))
        if None not in bs:
            self.num("NumNegBaseSmallMacro", pct(bs[0]))
            self.num("NumNegBaseSmallArEn", pct(bs[1]))
        if None not in bm3:
            self.num("NumNegBaseBgeMacro", pct(bm3[0]))
            self.num("NumNegBaseBgeArEn", pct(bm3[1]))
            self.num("NumNegBaseBgeDevM", pct(bm3[2]))
        if gains:
            self.num("NumNegRows", str(len(gains)))
            self.num("NumNegMaxGainMacro", sfmt(max(g[0] for g in gains)))
            self.num("NumNegMinMacro", sfmt(min(g[0] for g in gains)))
            self.num("NumNegMaxGainArEn", sfmt(max(g[1] for g in gains)))
        self._neg_gains = gains
        return (
            "\\begin{tabular}{@{}lrrrc@{}}\n\\toprule\nChange (DEV, versus its own baseline) & $\\Delta$ macro & $\\Delta$ "
            + DIR_TEX["ar->en"]
            + " & $\\Delta$ MIRACL-ar & kept \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # ------------------------------------------------------------------------------------------------ MTEB(ara) uncertainty
    def paired_comparisons(self) -> list[PD.Comparison]:
        return PD.load(V2 / "mteb_paired.json")

    @property
    def subsets(self) -> dict:
        return PD.load_subsets(V2 / "mteb_subsets.json")

    def _register_cross(self, st: dict) -> None:
        self.num("NumPairedMCrossDiff", sfmt(st["mean_diff"]))
        self.num("NumPairedMCrossCiLo", sfmt(st["ci_lo"]))
        self.num("NumPairedMCrossCiHi", sfmt(st["ci_hi"]))
        self.num("NumPairedMCrossWins", str(st["wins"]))
        self.num("NumPairedMCrossLosses", str(st["losses"]))
        self.num("NumPairedMCrossTies", str(st["ties"]))
        self.num("NumPairedMCrossTasks", str(st["n"]))
        self.num("NumPairedMCrossP", _fmt_p(st["sign_p"]))

    def tab_mteb_paired(self) -> str:
        comps = self.paired_comparisons()
        lb = load_json(MTEB / "leaderboard.json") or {}
        refs = PD.from_leaderboard(lb, LEADERBOARD_PAIRS)
        rows = []

        def row(c: PD.Comparison, st: dict, label: str) -> str:
            return f"{esc(label)} & {st['n']} & {sfmt(st['mean_diff'])} & [{sfmt(st['ci_lo'])}, {sfmt(st['ci_hi'])}] & {st['wins']}/{st['losses']}/{st['ties']} & {_fmt_p(st['sign_p'])} \\\\"

        if comps:
            for c in comps:
                st = PD.paired_stats(c.scores)
                rows.append(row(c, st, c.label.replace("BAAI__", "").replace("final_m3_wa75_s13", "v2-m3").replace("_", " ")))
                subs = dict(c.subsets)
                for name, kind in (("ar-en cross-lingual subsets", "cross"), ("ar-only subsets", "mono")):
                    if name not in subs:
                        sc = PD.subset_comparison(self.subsets, c.new, c.ref, kind)
                        if sc:
                            subs[name] = sc
                got = False
                for name in SUBSET_ORDER:
                    if name in subs:
                        stat = PD.paired_stats(subs[name])
                        rows.append(row(c, stat, f"\\quad {name}"))
                        if name == "ar-en cross-lingual subsets":
                            got = True
                            if "bge-m3" in c.ref and "e5" not in c.ref:
                                self._register_cross(stat)
                if not got:
                    rows.append(
                        f"\\quad ar-en cross-lingual subsets & \\multicolumn{{5}}{{c}}{{{todo('per-subset scores not available (mteb_subsets.json)')}}} \\\\"
                    )
            self._register_paired(comps)
        else:
            rows.append(f"v2-m3 vs bge-m3 & \\multicolumn{{5}}{{c}}{{{todo('mteb_paired.json pending')}}} \\\\")
            rows.append(f"v2-base vs mE5-base, v2-small vs mE5-small & \\multicolumn{{5}}{{c}}{{{todo('mteb_paired.json pending')}}} \\\\")
        if not any(("e5-base" in c.ref or "e5_base" in c.ref) for c in comps):
            rows.append(
                f"v2-base vs mE5-base (student) & \\multicolumn{{5}}{{c}}{{{todo('MTEB(ara) scoring of the students pending')}}} \\\\"
            )
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{6}{@{}l}{\emph{reference pairs from the 10-model leaderboard}} \\")
        for c in refs:
            rows.append(row(c, PD.paired_stats(c.scores), c.label))
            if c.label == "bge-m3 vs mE5-base":
                st = PD.paired_stats(c.scores)
                self.num("NumPairedRefMeanDiff", sfmt(st["mean_diff"]))
                self.num("NumPairedRefCiLo", sfmt(st["ci_lo"]))
                self.num("NumPairedRefCiHi", sfmt(st["ci_hi"]))
                self.num("NumPairedRefSignP", f"{st['sign_p']:.3f}")
        return (
            "\\begin{tabular}{@{}lrrcrr@{}}\n\\toprule\nMean (Task) difference, new $-$ ref & tasks & diff & 95\\% CI (tasks) & W/L/T & sign p \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def _register_paired(self, comps: list[PD.Comparison]) -> None:
        for group, pick in (
            ("M", lambda c: "bge-m3" in c.ref and "e5" not in c.ref),
            ("B", lambda c: "e5-base" in c.ref or "e5_base" in c.ref),
        ):
            c = next((c for c in comps if pick(c)), None)
            if c is None:
                continue
            st = PD.paired_stats(c.scores)
            self.num(f"NumPaired{group}MeanDiff", sfmt(st["mean_diff"], 2))
            self.num(f"NumPaired{group}CiLo", sfmt(st["ci_lo"], 1))
            self.num(f"NumPaired{group}CiHi", sfmt(st["ci_hi"], 1))
            self.num(f"NumPaired{group}Wins", str(st["wins"]))
            self.num(f"NumPaired{group}Losses", str(st["losses"]))
            self.num(f"NumPaired{group}Ties", str(st["ties"]))
            self.num(f"NumPaired{group}SignP", f"{st['sign_p']:.2f}")
            self.num(f"NumPaired{group}Tasks", str(st["n"]))
            if group == "M":
                best = max(c.scores, key=lambda t: c.scores[t][0] - c.scores[t][1])
                worst = min(c.scores, key=lambda t: c.scores[t][0] - c.scores[t][1])
                self.num("NumPairedMBestTask", esc(best.replace(".v2", "")))
                self.num("NumPairedMBestGain", sfmt(c.scores[best][0] - c.scores[best][1]))
                self.num("NumPairedMWorstTask", esc(worst.replace(".v2", "")))
                self.num("NumPairedMWorstGain", sfmt(c.scores[worst][0] - c.scores[worst][1]))

    def mteb_extra_rows(self) -> list[str]:
        """Rows of the MTEB(ara) leaderboard table for v2 models that have per-task scores in mteb_paired.json (type means recomputed from the 19 tasks)."""
        cols = ["Mean (Task)", "Mean (TaskType)", "Retrieval", "Reranking", "STS", "Classification"]
        rows = []
        for c in self.paired_comparisons():
            vals = PD.type_means({t: n for t, (n, _) in c.scores.items()})
            if vals.get("Mean (Task)") is None:
                continue
            name = c.new.replace("final_m3_wa75_s13", "v2-m3").replace("_", " ")
            cells = [f"{vals[k]:.1f}" for k in cols]
            mono, cross = PD.subset_mean(self.subsets, c.new, "mono"), PD.subset_mean(self.subsets, c.new, "cross")
            sub_cells = [f"{v:.1f}" if v is not None else todo("per-subset") for v in (mono, cross)]
            rows.append(f"{esc(name)} & " + " & ".join(cells) + " & " + " & ".join(sub_cells) + " \\\\")
        return rows

    # ------------------------------------------------------------------------------------------------ confirmatory seeds
    def confirmatory_systems(self) -> dict[str, dict[int, str]]:
        found: dict[str, dict[int, str]] = {}
        for p in sorted(V2.glob("*_s1[67].test.json")):
            m = CONFIRM_RE.match(p.name)
            if m:
                found.setdefault(m.group("recipe"), {})[int(m.group("seed"))] = p.name[: -len(".test.json")]
        return found

    def tab_confirmatory(self) -> str:
        found = self.confirmatory_systems()
        ref = load_system("base_bge-m3")
        if not found:
            self.num("NumConfSeeds", todo("seeds 16 and 17 pending"))
            return (
                "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nRecipe, seed & "
                + DIR_TEX["ar->en"]
                + " & macro & $\\Delta$ "
                + DIR_TEX["ar->en"]
                + " vs bge-m3 \\\\\n\\midrule\n"
                + f"seeds 16 and 17 & \\multicolumn{{3}}{{c}}{{{todo('trained after the recipe was frozen; pending')}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
            )
        rows, headline = [], []
        for recipe, seeds in found.items():
            vals = []
            for seed, stem in sorted(seeds.items()):
                s = load_system(stem)
                ar, mac = dir_value(s, "ar->en"), dir_value(s, "macro")
                cell = "--"
                if ref and ref["perq"] and s["perq"]:
                    ci = DS.macro_bootstrap(ref["perq"], s["perq"], self.dir_of)
                    cell = f"{spct(ar - dir_value(ref, 'ar->en'))} [{spct(ci['ar->en']['lo'])}, {spct(ci['ar->en']['hi'])}]"
                rows.append(f"{esc(recipe)}, seed {seed} & {pct(ar)} & {pct(mac)} & {cell} \\\\")
                vals.append((ar, mac))
            if len(vals) > 1:
                a = np.array(vals)
                rows.append(
                    f"{esc(recipe)}, mean of {len(vals)} seeds & {pct(a[:, 0].mean())} & {pct(a[:, 1].mean())} & {spct(a[:, 0].mean() - dir_value(ref, 'ar->en')) if ref else '--'} \\\\"
                )
            headline.append((recipe, vals))
        recipe, vals = headline[0]
        a = np.array(vals)
        self.num("NumConfSeeds", ", ".join(str(s) for s in sorted(found[recipe])))
        self.num("NumConfArEnMean", pct(a[:, 0].mean()))
        self.num("NumConfMacroMean", pct(a[:, 1].mean()))
        if ref:
            self.num("NumConfDeltaArEn", spct(a[:, 0].mean() - dir_value(ref, "ar->en")))
            self.num("NumConfDeltaMacro", spct(a[:, 1].mean() - dir_value(ref, "macro")))
        if len(vals) > 1:
            self.num("NumConfSd", pct(a[:, 0].std(ddof=1)))
        return (
            "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nRecipe, seed & "
            + DIR_TEX["ar->en"]
            + " & macro & $\\Delta$ "
            + DIR_TEX["ar->en"]
            + " vs bge-m3 \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # ------------------------------------------------------------------------------------------------ licences / data card
    def tab_licences(self) -> str:
        card = load_json(V2 / "data" / "licences.json")
        if card is None:
            return todo("licences.json missing: run report/verify_licences.py")
        rows = []
        for s in card["sources"]:
            lic = s.get("hub_licence") or "no licence tag on the Hub card"
            rows.append(f"{esc(s['name'])} & {ROLE_TEXT.get(s['role'], s['role'])} & {esc(lic)} \\\\")
        rows.append(r"\midrule")
        rows.append(
            f"Weights of v2-m3, v2-base, v2-small & released models & {todo('licence of the weights: author decision (training data include share-alike sources)')} \\\\"
        )
        self.num("NumLicChecked", card["checked"])
        self.num("NumLicSources", str(len(card["sources"])))
        return (
            "\\begin{tabular}{@{}p{0.34\\textwidth}p{0.22\\textwidth}p{0.3\\textwidth}@{}}\n\\toprule\nSource & Role & Licence (Hub card) \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def numbers_licence_card(self) -> None:
        """Licence macros of the sources that are not in the v1 manifest, from the verified data card."""
        card = load_json(V2 / "data" / "licences.json")
        if not card:
            return
        pretty = {"apache-2.0": "Apache-2.0", "mit": "MIT", "cc-by-sa-4.0": "CC BY-SA 4.0"}
        by = {s["id"]: s for s in card["sources"]}
        for ident, macro in (
            ("google/gemma-4-26B-A4B-it", "LicGemma"),
            ("miracl/miracl", "LicMiracl"),
            ("BAAI/bge-m3", "LicBge"),
            ("intfloat/multilingual-e5-base", "LicEBase"),
        ):
            lic = (by.get(ident) or {}).get("hub_licence")
            if lic:
                self.num(macro, pretty.get(lic, esc(lic)))

    # ------------------------------------------------------------------------------------------------ WiSE curve numbers
    def curve_numbers(self) -> bool:
        points, selected = MF.load_curve(V2 / "alpha_curve.json")
        if not points:
            return False
        self.num("NumCurvePoints", str(len(points)))
        if selected is not None:
            self.num("NumCurveSelected", f"{selected:g}")
        at = {p.alpha: p for p in points}
        if selected in at:
            for key, macro in (("ar->en", "NumCurveArEnSel"), ("miracl", "NumCurveMiraclSel")):
                if key in at[selected].values:
                    self.num(macro, f"{at[selected].values[key]:.1f}")
        if 0.0 in at and "miracl" in at[0.0].values:
            self.num("NumCurveMiraclZero", f"{at[0.0].values['miracl']:.1f}")
            bar = (
                float(load_json(V2 / "data" / "protocol.json")["guard_bar_miracl_points"]["value"])
                if load_json(V2 / "data" / "protocol.json")
                else 1.0
            )
            ok = [
                p for p in points if "miracl" in p.values and "ar->en" in p.values and p.values["miracl"] >= at[0.0].values["miracl"] - bar
            ]
            if ok:
                best = max(ok, key=lambda p: p.values["ar->en"])
                self.num("NumCurveBarAlpha", f"{best.alpha:g}")
                self.num("NumCurveBarArEn", f"{best.values['ar->en']:.1f}")
        return True

    # ------------------------------------------------------------------------------------------------ pre-registration appendix
    def prereg_tex(self) -> str:
        trail = load_json(V2 / "data" / "prereg_trail.json")
        if trail is None:
            return todo("prereg_trail.json missing: run report/pull_results.sh")
        out = []
        for e in trail["entries"]:
            title = latex_text(e["title"]) + (f" ({latex_text(e['timestamp'])})" if e.get("timestamp") else "")
            if e.get("missing"):
                out.append(f"\\paragraph{{{title}}} {todo('entry not found in the lab notebook')}")
                continue
            lines = [ln.strip() for ln in e["text"].split("\n") if ln.strip()]
            lines = [ln[3:] if ln.startswith("## ") else ln for ln in lines]  # the markdown heading marker is not part of the quoted rule
            body = "\n\n".join(latex_text(ln) for ln in lines)
            out.append(f"\\paragraph{{{title}}}\n{{\\footnotesize\\raggedright\n{body}\\par}}")
        return "\n\n".join(out) + "\n"

    # ------------------------------------------------------------------------------------------------ claims
    def study_claims(self) -> list[tuple[str, bool | None]]:
        out: list[tuple[str, bool | None]] = []
        s0, s25, sfull = (dev_value(k, "ar->en") for k in ("p1_r0", "p2_s_f25", "p2_s_full"))
        if None not in (s0, s25, sfull):
            out.append(
                ("scaling: the full translation set beats no new translations on dev ar->en by more than one point", sfull - s0 > 0.01)
            )
            out.append(
                (
                    "scaling: the smallest added batch already gives more than half of the full dev ar->en gain",
                    (s25 - s0) > 0.5 * (sfull - s0),
                )
            )
        gains = getattr(self, "_neg_gains", [])
        if gains:
            out.append(("negative results: no listed change raises dev macro by more than 0.5 points", max(g[0] for g in gains) < 0.5))
            out.append(
                (
                    "negative results: no listed bge-m3 recipe change beats the plain fine-tune on dev macro",
                    all(g[0] <= 0.05 for g in gains if g[2] == "m3"),
                )
            )
        comps = self.paired_comparisons()
        c = next((c for c in comps if "bge-m3" in c.ref and "e5" not in c.ref), None)
        if c is not None:
            st = PD.paired_stats(c.scores)
            out.append(
                ("MTEB(ara): the new v2-m3 vs bge-m3 Mean (Task) difference has a CI that includes 0", st["ci_lo"] <= 0 <= st["ci_hi"])
            )
            out.append(("MTEB(ara): v2-m3 vs bge-m3 wins and losses are balanced (sign test p > 0.05)", st["sign_p"] > 0.05))
            pre = c.precomputed
            if "mean_diff" in pre:
                out.append(
                    (
                        "MTEB(ara): recomputed mean difference agrees with the file's precomputed value",
                        abs(pre["mean_diff"] - st["mean_diff"]) < 1e-6,
                    )
                )
            if "wins" in pre:
                out.append(
                    ("MTEB(ara): recomputed wins/losses agree with the file", (pre["wins"], pre["losses"]) == (st["wins"], st["losses"]))
                )
        if c is not None and self.subsets:
            cross = PD.paired_stats(PD.subset_comparison(self.subsets, c.new, c.ref, "cross"))
            mono = PD.paired_stats(PD.subset_comparison(self.subsets, c.new, c.ref, "mono"))
            if cross["n"]:
                out.append(("MTEB(ara): v2-m3 loses on no Arabic<->English subset task relative to bge-m3", cross["losses"] == 0))
                out.append(("MTEB(ara): v2-m3 gains on the Arabic<->English subsets (CI above 0)", cross["ci_lo"] > 0))
            if mono["n"]:
                out.append(
                    (
                        "MTEB(ara): the Arabic-only subset difference between v2-m3 and bge-m3 has a CI that includes 0",
                        mono["ci_lo"] <= 0 <= mono["ci_hi"],
                    )
                )
        lb = load_json(MTEB / "leaderboard.json") or {}
        refs = PD.from_leaderboard(lb, LEADERBOARD_PAIRS)
        ref = next((c for c in refs if c.label == "bge-m3 vs mE5-base"), None)
        if ref is not None:
            st = PD.paired_stats(ref.scores)
            out.append(("MTEB(ara): the leaderboard's clearest gap (bge-m3 vs mE5-base) has a CI above 0", st["ci_lo"] > 0))
        return out
