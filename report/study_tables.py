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
    load_board,
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
    ("3-seed soup at the pick's $\\alpha$ (versus the single seed)", "soup_a40", "w_m3b_a40", "m3", "no$^\\dagger$"),
]
SEED_WORD = {16: "Sixteen", 17: "Seventeen"}  # LaTeX macro names cannot contain digits
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
PAIRED_GROUPS = {  # (new, ref) of mteb_paired.json -> macro group
    ("final_m3", "BAAI__bge-m3"): "M",
    ("final_m3_wa75_s13", "BAAI__bge-m3"): "F",
    ("final_base", "intfloat__multilingual-e5-base"): "B",
    ("final_small", "intfloat__multilingual-e5-small"): "S",
    ("final_small", "mzahran1__arabic-english-e5-small-retrieval"): "V",
    ("w_m3b_a40_s16", "BAAI__bge-m3"): "X",
    ("w_m3b_a40_s17", "BAAI__bge-m3"): "Y",
}
PAIRED_NAMES = {
    "final_m3": "v2-m3",
    "BAAI__bge-m3": "bge-m3",
    "w_m3b_a40_s16": "v2-m3 seed 16",
    "w_m3b_a40_s17": "v2-m3 seed 17",
    "final_base": "v2-base",
    "final_small": "v2-small",
    "intfloat__multilingual-e5-base": "mE5-base",
    "intfloat__multilingual-e5-small": "mE5-small",
    "mzahran1__arabic-english-e5-small-retrieval": "v1",
}
_PAIRED_KEYS = ("MeanDiff", "CiLo", "CiHi", "Wins", "Losses", "Ties", "SignP", "Tasks", "New", "Ref")
PAIRED_DEFAULTS = (
    [f"NumPaired{g}{k}" for g in "MFBSVXY" for k in _PAIRED_KEYS]
    + [f"NumPaired{g}Cross{k}" for g in "MBS" for k in ("Diff", "CiLo", "CiHi", "Wins", "Losses", "Ties", "Tasks", "P")]
    + [f"NumPaired{g}{k}" for g in "MBS" for k in ("BestTask", "BestGain", "WorstTask", "WorstGain")]
    + ["NumPairedMNamaaGain", "NumPairedMMiraclGain", "NumPairedMSecondTask", "NumPairedMSecondGain"]
)
CURVE_DEFAULTS = [
    *[f"NumCurve{n}{t}" for n in ("ArEn", "Macro", "Miracl", "Namaa", "SemRel") for t in ("Zero", "Sel", "Raw")],
    "NumCurveShareArEn",
    "NumCurveShareMacro",
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
CONF_DEFAULTS = [
    "NumConfSeeds", "NumConfFrozen",
    *[f"Num{t}{k}" for t in ("Conf", "ConfRaw") for k in ("ArEnMean", "MacroMean", "DeltaArEn", "DeltaArEnLo", "DeltaArEnHi", "DeltaMacro", "DeltaMacroLo", "DeltaMacroHi", "Sd")],
    *[f"Num{t}ArEnSeed{n}" for t in ("Conf", "ConfRaw") for n in ("Sixteen", "Seventeen")],
]  # fmt: skip


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

        def row(st: dict, label: str) -> str:
            return f"{label} & {st['n']} & {sfmt(st['mean_diff'], 2)} & [{sfmt(st['ci_lo'], 2)}, {sfmt(st['ci_hi'], 2)}] & {st['wins']}/{st['losses']}/{st['ties']} & {_fmt_p(st['sign_p'])} \\\\"

        if not comps:
            rows.append(f"v2-m3 vs bge-m3 & \\multicolumn{{5}}{{c}}{{{todo('mteb_paired.json pending')}}} \\")
        alpha_first = (load_json(V2 / "models" / "wm_a75" / "merge.json") or {}).get("alpha")
        for c in comps:
            names = dict(PAIRED_NAMES)
            if alpha_first is not None:
                names["final_m3_wa75_s13"] = rf"first candidate (WiSE $\alpha{{=}}{alpha_first:g}$)"
            label = f"{names.get(c.new, esc(c.new))} vs {names.get(c.ref, esc(c.ref))}"
            st = PD.paired_stats(c.scores)
            rows.append(row(st, label))
            group = PAIRED_GROUPS.get((c.new, c.ref))
            if group:
                self._register_paired(group, c, st)
            if c.new not in ("final_m3", "final_base", "final_small"):
                continue
            for name, kind in (("ar-en cross-lingual subsets", "cross"), ("ar-only subsets", "mono")):
                sc = c.subsets.get(name) or PD.subset_comparison(self.subsets, c.new, c.ref, kind)
                if sc:
                    stat = PD.paired_stats(sc)
                    rows.append(row(stat, f"\\quad {name}"))
                    if kind == "cross" and group:
                        self._register_cross(group, stat)
                elif kind == "cross":
                    rows.append(
                        f"\\quad {name} & \\multicolumn{{5}}{{c}}{{{todo('per-subset scores not available (mteb_subsets.json)')}}} \\\\"
                    )
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{6}{@{}l}{\emph{reference pairs from the leaderboard of baseline models}} \\")
        for c in refs:
            st = PD.paired_stats(c.scores)
            rows.append(row(st, esc(c.label)))
            if c.label == "bge-m3 vs mE5-base":
                self.num("NumPairedRefMeanDiff", sfmt(st["mean_diff"]))
                self.num("NumPairedRefCiLo", sfmt(st["ci_lo"]))
                self.num("NumPairedRefCiHi", sfmt(st["ci_hi"]))
                self.num("NumPairedRefSignP", _fmt_p(st["sign_p"]))
        return (
            "\\begin{tabular}{@{}lrrcrr@{}}\n\\toprule\nMean (Task) difference, new $-$ ref & tasks & diff & \\NumCiLevel\\% CI (tasks) & W/L/T & sign p \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def _register_cross(self, group: str, st: dict) -> None:
        self.num(f"NumPaired{group}CrossDiff", sfmt(st["mean_diff"]))
        self.num(f"NumPaired{group}CrossCiLo", sfmt(st["ci_lo"]))
        self.num(f"NumPaired{group}CrossCiHi", sfmt(st["ci_hi"]))
        self.num(f"NumPaired{group}CrossWins", str(st["wins"]))
        self.num(f"NumPaired{group}CrossLosses", str(st["losses"]))
        self.num(f"NumPaired{group}CrossTies", str(st["ties"]))
        self.num(f"NumPaired{group}CrossTasks", str(st["n"]))
        self.num(f"NumPaired{group}CrossP", _fmt_p(st["sign_p"]))

    def _register_paired(self, group: str, c: PD.Comparison, st: dict) -> None:
        self.num(f"NumPaired{group}MeanDiff", sfmt(st["mean_diff"], 2))
        self.num(f"NumPaired{group}CiLo", sfmt(st["ci_lo"], 2))
        self.num(f"NumPaired{group}CiHi", sfmt(st["ci_hi"], 2))
        self.num(f"NumPaired{group}Wins", str(st["wins"]))
        self.num(f"NumPaired{group}Losses", str(st["losses"]))
        self.num(f"NumPaired{group}Ties", str(st["ties"]))
        self.num(f"NumPaired{group}SignP", _fmt_p(st["sign_p"]))
        self.num(f"NumPaired{group}Tasks", str(st["n"]))
        self.num(f"NumPaired{group}New", f"{st['mean_new']:.2f}")
        self.num(f"NumPaired{group}Ref", f"{st['mean_ref']:.2f}")
        if group in ("M", "B", "S"):
            best = max(c.scores, key=lambda t: c.scores[t][0] - c.scores[t][1])
            worst = min(c.scores, key=lambda t: c.scores[t][0] - c.scores[t][1])
            self.num(f"NumPaired{group}BestTask", esc(best.replace(".v2", "")))
            self.num(f"NumPaired{group}BestGain", sfmt(c.scores[best][0] - c.scores[best][1]))
            self.num(f"NumPaired{group}WorstTask", esc(worst.replace(".v2", "")))
            self.num(f"NumPaired{group}WorstGain", sfmt(c.scores[worst][0] - c.scores[worst][1]))
            if group == "M":
                gains = sorted(((n - r, t) for t, (n, r) in c.scores.items()), reverse=True)
                self.num(
                    "NumPairedMNamaaGain",
                    sfmt(c.scores["NamaaMrTydiReranking"][0] - c.scores["NamaaMrTydiReranking"][1])
                    if "NamaaMrTydiReranking" in c.scores
                    else todo("missing"),
                )
                self.num(
                    "NumPairedMMiraclGain",
                    sfmt(c.scores["MIRACLRetrievalHardNegatives.v2"][0] - c.scores["MIRACLRetrievalHardNegatives.v2"][1])
                    if "MIRACLRetrievalHardNegatives.v2" in c.scores
                    else todo("missing"),
                )
                self.num("NumPairedMSecondTask", esc(gains[1][1].replace(".v2", "")))
                self.num("NumPairedMSecondGain", sfmt(gains[1][0]))

    # ------------------------------------------------------------------------------------------------ confirmatory seeds
    def confirmatory_systems(self) -> dict[str, dict[int, str]]:
        found: dict[str, dict[int, str]] = {}
        for p in sorted(V2.glob("*_s1[67].test.json")):
            m = CONFIRM_RE.match(p.name)
            if m:
                found.setdefault(m.group("recipe"), {})[int(m.group("seed"))] = p.name[: -len(".test.json")]
        return found

    def confirmatory_stems(self) -> tuple[dict[int, str], dict[int, str], dict]:
        """(WiSE-recipe stems by seed, raw-fine-tune stems by seed, confirmatory.json) for the seeds that were trained after the freeze."""
        conf = load_json(V2 / "confirmatory.json") or {}
        if conf:
            seeds = [int(x) for x in conf.get("new_seeds", [])]
            blended = {int(re.search(r"_s(\d+)$", k).group(1)): k for k in conf.get("blended_models", {})}
            raw = {int(re.search(r"_s(\d+)$", k).group(1)): k for k in conf.get("raw_models", {})}
            return {s_: blended[s_] for s_ in seeds if s_ in blended}, {s_: raw[s_] for s_ in seeds if s_ in raw}, conf
        found = self.confirmatory_systems()  # fallback: any <recipe>_s16 / _s17 test file
        first = next(iter(found.values()), {})
        return dict(first), {}, conf

    def tab_confirmatory(self) -> str:
        blended, raw, conf = self.confirmatory_stems()
        ref = load_system("base_bge-m3")
        board = load_board()
        head = (
            "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nModel, seed & "
            + DIR_TEX["ar->en"]
            + " & macro & $\\Delta$ "
            + DIR_TEX["ar->en"]
            + " vs bge-m3 & MTEB Mean (Task) \\\\\n\\midrule\n"
        )
        if not blended:
            self.num("NumConfSeeds", todo("seeds 16 and 17 pending"))
            return (
                head
                + f"seeds 16 and 17 & \\multicolumn{{4}}{{c}}{{{todo('trained after the recipe was frozen; pending')}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
            )
        rows = []
        for label, stems, tag in (("v2-m3 recipe (WiSE)", blended, "Conf"), ("raw fine-tune ($\\alpha{=}1$)", raw, "ConfRaw")):
            if not stems:
                continue
            systems = {seed: load_system(stem) for seed, stem in sorted(stems.items())}
            for seed, stem in sorted(stems.items()):
                sysd = systems[seed]
                if sysd is None:
                    rows.append(f"{label}, seed {seed} & \\multicolumn{{4}}{{c}}{{{todo('result missing')}}} \\\\")
                    continue
                ar, mac = dir_value(sysd, "ar->en"), dir_value(sysd, "macro")
                cell = "--"
                if ref and ref["perq"] and sysd["perq"]:
                    ci = DS.macro_bootstrap(ref["perq"], sysd["perq"], self.dir_of)
                    cell = f"{spct(ar - dir_value(ref, 'ar->en'))} [{spct(ci['ar->en']['lo'])}, {spct(ci['ar->en']['hi'])}]"
                mt = f"{board[stem]['Mean(Task)']:.2f}" if stem in board else "--"
                rows.append(f"{label}, seed {seed} & {pct(ar)} & {pct(mac)} & {cell} & {mt} \\\\")
                self.num(f"Num{tag}ArEnSeed{SEED_WORD.get(seed, 'Other')}", pct(ar))
            ok = [x for x in systems.values() if x]
            if len(ok) == len(systems) and len(ok) > 1 and ref and ref["perq"] and all(x["perq"] for x in ok):
                a = np.array([(dir_value(x, "ar->en"), dir_value(x, "macro")) for x in ok])
                # paired bootstrap of the seed-averaged per-query scores against bge-m3
                avg = {sl: np.mean([x["perq"][sl] for x in ok], axis=0) for sl in ok[0]["perq"]}
                ci = DS.macro_bootstrap(ref["perq"], avg, self.dir_of)
                mts = [board[stem]["Mean(Task)"] for stem in stems.values() if stem in board]
                mt = f"{np.mean(mts):.2f}" if len(mts) == len(stems) else "--"
                rows.append(
                    f"{label}, mean of {len(ok)} seeds & {pct(a[:, 0].mean())} & {pct(a[:, 1].mean())} & {spct(a[:, 0].mean() - dir_value(ref, 'ar->en'))} [{spct(ci['ar->en']['lo'])}, {spct(ci['ar->en']['hi'])}] & {mt} \\\\"
                )
                self.num(f"Num{tag}ArEnMean", pct(a[:, 0].mean()))
                self.num(f"Num{tag}MacroMean", pct(a[:, 1].mean()))
                self.num(f"Num{tag}DeltaArEn", spct(a[:, 0].mean() - dir_value(ref, "ar->en")))
                self.num(f"Num{tag}DeltaArEnLo", spct(ci["ar->en"]["lo"]))
                self.num(f"Num{tag}DeltaArEnHi", spct(ci["ar->en"]["hi"]))
                self.num(f"Num{tag}DeltaMacro", spct(a[:, 1].mean() - dir_value(ref, "macro")))
                self.num(f"Num{tag}DeltaMacroLo", spct(ci["macro"]["lo"]))
                self.num(f"Num{tag}DeltaMacroHi", spct(ci["macro"]["hi"]))
                self.num(f"Num{tag}Sd", pct(a[:, 0].std(ddof=1)))
            if rows and tag == "Conf" and raw:
                rows.append(r"\midrule")
        self.num("NumConfSeeds", " and ".join(str(x) for x in sorted(blended)))
        if conf.get("recipe_frozen_at"):
            self.num("NumConfFrozen", conf["recipe_frozen_at"].replace("T", " ").replace("+03:00", " (UTC+3)"))
        return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"

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
        for tag, alpha in (("Zero", 0.0), ("Sel", selected), ("Raw", 1.0)):
            if alpha in at:
                for key, name in (("ar->en", "ArEn"), ("macro", "Macro"), ("miracl", "Miracl"), ("namaa", "Namaa"), ("semrel", "SemRel")):
                    if key in at[alpha].values:
                        self.num(f"NumCurve{name}{tag}", f"{at[alpha].values[key]:.1f}")
        if all(a in at for a in (0.0, 1.0)) and selected in at:
            for key, name in (("ar->en", "ArEn"), ("macro", "Macro")):
                lo, hi, mid = (at[a].values.get(key) for a in (0.0, 1.0, selected))
                if None not in (lo, hi, mid) and hi != lo:
                    self.num(f"NumCurveShare{name}", f"{100 * (mid - lo) / (hi - lo):.0f}")
        if 0.0 in at and "miracl" in at[0.0].values:
            proto = load_json(V2 / "data" / "protocol.json") or {}
            bar = float(proto["guard_bar_miracl_points"]["value"]) if "guard_bar_miracl_points" in proto else 1.0
            ref = round(at[0.0].values["miracl"], 1)
            # the bar is applied to the one-decimal numbers that are reported
            ok = [p for p in points if "miracl" in p.values and "ar->en" in p.values and round(p.values["miracl"], 1) >= ref - bar - 1e-9]
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
                    "negative results: no listed bge-m3 recipe change beats the plain fine-tune on dev macro by more than 0.15",
                    all(g[0] <= 0.15 for g in gains if g[2] == "m3"),
                )
            )
            soup = [g for g in gains if g[3].startswith("3-seed soup")]
            out.append(
                (
                    "the soup at the pick's alpha is not better on dev macro than the single seed",
                    all(g[0] <= 0.05 for g in soup) if soup else None,
                )
            )
        comps = {(c.new, c.ref): c for c in self.paired_comparisons()}
        for key, nm in (
            (("final_m3", "BAAI__bge-m3"), "v2-m3 vs bge-m3"),
            (("w_m3b_a40_s16", "BAAI__bge-m3"), "confirmatory seed 16 vs bge-m3"),
            (("w_m3b_a40_s17", "BAAI__bge-m3"), "confirmatory seed 17 vs bge-m3"),
        ):
            c = comps.get(key)
            if c is None:
                out.append((f"MTEB(ara): {nm} is present in mteb_paired.json", None))
                continue
            st = PD.paired_stats(c.scores)
            out.append((f"MTEB(ara): {nm}: CI of the Mean (Task) difference includes 0", st["ci_lo"] <= 0 <= st["ci_hi"]))
            out.append((f"MTEB(ara): {nm}: wins and losses are balanced (sign test p > 0.05)", st["sign_p"] > 0.05))
            pre = c.precomputed
            if "mean_diff" in pre:
                out.append(
                    (f"MTEB(ara): {nm}: recomputed mean difference agrees with the file", abs(pre["mean_diff"] - st["mean_diff"]) < 1e-6)
                )
            if "wins" in pre:
                out.append(
                    (
                        f"MTEB(ara): {nm}: recomputed wins/losses/ties agree with the file",
                        (pre["wins"], pre["losses"], pre["ties"]) == (st["wins"], st["losses"], st["ties"]),
                    )
                )
        for key, nm, expect_incl in (
            (("final_base", "intfloat__multilingual-e5-base"), "v2-base vs mE5-base", True),
            (("final_small", "intfloat__multilingual-e5-small"), "v2-small vs mE5-small", True),
            (("final_small", "mzahran1__arabic-english-e5-small-retrieval"), "v2-small vs v1", False),
        ):
            c = comps.get(key)
            if c is not None:
                st = PD.paired_stats(c.scores)
                out.append(
                    (
                        f"MTEB(ara): {nm}: CI " + ("includes 0" if expect_incl else "excludes 0"),
                        (st["ci_lo"] <= 0 <= st["ci_hi"]) == expect_incl,
                    )
                )
        lb = load_json(MTEB / "leaderboard.json") or {}
        ref = next((c for c in PD.from_leaderboard(lb, LEADERBOARD_PAIRS) if c.label == "bge-m3 vs mE5-base"), None)
        if ref is not None:
            st = PD.paired_stats(ref.scores)
            out.append(("MTEB(ara): the clearest baseline gap (bge-m3 vs mE5-base) has a CI above 0", st["ci_lo"] > 0))
        c = comps.get(("final_m3", "BAAI__bge-m3"))
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
        # the two MTEB sources (baseline leaderboard, v2 board) and the two data summaries agree
        board = load_board()
        if lb and board:
            diffs = [
                abs(r["Mean (Task)"] - board[r["model_name"].replace("/", "__")]["Mean(Task)"])
                for r in lb["summary"]
                if r["model_name"].replace("/", "__") in board
            ]
            out.append(
                (
                    "the baseline models score the same on the leaderboard and in the v2 board (within 0.01)",
                    max(diffs) < 0.01 if diffs else None,
                )
            )
        card = load_json(V2 / "data_card" / "exclude_qids.json")
        summ = load_json(V2 / "data" / "exclude_summary.json")
        if card and summ:
            out.append(("the training-data summary of the report equals the lab's exclusion summary", card.get("summary") == summ))
        # reranker pairing: level
        m, v = load_system("base_bge-m3>>rr_ft_bce_s13"), load_system("final_m3>>rr_ft_bce_s13")
        if m and v and m["perq"] and v["perq"]:
            ci = DS.macro_bootstrap(m["perq"], v["perq"], self.dir_of)
            out.append(
                (
                    "with the reranker, v2-m3 and bge-m3 are level on ar->en (paired CI includes 0)",
                    ci["ar->en"]["lo"] <= 0 <= ci["ar->en"]["hi"],
                )
            )
            out.append(
                (
                    "with the reranker, v2-m3 and bge-m3 are within one point on macro",
                    abs(dir_value(v, "macro") - dir_value(m, "macro")) < 0.01,
                )
            )
        # leak: negligible
        leak = load_json(V2 / "leak_exact.json")
        if leak:
            t = leak["tasks"]["miracl_hn_v2_ar_dev"]
            out.append(("the exact leak touches fewer than 1% of the MIRACL-HN queries", t["share_of_normalised_eval_queries"] < 0.01))
        # confirmatory seeds reproduce the development seeds
        blended, _, conf = self.confirmatory_stems()
        seeds = [load_system(x) for x in blended.values()]
        dev = load_system("final_m3")
        if seeds and all(seeds) and dev:
            out.append(
                (
                    "confirmatory seeds: test ar->en within 1.5 points of the development seed 13",
                    all(abs(dir_value(x, "ar->en") - dir_value(dev, "ar->en")) < 0.015 for x in seeds),
                )
            )
        return out
