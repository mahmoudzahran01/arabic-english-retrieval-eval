"""Guard tasks, the contamination split, the exact-copy leak and the reranker pairing: tables and number macros.

`GuardTables` is a mixin of `make_tables.Tables` (needs `self.nums`, `self.num`, `self.dir_of`, `self.sys`). Inputs, all under results/retrieval_v2:
  mteb_board.json        MTEB(ara) per-task scores of the models (the guard-task scores come from here, the same run as the leaderboard table)
  guard_split/*.json     per-query scores of MIRACL-HN.v2 ar and NamaaMrTydi, guard_split/_labels.json the contamination label of every query
  leak_exact.json        exact copies of evaluation queries among the kept training rows (read-only check of the lab)
  data/prereg_trail.json counts quoted by the lab notebook's correction entry (exact normalised copies of TyDi-ar train questions)
  <stem>>rr_ft_bce_s13   the v1 cross-encoder on the top-50 of a first stage
"""

from __future__ import annotations

import numpy as np
from report.tabcommon import DIR_TEX, V2, dir_value, esc, load_board, load_json, load_system, pct, sfmt, spct, thousands, todo

from deploy import summarize as DS

GUARD_TASKS = [
    ("MIRACLRetrievalHardNegatives.v2", "MIRACL-HN", "Miracl"),
    ("NamaaMrTydiReranking", "Mr.TyDi rerank", "Namaa"),
    ("SemRel24STS", "SemRel24", "SemRel"),
]
# table label, model name in mteb_board.json, macro suffix
GUARD_MODELS = [
    ("mE5-small", "intfloat__multilingual-e5-small", "ESmall"),
    ("v1", "mzahran1__arabic-english-e5-small-retrieval", "VOne"),
    ("mE5-base", "intfloat__multilingual-e5-base", "EBase"),
    ("bge-m3", "BAAI__bge-m3", "Bge"),
    ("v2-small", "final_small", "VTwoS"),
    ("v2-base", "final_base", "VTwoB"),
    ("v2-m3", "final_m3", "VTwoM"),
]
GUARD_DELTAS = [
    ("v2-m3 $-$ bge-m3", "final_m3", "BAAI__bge-m3", ""),
    ("v2-base $-$ mE5-base", "final_base", "intfloat__multilingual-e5-base", "B"),
    ("v2-small $-$ mE5-small", "final_small", "intfloat__multilingual-e5-small", "S"),
]
# split-table rows: label, guard_split file stem
SPLIT_MODELS = [
    ("bge-m3", "bge-m3"),
    ("mE5-base", "e5-base"),
    ("mE5-small", "e5-small"),
    ("v1", "v1"),
    ("v2-m3 raw FT", "m3_b_s13"),
    ("v2-m3", "final_m3"),
    ("v2-base", "final_base"),
    ("v2-small", "final_small"),
]
SPLIT_PAIRS = [
    ("v2-m3 $-$ bge-m3", "final_m3", "bge-m3", ""),
    ("v2-base $-$ mE5-base", "final_base", "e5-base", "B"),
    ("v2-small $-$ mE5-small", "final_small", "e5-small", "S"),
]
SPLIT_TASKS = [("miracl", "MIRACL-HN ar", "Miracl"), ("namaa", "Mr.TyDi rerank", "Namaa")]
RERANK_ROWS = [
    ("bge-m3", "base_bge-m3", "Bge"),
    ("mE5-base", "base_e5-base", "EBase"),
    ("mE5-small", "base_e5-small", "ESmall"),
    ("v2-m3", "final_m3", "VTwoM"),
    ("v2-base", "st_b_soup", "VTwoB"),
    ("v2-small", "st_s_soup", "VTwoS"),
]
N_BOOT = 2000
GUARD_DEFAULTS = [
    *[f"NumGuard{m}{t}" for _, _, m in GUARD_MODELS for _, _, t in GUARD_TASKS],
    *[f"NumGuard{g}Gap{t}" for g in ("", "B", "S") for _, _, t in GUARD_TASKS],
    *[
        f"NumSplit{k}{task}"
        for k in ("CleanGap", "CleanGapLo", "CleanGapHi", "FlagGap", "FlagGapLo", "FlagGapHi", "CleanGapB", "CleanGapS", "FlagGapB", "FlagGapS")
        for _, _, task in SPLIT_TASKS
    ],
    "NumMiraclQueries", "NumMiraclFlagged", "NumMiraclCleanN", "NumMiraclExact", "NumMiraclJaccardFlagged", "NumMiraclJaccardFlaggedBoth", "NumMiraclNearDup",
    "NumNamaaQueries", "NumNamaaFlagged", "NumNamaaCleanN",
    "NumLeakRows", "NumLeakDistinct", "NumLeakTranslated", "NumLeakShare", "NumLeakNamaa", "NumLeakEffectMin", "NumLeakEffectMax",
    *[f"NumRerank{m}{k}" for _, _, m in RERANK_ROWS for k in ("ArEn", "ArEnFirst", "Macro", "MacroFirst")],
    "NumRerankDeltaArEn", "NumRerankDeltaArEnLo", "NumRerankDeltaArEnHi", "NumRerankDeltaMacro", "NumRerankDeltaMacroLo", "NumRerankDeltaMacroHi",
    "NumRerankGainB", "NumRerankGainS",
]  # fmt: skip


def _boot(a: np.ndarray, b: np.ndarray, seed: int = 13) -> tuple[float, float, float]:
    """Mean of b - a and its percentile bootstrap interval over queries (paired)."""
    d = b - a
    idx = np.random.default_rng(seed).integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


class GuardTables:
    # ------------------------------------------------------------------------------------------------ guard tasks (MTEB board)
    def board(self) -> dict[str, dict]:
        return load_board()

    def guard_scores(self) -> dict[str, dict[str, float]]:
        """{model name in the board: {task: score x100}}."""
        return {m: {t: r["per_task"][t] for t, _, _ in GUARD_TASKS if t in r.get("per_task", {})} for m, r in self.board().items()}

    def tab_guards(self) -> str:
        g = self.guard_scores()
        rows = []
        for label, model, short in GUARD_MODELS:
            cells = [f"{g[model][t]:.1f}" if model in g and t in g[model] else todo("missing") for t, _, _ in GUARD_TASKS]
            rows.append(f"{esc(label)} & " + " & ".join(cells) + r" \\")
            for t, _, tshort in GUARD_TASKS:
                if model in g and t in g[model]:
                    self.num(f"NumGuard{short}{tshort}", f"{g[model][t]:.1f}")
        rows.append(r"\midrule")
        for label, new, ref, suffix in GUARD_DELTAS:
            if new in g and ref in g and all(t in g[new] and t in g[ref] for t, _, _ in GUARD_TASKS):
                rows.append(f"{label} & " + " & ".join(sfmt(g[new][t] - g[ref][t]) for t, _, _ in GUARD_TASKS) + r" \\")
                for t, _, tshort in GUARD_TASKS:
                    self.num(f"NumGuard{suffix}Gap{tshort}", sfmt(g[new][t] - g[ref][t]))
        head = " & ".join(esc(s) for _, s, _ in GUARD_TASKS)
        return (
            "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nModel & "
            + head
            + " \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # ------------------------------------------------------------------------------------------------ contamination split
    def _split_arrays(self, stem: str, task: str, labels: dict) -> tuple[np.ndarray, np.ndarray] | None:
        j = load_json(V2 / "guard_split" / f"{stem}.json")
        if j is None or task not in j:
            return None
        ids = list(labels[task])
        scores = np.array([j[task][i] for i in ids], dtype=np.float64)
        flagged = np.array([labels[task][i] != "" for i in ids])
        return scores, flagged

    def clean_split(self) -> tuple[str, dict, str]:
        """(table with the per-subset means, stats, table with the paired per-subset differences)."""
        labels = load_json(V2 / "guard_split" / "_labels.json")
        if labels is None:
            return todo("guard_split missing"), {}, todo("guard_split missing")
        rows, stats = [], {}
        for label, stem in SPLIT_MODELS:
            cells, ok = [], True
            for task, _, _ in SPLIT_TASKS:
                arr = self._split_arrays(stem, task, labels)
                if arr is None:
                    ok = False
                    break
                v, fl = arr
                stats[(stem, task)] = (v.mean(), v[~fl].mean(), int((~fl).sum()), int(fl.sum()), v[fl].mean())
                stats[(task, "n")] = (int((~fl).sum()), int(fl.sum()))
                cells += [pct(v.mean()), pct(v[~fl].mean()), pct(v[fl].mean())]
            rows.append(
                f"{esc(label)} & " + " & ".join(cells) + r" \\"
                if ok
                else f"{esc(label)} & \\multicolumn{{6}}{{c}}{{{todo('missing')}}} \\\\"
            )
        n = {t: stats.get((t, "n")) for t, _, _ in SPLIT_TASKS}
        if None in n.values():
            return todo("guard_split incomplete"), stats, todo("guard_split incomplete")
        head2 = " & ".join(rf"\multicolumn{{3}}{{c}}{{{name} (n={n[t][0]}+{n[t][1]})}}" for t, name, _ in SPLIT_TASKS) + r" \\"
        tab = (
            "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & "
            + head2
            + "\nModel & all & clean & flagged & all & clean & flagged \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )
        return tab, stats, self._split_paired(labels)

    def _split_paired(self, labels: dict) -> str:
        rows = []
        for task, name, tshort in SPLIT_TASKS:
            for subset, flag in (("clean", False), ("flagged", True)):
                cells = []
                size = None
                for _, new, ref, suffix in SPLIT_PAIRS:
                    a, b = self._split_arrays(ref, task, labels), self._split_arrays(new, task, labels)
                    if a is None or b is None:
                        cells.append(todo("missing"))
                        continue
                    mask = a[1] == flag
                    size = int(mask.sum())
                    d, lo, hi = _boot(a[0][mask], b[0][mask])
                    cells.append(f"{spct(d)} [{spct(lo)}, {spct(hi)}]")
                    key = ("Clean" if not flag else "Flag") + "Gap"
                    self.num(f"NumSplit{key}{suffix}{tshort}", spct(d))
                    if suffix == "":
                        self.num(f"NumSplit{key}Lo{tshort}", spct(lo))
                        self.num(f"NumSplit{key}Hi{tshort}", spct(hi))
                rows.append(f"{name} & {subset} & {size} & " + " & ".join(cells) + r" \\")
        head = " & ".join(lbl for lbl, *_ in SPLIT_PAIRS)
        return (
            "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\nTask & subset & n & "
            + head
            + " \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    def numbers_contamination(self, stats: dict) -> None:
        labels = load_json(V2 / "guard_split" / "_labels.json")
        audit = load_json(V2 / "data" / "audit_ood.json")
        if audit:
            a = audit["MIRACLRetrievalHardNegatives.v2/ar/dev"]
            self.num("NumMiraclNearDup", thousands(a["queries_near_duplicate_of_a_train_query"]))
        overlap = load_json(V2 / "exact_overlap.json")
        if overlap:
            self.num("NumMiraclExact", thousands(overlap["exact_normalised_tydi_ar_train"]))
            self.num("NumMiraclJaccardFlagged", thousands(overlap["flagged_old_detector_tydi"]))
            self.num("NumMiraclJaccardFlaggedBoth", thousands(overlap["flagged_old_detector_tydi_or_miracl_train"]))
        else:
            for name in ("NumMiraclExact", "NumMiraclJaccardFlagged", "NumMiraclJaccardFlaggedBoth"):
                self.num(name, todo("exact_overlap.json missing"))
        if labels:
            m, nm = labels["miracl"], labels["namaa"]
            self.num("NumMiraclQueries", thousands(len(m)))
            self.num("NumMiraclFlagged", thousands(sum(v != "" for v in m.values())))
            self.num("NumMiraclCleanN", thousands(sum(v == "" for v in m.values())))
            self.num("NumNamaaQueries", thousands(len(nm)))
            self.num("NumNamaaFlagged", thousands(sum(v != "" for v in nm.values())))
            self.num("NumNamaaCleanN", thousands(sum(v == "" for v in nm.values())))
        g, official = self.guard_scores(), None
        official = g.get("BAAI__bge-m3", {}).get("MIRACLRetrievalHardNegatives.v2")
        bge_all = stats.get(("bge-m3", "miracl"), (None,))[0]
        if bge_all is not None and official is not None and abs(100 * bge_all - official) > 1.0:
            self.num(
                "TodoCleanProtocol",
                todo(f"split scores differ from the MTEB protocol for bge-m3 by {abs(100 * bge_all - official):.1f} points; rerun pending"),
            )
        else:
            self.num("TodoCleanProtocol", "")
        self.numbers_leak(labels)

    def contamination_claims(self) -> list[tuple[str, bool | None]]:
        """Cross-checks between the exact-overlap file, the corrected labels and the notebook prose."""
        overlap = load_json(V2 / "exact_overlap.json")
        if not overlap:
            return [("contamination: exact_overlap.json is present", None)]
        out: list[tuple[str, bool | None]] = []
        exact = overlap["exact_normalised_tydi_ar_train"]
        fixed = overlap["flagged_fixed_detector_tydi_or_miracl_train"]
        out.append(("contamination: raw and normalised exact-copy counts agree", overlap["exact_raw_tydi_ar_train"] == exact))
        out.append(("contamination: the exact copies are all among the queries the corrected detector flags", exact <= fixed))
        labels = load_json(V2 / "guard_split" / "_labels.json")
        if labels:
            m = labels["miracl"]
            out.append(("contamination: the split covers the queries of the overlap check", len(m) == overlap["n_queries"]))
            out.append(
                ("contamination: the split flags as many queries as the corrected detector", sum(v != "" for v in m.values()) == fixed)
            )
        trail = load_json(V2 / "data" / "prereg_trail.json") or {}
        fix = next((e for e in trail.get("entries", []) if e["id"] == "contam_fix"), {})
        nums = fix.get("numbers", {})
        if nums:
            out.append(
                (
                    "contamination: the notebook quotes the counts of the overlap file (exact copies, old detector)",
                    (nums.get("exact_copies"), nums.get("queries"), nums.get("jaccard_flagged"))
                    == (exact, overlap["n_queries"], overlap["flagged_old_detector_tydi"]),
                )
            )
        return out

    def numbers_leak(self, labels: dict | None) -> None:
        leak = load_json(V2 / "leak_exact.json")
        if not leak:
            return
        t = leak["tasks"]["miracl_hn_v2_ar_dev"]
        self.num("NumLeakRows", str(t["exact_matches_in_kept_training_rows"]))
        self.num("NumLeakDistinct", str(t["distinct_questions"]))
        self.num("NumLeakTranslated", str(t["translated_copies_of_those_rows_in_kept_training_rows"]))
        self.num("NumLeakShare", f"{100 * t['share_of_normalised_eval_queries']:.1f}")
        self.num("NumLeakNamaa", str(leak["tasks"]["namaa_mrtydi_test"]["exact_matches_in_kept_training_rows"]))
        if labels:  # effect of dropping the leaked evaluation queries on every model's MIRACL-HN score (from the cached per-query nDCG@10)
            leaked = set(t["leaked_eval_qids"])
            effects = []
            for _, stem in SPLIT_MODELS:
                j = load_json(V2 / "guard_split" / f"{stem}.json")
                if j and "miracl" in j:
                    ids = list(j["miracl"])
                    vals = np.array([j["miracl"][i] for i in ids])
                    keep = np.array([i not in leaked for i in ids])
                    effects.append(100 * (vals[keep].mean() - vals.mean()))
            if effects:
                self.num("NumLeakEffectMin", sfmt(min(effects)))
                self.num("NumLeakEffectMax", sfmt(max(effects)))

    # ------------------------------------------------------------------------------------------------ reranker pairing
    def tab_rerank(self) -> str:
        rows, loaded = [], {}
        for label, stem, short in RERANK_ROWS:
            first, rr = load_system(stem), load_system(f"{stem}>>rr_ft_bce_s13")
            loaded[stem] = (first, rr)
            if first is None or rr is None:
                rows.append(f"{esc(label)} & \\multicolumn{{4}}{{c}}{{{todo('missing')}}} \\\\")
                continue
            vals = [dir_value(first, "ar->en"), dir_value(rr, "ar->en"), dir_value(first, "macro"), dir_value(rr, "macro")]
            rows.append(f"{esc(label)} & " + " & ".join(pct(v) for v in vals) + r" \\")
            for k, v in zip(("ArEnFirst", "ArEn", "MacroFirst", "Macro"), vals, strict=True):
                self.num(f"NumRerank{short}{k}", pct(v))
        m, v = loaded.get("base_bge-m3"), loaded.get("final_m3")
        if m and v and all(m) and all(v) and m[1]["perq"] and v[1]["perq"]:
            ci = DS.macro_bootstrap(m[1]["perq"], v[1]["perq"], self.dir_of)
            for d, key in (("ar->en", "ArEn"), ("macro", "Macro")):
                self.num(f"NumRerankDelta{key}", spct(dir_value(v[1], d) - dir_value(m[1], d)))
                self.num(f"NumRerankDelta{key}Lo", spct(ci[d]["lo"]))
                self.num(f"NumRerankDelta{key}Hi", spct(ci[d]["hi"]))
        for key, student, base in (("B", "st_b_soup", "base_e5-base"), ("S", "st_s_soup", "base_e5-small")):
            s, b = loaded.get(student), loaded.get(base)
            if s and b and s[1] and b[1]:
                self.num(f"NumRerankGain{key}", spct(dir_value(s[1], "ar->en") - dir_value(b[1], "ar->en")))
        head = rf"\multicolumn{{2}}{{c}}{{{DIR_TEX['ar->en']}}} & \multicolumn{{2}}{{c}}{{macro}} \\"
        return (
            "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nFirst stage (top-50) & "
            + head
            + "\n & alone & + reranker & alone & + reranker \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )
