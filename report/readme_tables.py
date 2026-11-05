"""The v2 block of README.md: a model table and the caveats that carry numbers, generated from the macros of the report (so README and paper cannot disagree).

`make -C report tables` rewrites the text between START and END in README.md; tests/report/test_readme_v2.py checks that it is current.
"""

from __future__ import annotations

from pathlib import Path

from report.tabcommon import ROOT, dir_value, pct

START, END = "<!-- v2-results:start -->", "<!-- v2-results:end -->"
README = ROOT / "README.md"

# name, parameters (safetensors bytes / 4: 117.7M, 278.0M, 567.8M; multilingual-e5-large as published), Tables.sys key (None: no test run), MTEB macro stem, paired macro letter
ROWS = [
    ("**v2-m3**", "568M", "v2m", "VTwoM", "M", "bge-m3"),
    ("bge-m3 (reference)", "568M", "m3", "Bge", None, ""),
    ("multilingual-e5-large", "560M", None, "ELarge", None, ""),
    ("**v2-base**", "278M", "v2b", "VTwoB", "B", "mE5-base"),
    ("multilingual-e5-base (reference)", "278M", "e5b", "EBase", None, ""),
    ("**v2-small**", "118M", "v2s", "VTwoS", "S", "mE5-small"),
    ("multilingual-e5-small (reference)", "118M", "e5s", "ESmall", None, ""),
]


def plain(s: str) -> str:
    """A number macro (LaTeX text) as Markdown text."""
    return s.replace("$-$", "-").replace("{,}", ",").replace("\\%", "%")


class ReadmeTables:
    def _n(self, name: str) -> str:
        return plain(self.nums[name])

    def readme_v2(self) -> str:
        n = self._n
        lines = [
            "| Model | Params | Benchmark ar->en | Benchmark macro | MTEB(ara) mean | MTEB(ara) difference to the reference [95% CI] |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for name, params, key, mteb, paired, ref in ROWS:
            s = self.sys.get(key) if key else None
            test = (pct(dir_value(s, "ar->en")), pct(dir_value(s, "macro"))) if s else ("n/a", "n/a")
            diff = (
                f"{n(f'NumPaired{paired}MeanDiff')} [{n(f'NumPaired{paired}CiLo')}, {n(f'NumPaired{paired}CiHi')}] vs {ref}"
                if paired
                else ""
            )
            lines.append(f"| {name} | {params} | {test[0]} | {test[1]} | {n(f'NumMteb{mteb}Task')} | {diff} |")
        lines += [
            "",
            "Benchmark: this repository's test split (MRR@10 x 100; macro = mean of the four directions); v2-m3 is one training seed, v2-base and v2-small are weight averages of three student seeds. "
            f"MTEB(ara): the {n('NumMtebTasks')} tasks of MTEB(ara, v1), mean over tasks; the interval resamples the tasks, so it is wide.",
            "",
            f"* **On MTEB(ara), v2-m3 and bge-m3 are a statistical tie** ({n('NumPairedMNew')} vs {n('NumPairedMRef')}; difference {n('NumPairedMMeanDiff')}, "
            f"95% interval {n('NumPairedMCiLo')} to {n('NumPairedMCiHi')}; v2-m3 wins on {n('NumPairedMWins')} tasks, loses on {n('NumPairedMLosses')}, "
            f"ties on {n('NumPairedMTies')}; sign test p = {n('NumPairedMSignP')}). Two more seeds, trained after the recipe was frozen, score "
            f"{n('NumMtebSixteenSeedTask')} and {n('NumMtebSeventeenSeedTask')}. What v2-m3 adds is cross-lingual retrieval on the benchmark: "
            f"ar->en {n('NumVTwoMArEn')} against {n('NumBgeArEn')} for bge-m3 (paired difference {n('NumDeltaVTwoMVsBgeArEn')} points, interval "
            f"{n('NumDeltaVTwoMVsBgeArEnLo')} to {n('NumDeltaVTwoMVsBgeArEnHi')}).",
            f"* **It does not lead every slice.** multilingual-e5-large scores {n('NumMtebELargeCross')} on the MTEB(ara) Arabic-English subsets, "
            f"against {n('NumMtebVTwoMCross')} for v2-m3. v2-base and v2-small beat their off-the-shelf bases by {n('NumDeltaVTwoBVsEBaseArEn')} and "
            f"{n('NumDeltaVTwoSVsESmallArEn')} points on benchmark ar->en, but their MTEB(ara) gains ({n('NumPairedBMeanDiff')}, {n('NumPairedSMeanDiff')}) "
            f"have intervals that include zero; only v2-small over the first release ({n('NumPairedVMeanDiff')}, {n('NumPairedVCiLo')} to {n('NumPairedVCiHi')}) "
            f"excludes it. v2-small's mean over task types ({n('NumMtebVTwoSType')}) is below multilingual-e5-small's ({n('NumMtebESmallType')}).",
            f"* **The gain is paid for on Arabic-only tasks.** Mr.TyDi re-ranking changes by {n('NumGuardGapNamaa')} points for v2-m3 against bge-m3, "
            f"{n('NumGuardBGapNamaa')} for v2-base and {n('NumGuardSGapNamaa')} for v2-small against their bases; MIRACL retrieval by "
            f"{n('NumGuardGapMiracl')}, {n('NumGuardBGapMiracl')} and {n('NumGuardSGapMiracl')}. Behind the unchanged cross-encoder reranker of the first release, "
            f"v2-m3 and bge-m3 are level on ar->en ({n('NumRerankVTwoMArEn')} and {n('NumRerankBgeArEn')}).",
            f"* **A disclosed leak.** {n('NumMiraclExact')} of the {n('NumMiraclQueries')} Arabic dev queries of MIRACL-HardNegatives are exact copies of TyDi QA "
            f"training questions (data that bge-m3 also trained on), and {n('NumLeakRows')} kept training rows ({n('NumLeakDistinct')} distinct questions) equal "
            f"such a query. Dropping them moves every model's score by {n('NumLeakEffectMin')} to {n('NumLeakEffectMax')} points, so no comparison changes. "
            "The audit, the corrected clean / flagged split and all intervals are in the report.",
        ]
        return "\n".join(lines)


def sync_readme(block: str, path: Path = README) -> bool:
    """Replace the text between START and END in the README; False if the markers are absent."""
    text = path.read_text()
    if START not in text or END not in text:
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{block}\n{END}{tail}"
    if new != text:
        path.write_text(new)
    return True


def current_block(path: Path = README) -> str | None:
    text = path.read_text()
    if START not in text or END not in text:
        return None
    return text.split(START, 1)[1].split(END, 1)[0].strip("\n")
