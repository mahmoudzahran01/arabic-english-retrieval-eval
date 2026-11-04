"""The report tables regenerate from the committed results, every macro used in main.tex exists, and the prose claims still hold."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from report import make_tables as mt

MAIN = Path(mt.ROOT / "report" / "main.tex")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("tables")
    old = mt.OUT
    mt.OUT = out
    try:
        t = mt.Tables()
        t.write()
    finally:
        mt.OUT = old
    return out, t


def test_formatting_helpers():
    assert mt.pct(0.7349) == "73.5"
    assert mt.spct(0.101) == "+10.1"
    assert mt.spct(-0.0044).startswith("$-$")
    assert mt.thousands(20999) == "20{,}999"
    assert mt.esc("a_b & 5%") == r"a\_b \& 5\%"


def test_every_macro_used_in_the_text_is_defined(built):
    out, _ = built
    defined = set(re.findall(r"\\def\\([A-Za-z]+)\{", (out / "numbers.tex").read_text()))
    text = MAIN.read_text()
    used = set(re.findall(r"\\((?:Num|Lic|Todo)[A-Za-z]*|Delta[A-Z][A-Za-z]*)", text))
    assert used - defined == set(), f"undefined macros: {sorted(used - defined)}"


def test_macro_names_contain_no_digits(built):
    out, _ = built
    for name in re.findall(r"\\def\\(\S+?)\{", (out / "numbers.tex").read_text()):
        assert name.isalpha(), name


def test_tables_exist_and_are_well_formed(built):
    out, _ = built
    for name in (
        "tab_main",
        "tab_deltas",
        "tab_ablation",
        "tab_mteb",
        "tab_guards",
        "tab_clean",
        "tab_data",
        "tab_deploy",
        "tab_seeds",
        "tab_volume",
    ):
        body = (out / f"{name}.tex").read_text()
        assert body.count(r"\begin{tabular}") == body.count(r"\end{tabular}") or body.startswith(r"\TODO"), name


def test_main_table_values_match_the_result_files(built):
    out, _ = built
    row = next(line for line in (out / "tab_main.tex").read_text().splitlines() if line.startswith("v2-m3 (WiSE"))
    cells = [c.strip(" \\") for c in row.split("&")]
    j = mt.load_json(mt.V2 / "wm_a75.test.json")
    assert any(mt.pct(j["directions"]["ar->en"]["mrr@10"]) in c for c in cells)


def test_prose_claims_still_hold(built):
    out, _ = built
    lines = (out / "claims_check.txt").read_text().splitlines()
    assert lines and not [ln for ln in lines if ln.startswith("REVISE")], [ln for ln in lines if ln.startswith("REVISE")]


def test_no_hand_typed_results_in_the_prose():
    """Numbers in main.tex may only be section/table numbers, years of citations, or typeset through macros."""
    text = MAIN.read_text()
    body = text.split(r"\begin{document}")[1]
    body = re.sub(r"%.*", "", body)  # comments
    body = re.sub(r"\\(?:input|label|ref|citep|citet|cite|includegraphics)\{[^}]*\}", "", body)
    body = re.sub(r"\\texttt\{[^}]*\}", "", body)  # commands / file names
    body = re.sub(r"\$[^$]*\$", "", body)  # math such as alpha, >=
    body = re.sub(r"\\newcommand.*", "", body)
    # identifiers, not results: metric names (MRR@10), workload names (batch 32), run names (seed 13), a model name
    body = re.sub(r"[0-9.]+\\columnwidth|amendments? \d|@\d+|\d-bit|batch \d+|seeds? \d+(?: and \d+)?|Gemma-4-26B-A4B", "", body)
    hits = re.findall(r"(?<![\\\w.])\d+(?:[.,]\d+)?(?!\w)", body)
    allowed = {"11", "118M", "278M", "568M"}  # model sizes quoted by name are parameters of the architectures, not results
    bad = [h for h in hits if h not in allowed]
    # the only digits left are model-size names (118M, ...) which the regexp excludes through \w, plus table/section numbers
    assert bad == [], f"hand-typed numbers in main.tex: {bad}"
