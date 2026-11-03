"""The committed result snapshot must regenerate the committed tables, and the headline table must stay consistent."""

import json
import re
import sys
from pathlib import Path

import pytest

from arretrieval.retrieval import build_results, make_report, report_lib
from arretrieval.retrieval.paths import SNAPSHOT

ROOT = Path(__file__).resolve().parents[2]


def test_cell_formatting():
    assert report_lib.cell(0.7234, 0.0, 1) == "72.3"
    assert report_lib.cell(0.7234, 0.0049, 3) == "72.3 ± 0.5"
    assert report_lib.cell(float("nan"), float("nan"), 0) == "n/a"


def test_aggregate_over_seeds_uses_the_sample_std():
    mean, std, n = report_lib.agg(["e5s_hard1_xl_e1_s13", "e5s_hard1_xl_e1_s14", "e5s_hard1_xl_e1_s15"], "test", "macro", "mrr@10")
    assert n == 3 and 0.75 < mean < 0.76 and 0 < std < 0.005


def test_boot_delta_is_deterministic_and_signed():
    a = report_lib.boot_direction_delta("e5-small", "e5s_hard1_xl_e1_s13", "test", "ar->en")
    b = report_lib.boot_direction_delta("e5-small", "e5s_hard1_xl_e1_s13", "test", "ar->en")
    assert a == b and a["ci_lo"] > 0 and a["delta"] > 0.10


@pytest.mark.skipif(not (SNAPSHOT / "tables.md").exists(), reason="snapshot not generated yet")
def test_tables_regenerate_byte_for_byte(tmp_path, monkeypatch):
    out = tmp_path / "tables.md"
    monkeypatch.setattr(sys, "argv", ["make_report", "--out", str(out)])
    make_report.main()
    assert out.read_text() == (SNAPSHOT / "tables.md").read_text()


@pytest.mark.skipif(not (ROOT / "docs/retrieval/RESULTS.md").exists(), reason="docs not generated yet")
def test_results_md_is_up_to_date_with_the_tables():
    rendered, missing = build_results.render(
        (ROOT / "docs/retrieval/RESULTS.template.md").read_text(), (SNAPSHOT / "tables.md").read_text()
    )
    assert missing == []
    assert rendered == (ROOT / "docs/retrieval/RESULTS.md").read_text()


def test_headline_numbers_match_the_stored_results():
    """The numbers quoted in the README headline: re-derived from the per-system result files."""
    load = report_lib.load
    val = lambda name, d: load(name, "test")["directions"][d]["mrr@10"] * 100  # noqa: E731
    assert round(val("e5-small", "macro"), 1) == 66.9
    assert round(val("bge-m3", "macro"), 1) == 74.8
    assert round(val("e5-small__rr_ft_bce_s13", "macro"), 1) == 82.7
    assert round(val("e5-small__rr_ft_bce_s13", "ar->en"), 1) == 68.9
    assert round(val("e5s_hard1_xl_e1_s13__rr_ft_bce_s13", "macro"), 1) == 83.3
    assert round(val("e5s_hard1_xl_e1_s13__rr_ft_bce_s13", "ar->en"), 1) == 72.3
    assert round(val("e5-small", "ar->en"), 1) == 41.4


def test_manifest_matches_the_documented_corpus_size():
    manifest = json.loads((SNAPSHOT / "data_card" / "manifest.json").read_text())
    assert manifest["passages_total"] == 24506
    assert sum(manifest["queries_by_split"].values()) == 31404
    audit = json.loads((SNAPSHOT / "data_card" / "leakage_audit.json").read_text())
    assert audit["gold_outside_pool"] == 0 and audit["duplicate_pids_across_pools"] == 0
    assert re.fullmatch(r"[0-9a-f]{12}", "ar-83f48010c5fd".split("-")[1])


def _headline_table(markdown: str) -> str:
    """The first `| System | ar->ar | ...` table of a document (the headline table in both README.md and RESULTS.md)."""
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| System | ar->ar"))
    end = start
    while lines[end].startswith("|"):
        end += 1
    return "\n".join(lines[start:end])


def test_readme_headline_table_is_copied_verbatim_from_results_md():
    readme = (ROOT / "README.md").read_text()
    results = (ROOT / "docs/retrieval/RESULTS.md").read_text()
    table = _headline_table(readme)
    assert table.count("\n") == 10 and table == _headline_table(results)


def test_chart_plots_exactly_the_headline_numbers():
    import importlib.util

    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("make_chart", ROOT / "scripts" / "make_chart.py")
    chart = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chart)
    vals = chart.values().round(1)
    assert vals[:, 4].tolist() == [66.9, 74.8, 75.7, 83.3, 85.0]  # macro column = headline table
    assert vals[:, 2].tolist() == [41.4, 63.3, 54.7, 72.3, 75.3]  # ar->en column
