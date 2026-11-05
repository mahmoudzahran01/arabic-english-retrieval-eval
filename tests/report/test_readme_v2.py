"""The v2 block of README.md is generated from the report's macros (report/readme_tables.py); it must be current and must not hide the caveats."""

from __future__ import annotations

import re

import pytest

from report import make_tables as mt
from report import readme_tables as rt


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    old = mt.OUT
    mt.OUT = tmp_path_factory.mktemp("tables")
    try:
        t = mt.Tables()
        t.write()
    finally:
        mt.OUT = old
    return t


def test_readme_block_is_current(tables):
    assert rt.current_block() == tables.readme_v2(), "run `make -C report tables` to refresh the v2 block of README.md"


def test_block_carries_the_honest_caveats(tables):
    block = tables.readme_v2()
    for phrase in (
        "statistical tie",
        "does not lead every slice",
        "paid for on Arabic-only tasks",
        "disclosed leak",
        "intervals that include zero",
    ):
        assert phrase in block
    assert "TODO" not in block and "\\" not in block and "$" not in block  # plain Markdown, no TeX left over
    assert re.search(r"\| \*\*v2-m3\*\* \| 568M \| [\d.]+ \| [\d.]+ \| 72\.\d\d \| \+0\.\d\d \[-0\.\d\d, \+0\.\d\d\] vs bge-m3 \|", block)


def test_release_decisions_are_filled_in():
    text = rt.README.read_text()
    assert "[TODO" not in text
    for repo in ("arabic-english-bge-m3-retrieval", "arabic-english-e5-base-retrieval", "arabic-english-e5-small-retrieval-v2"):
        assert f"https://huggingface.co/mzahran1/{repo}" in text, repo
    assert "**Licence of the weights:** MIT" in text


def test_sync_replaces_only_the_marked_block(tmp_path):
    p = tmp_path / "R.md"
    p.write_text(f"head\n{rt.START}\nold\n{rt.END}\ntail\n")
    assert rt.sync_readme("new", p) and p.read_text() == f"head\n{rt.START}\nnew\n{rt.END}\ntail\n"
    p.write_text("no markers\n")
    assert not rt.sync_readme("new", p) and p.read_text() == "no markers\n"
