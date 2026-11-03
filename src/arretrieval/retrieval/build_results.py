"""Fill the {{T:<table title>}} placeholders of docs/retrieval/RESULTS.template.md from <results dir>/tables.md -> docs/retrieval/RESULTS.md.

Keeps the prose (hand-written) and the numbers (generated) separate so tables are never transcribed by hand.
Run: python -m arretrieval.retrieval.build_results      (pure python, no dependencies)
"""

from __future__ import annotations

import re
import sys

from ..config import ROOT
from .paths import REPORT_RESULTS

DOCS = ROOT / "docs" / "retrieval"


def sections(md: str) -> dict[str, str]:
    """Split the generated tables file into {title: body} by its `### ` headings."""
    out = {}
    for part in md.split("\n### ")[1:]:
        title, _, body = part.partition("\n")
        out[title.strip()] = body.strip("\n")
    return out


def render(template: str, tables_md: str) -> tuple[str, list[str]]:
    """Substitute every {{T:<title prefix>}} placeholder; returns (markdown, titles that matched no table)."""
    tables = sections(tables_md)
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        key = m.group(1).strip()
        hits = [t for t in tables if t.startswith(key)]
        if not hits:
            missing.append(key)
            return f"**[MISSING TABLE: {key}]**"
        return tables[hits[0]]

    return re.sub(r"\{\{T:(.+?)\}\}", sub, template), missing


def main() -> None:
    out, missing = render((DOCS / "RESULTS.template.md").read_text(), (REPORT_RESULTS / "tables.md").read_text())
    (DOCS / "RESULTS.md").write_text(out)
    print(f"wrote RESULTS.md ({len(out)} chars); missing tables: {missing}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
