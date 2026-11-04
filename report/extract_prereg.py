"""Extract the timestamped pre-registration trail from the lab notebook (runs/retrieval/v2/PROGRESS.md on the Spark; a read-only copy is passed in).

    python report/extract_prereg.py PROGRESS.md results/retrieval_v2/data/prereg_trail.json

Entries (verbatim text, no paraphrase; the report prints them in the appendix):
  alpha_rule   the sentence of the ~11:05 results entry that fixes the WiSE alpha selection rule before any guard evaluation
  rule_1       the ~14:00 section with the pre-registered replay / preservation selection rule
  amendment_1  the ~19:05 section (validation-suite selection) and its ~19:20 correction
  amendment_2  the ~21:35 section (alpha extension and LoRA rerun)
An entry that cannot be found is written with `"missing": true`, so the report shows a TODO instead of silently dropping it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TIME = re.compile(r"(\d{4}-\d{2}-\d{2})[^\d]*~?\s*(\d{2}:\d{2})")


def _section(lines: list[str], heading_re: str, stop_res: list[str]) -> list[str] | None:
    """Lines from the heading matching `heading_re` up to (excluding) the first later line matching one of `stop_res` or the next '## ' heading."""
    for i, line in enumerate(lines):
        if re.match(heading_re, line):
            out = [line]
            for nxt in lines[i + 1 :]:
                if nxt.startswith("## ") or any(re.match(r, nxt) for r in stop_res):
                    break
                out.append(nxt)
            return out
    return None


def _alpha_rule(lines: list[str]) -> tuple[str, str] | None:
    """The sentence that fixes the alpha rule, with the timestamp of the closest preceding '~HH:MM' entry."""
    for idx, line in enumerate(lines):
        m = re.search(r"(Selection rule fixed BEFORE any guard eval.*?alpha [0-9.]*[0-9]\.)", line)
        if not m:
            continue
        for back in reversed(lines[: idx + 1]):
            t = re.search(r"(\d{4}-\d{2}-\d{2}) ~(\d{2}:\d{2})", back)
            if t:
                return f"{t.group(1)} {t.group(2)}", m.group(1)
        return None
    return None


def extract(text: str) -> list[dict]:
    lines = text.splitlines()
    out: list[dict] = []
    a = _alpha_rule(lines)
    out.append(
        {
            "id": "alpha_rule",
            "title": "WiSE alpha selection rule, as recorded in the results entry",
            "timestamp": a[0] if a else None,
            "text": a[1] if a else "",
            "missing": a is None,
        }
    )
    specs = [
        (
            "rule_1",
            r"## \d{4}-\d{2}-\d{2} ~\d{2}:\d{2} coordinator: fix the MIRACL forgetting",
            [r"- 13:53 "],
            "Original pre-registered rule",
        ),
        (
            "amendment_1",
            r"## \d{4}-\d{2}-\d{2} ~\d{2}:\d{2} - PRE-REGISTRATION: validation suite",
            [r"- 18:36 RE-PRIORITISATION"],
            "Amendment 1: validation-suite selection",
        ),
        (
            "amendment_2",
            r"## \d{4}-\d{2}-\d{2} ~\d{2}:\d{2} - AMENDMENT 2",
            [r"- 21:23 drive15"],
            "Amendment 2: alpha extension and LoRA rerun",
        ),
    ]
    for ident, head, stops, title in specs:
        sec = _section(lines, head, stops)
        if sec is None:
            out.append({"id": ident, "title": title, "timestamp": None, "text": "", "missing": True})
            continue
        m = TIME.search(sec[0])
        out.append(
            {
                "id": ident,
                "title": title,
                "timestamp": f"{m.group(1)} {m.group(2)}" if m else None,
                "text": "\n".join(sec).strip(),
                "missing": False,
            }
        )
    return out


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    entries = extract(src.read_text())
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"source": "runs/retrieval/v2/PROGRESS.md (lab notebook)", "entries": entries}, indent=1, ensure_ascii=False))
    print("; ".join(f"{e['id']}: {'MISSING' if e['missing'] else e['timestamp']}" for e in entries))


if __name__ == "__main__":
    main()
