"""Copy the small, curated part of a finished run into the committed snapshot `results/retrieval`.

Copies, from runs/retrieval/results:
  * every `<system>.{test,dev}.json` and `<system>.train.json` (metrics per slice / direction, training stats)
  * every `<system>.{test,dev}.perq.npz` (per-query metric arrays, ~2 MB in total; they feed the paired bootstrap)
  * latency.json, robustness.json, mining_*.json, fn_judge.jsonl, subset_150*.json
  * the `data_card/` directory (manifest, leakage audit, id-only test query index)
It never copies `*.runs.json` (stored top-100 rankings, ~850 MB) or any dataset text.

Run: python -m arretrieval.retrieval.publish_results
"""

from __future__ import annotations

import re
import shutil

from .paths import RESULTS, SNAPSHOT

KEEP = re.compile(r"(\.(test|dev)\.(json|perq\.npz)|\.train\.json)$")
EXTRA = ("latency.json", "robustness.json", "fn_judge.jsonl", "subset_150.json", "subset_150.scores.json")


def main() -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(RESULTS.iterdir()):
        if p.is_file() and (KEEP.search(p.name) or p.name in EXTRA or re.fullmatch(r"mining_.*\.json", p.name)):
            shutil.copy2(p, SNAPSHOT / p.name)
            n += 1
    card = RESULTS / "data_card"
    if card.is_dir():
        shutil.copytree(card, SNAPSHOT / "data_card", dirs_exist_ok=True)
    print(f"copied {n} files (+ data_card) into {SNAPSHOT}")


if __name__ == "__main__":
    main()
