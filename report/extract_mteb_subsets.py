"""Per-task Arabic-only and Arabic<->English subset scores from raw MTEB result JSONs (stdlib only, read-only).

    python3 report/extract_mteb_subsets.py ROOT [ROOT ...] > results/retrieval_v2/mteb_subsets.json

ROOT is an mteb result cache `.../results` (model folders `ORG__name/<revision>/<Task>.json`). The output maps each model folder name to
`{task: {"all": .., "cross": .., "mono": ..}}` on the 0-100 scale: the mean of the main score over all subset/split rows of the task, over its
Arabic<->English subsets (null when the task has none), and over the remaining subsets. The cross-lingual subset list is the one used for the
"Mean ar-en cross-lingual subsets" column of the benchmark leaderboard (mteb-arabic/scripts/make_leaderboard.py), so the numbers are comparable with it.
`mteb_paired.json` has no per-subset scores, so `report/make_tables.py` uses this file for the cross-lingual row of the paired MTEB(ara) table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CROSS = {
    "MLQARetrieval": {"ara-eng", "eng-ara"},
    "BelebeleRetrieval": {"arb_Arab-eng_Latn", "eng_Latn-arb_Arab"},
    "XPQARetrieval": {"ara-eng", "eng-ara"},
    "STS17": {"en-ar"},
    "FloresBitextMining": {"arb_Arab-eng_Latn", "eng_Latn-arb_Arab"},
    "Tatoeba": {"ara-eng"},
}
TASKS = [
    "MIRACLRetrievalHardNegatives.v2", "XQuADRetrieval", "MLQARetrieval", "BelebeleRetrieval", "XPQARetrieval", "WebFAQRetrieval", "NamaaMrTydiReranking",
    "STS17", "STS22.v2", "SemRel24STS", "XNLI", "ArEntail", "AJGT.v2", "TweetEmotionClassification.v2", "TweetSarcasmClassification.v2",
    "HotelReviewSentimentClassification.v2", "SIB200ClusteringS2S.v2", "FloresBitextMining", "Tatoeba",
]  # fmt: skip


def _mean(xs: list[float]) -> float | None:
    return 100 * sum(xs) / len(xs) if xs else None


def task_scores(path: Path, task: str) -> dict[str, float | None]:
    rows = [
        (e.get("hf_subset"), float(e["main_score"]))
        for entries in json.loads(path.read_text()).get("scores", {}).values()
        for e in entries
        if "main_score" in e
    ]
    cross = CROSS.get(task, set())
    return {
        "all": _mean([s for _, s in rows]),
        "cross": _mean([s for sub, s in rows if sub in cross]),
        "mono": _mean([s for sub, s in rows if sub not in cross]),
    }


def extract(roots: list[Path]) -> dict[str, dict[str, dict[str, float | None]]]:
    out: dict[str, dict] = {}
    for root in roots:
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for rev in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                found = {t: task_scores(rev / f"{t}.json", t) for t in TASKS if (rev / f"{t}.json").exists()}
                if found:
                    out[model_dir.name] = found
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    json.dump(extract([Path(a) for a in sys.argv[1:]]), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
