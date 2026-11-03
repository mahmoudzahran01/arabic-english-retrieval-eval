"""Copy the curated, text-free part of a finished RAG-eval run into the committed snapshot `results/rageval`.

Copied as-is: calibration_*.{json,md}, ab_test*.md, bias_*.json / bias_table.md, ragas_*.{md,jsonl}, retrieval_eval.json,
the judge verdict directories (`judge_*`), data_card.json and meta_card.json.
Copied with the passage text removed: answers_*.jsonl (retrieved passages keep id / parent / lang / score / rank) and
data/meta_items.jsonl (the planted-error set: `contexts` keep id / lang, `ref_sentence` is dropped).
Never copied: the corpus, the question pools, the human-labelling sheet (they contain XQuAD paragraphs).

Run: python -m arretrieval.rageval.publish_results
"""

from __future__ import annotations

import json
import shutil

from .paths import DATA, RUNS, SNAPSHOT

PLAIN_FILES = (
    "calibration_all.json",
    "calibration_all.md",
    "calibration_dev.json",
    "calibration_dev.md",
    "calibration_test.json",
    "calibration_test.md",
    "ab_test.md",
    "ab_test_heldout.md",
    "ab_test_qwenjudge.md",
    "bias_gemma.json",
    "bias_qwen.json",
    "bias_table.md",
    "ragas_A.md",
    "ragas_A_baseline.jsonl",
    "ragas_meta_qwen.jsonl",
    "retrieval_eval.json",
)
TEXT_FIELDS_DROPPED = ("text",)


def _rewrite_jsonl(src, dst, fn) -> int:
    n = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if line.strip():
                out.write(json.dumps(fn(json.loads(line)), ensure_ascii=False) + "\n")
                n += 1
    return n


def _strip_passages(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in p.items() if k not in TEXT_FIELDS_DROPPED} for p in rows]


def slim_answer(row: dict) -> dict:
    return {**row, "retrieved": _strip_passages(row["retrieved"])}


def slim_item(item: dict) -> dict:
    out = {k: v for k, v in item.items() if k != "ref_sentence"}
    out["contexts"] = _strip_passages(item["contexts"])
    return out


def main() -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT / "data").mkdir(exist_ok=True)
    for name in PLAIN_FILES:
        if (RUNS / name).exists():
            shutil.copy2(RUNS / name, SNAPSHOT / name)
    for d in sorted(RUNS.glob("judge_*")):
        if d.is_dir():
            shutil.copytree(d, SNAPSHOT / d.name, dirs_exist_ok=True)
    for p in sorted(RUNS.glob("answers_*.jsonl")):
        _rewrite_jsonl(p, SNAPSHOT / p.name, slim_answer)
    _rewrite_jsonl(DATA / "meta_items.jsonl", SNAPSHOT / "data" / "meta_items.jsonl", slim_item)
    for name in ("data_card.json", "meta_card.json"):
        shutil.copy2(DATA / name, SNAPSHOT / "data" / name)
    print(f"snapshot written to {SNAPSHOT}")


if __name__ == "__main__":
    main()
