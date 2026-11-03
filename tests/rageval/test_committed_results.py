"""The committed snapshot re-scores to exactly the committed reports, with no LLM and no dataset text involved."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from arretrieval.rageval.publish_results import slim_answer, slim_item

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "results" / "rageval"


def _run(module: str, args: list[str], run_dir: Path) -> None:
    env = {**os.environ, "ARR_RAGEVAL_DIR": str(run_dir)}
    subprocess.run([sys.executable, "-m", f"arretrieval.rageval.{module}", *args], env=env, check=True, capture_output=True, cwd=run_dir)


@pytest.fixture()
def scratch(tmp_path):
    shutil.copytree(SNAP, tmp_path / "rageval")
    return tmp_path / "rageval"


@pytest.mark.parametrize("split", ["test", "dev", "all"])
def test_judge_calibration_tables_reproduce(scratch, tmp_path, split):
    out = tmp_path / f"calibration_{split}.md"
    _run("analyze_judges", ["--split", split, "--md", str(out)], scratch)
    assert out.read_text() == (SNAP / f"calibration_{split}.md").read_text()


def test_ab_test_reproduces(scratch, tmp_path):
    out = tmp_path / "ab.md"
    _run(
        "ab_test",
        ["--base", "A_baseline", "--others", "I_rerank_lang", "--judge-model", "qwen", "--faith-judge", "cite_v1", "--md", str(out)],
        scratch,
    )
    assert out.read_text() == (SNAP / "ab_test_qwenjudge.md").read_text()


def test_headline_acceptable_answer_rate(scratch):
    rows = {c: [json.loads(line) for line in open(SNAP / f"answers_{c}.jsonl")] for c in ("A_baseline", "I_rerank_lang")}
    assert all(len(r) == 180 for r in rows.values())
    base = rows["A_baseline"]
    answerable = [r for r in base if r["answerable"]]
    assert sum(bool(r["gold_retrieved"]) for r in answerable) == 130  # 81.3% gold in the top 3
    assert sum(r["abstained"] for r in base if not r["answerable"]) == 20  # 20 / 20 correct refusals
    wrong_lang = [r for r in base if not r["abstained"] and r["answer_lang"] != r["q_lang"]]
    assert len(wrong_lang) == 34 and all(r["q_lang"] == "en" for r in wrong_lang)  # the defect no Ragas metric flags


def test_published_snapshot_contains_no_passage_text():
    for name in ("answers_A_baseline.jsonl", "answers_heldout_I_rerank_lang.jsonl"):
        for line in open(SNAP / name):
            assert all("text" not in p for p in json.loads(line)["retrieved"])
    for line in open(SNAP / "data" / "meta_items.jsonl"):
        item = json.loads(line)
        assert "ref_sentence" not in item and all("text" not in c for c in item["contexts"])


def test_slimming_functions_keep_ids_and_scores():
    row = {"qid": "Q1", "retrieved": [{"id": "P1", "parent": "P1", "lang": "en", "text": "secret", "score": 0.5, "rank": 1}]}
    assert slim_answer(row)["retrieved"] == [{"id": "P1", "parent": "P1", "lang": "en", "score": 0.5, "rank": 1}]
    item = {"id": "M1", "ref_sentence": "x", "contexts": [{"id": "P1", "lang": "ar", "text": "t"}], "labels": {"faithful": 1}}
    slim = slim_item(item)
    assert "ref_sentence" not in slim and slim["contexts"] == [{"id": "P1", "lang": "ar"}] and slim["labels"] == {"faithful": 1}
