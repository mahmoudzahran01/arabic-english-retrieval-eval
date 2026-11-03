"""Filesystem layout shared by every script in `arretrieval.rageval`.

  runs/rageval                 all outputs (git-ignored): answers_*.jsonl, judge_*/, calibration_*.md, ...
  runs/rageval/data            corpus / questions / planted-error set (contain XQuAD text: never committed)
  runs/rageval/data_raw        the downloaded XQuAD json files
  runs/rageval/cache           LLM + embedding caches
  results/rageval             the curated snapshot that IS committed (dataset text removed)

`ARR_RAGEVAL_DIR` points every script at another output tree, e.g. `ARR_RAGEVAL_DIR=results/rageval` re-scores the
committed snapshot without any LLM call (see tests/rageval/test_committed_results.py).
"""

import os
from pathlib import Path

from ..config import ROOT, RUNS_DIR

HERE = Path(__file__).resolve().parent
RUNS = Path(os.getenv("ARR_RAGEVAL_DIR", RUNS_DIR / "rageval"))  # all outputs live here
DATA = RUNS / "data"
RAW = RUNS / "data_raw"
CACHE = RUNS / "cache"
SNAPSHOT = ROOT / "results" / "rageval"
for _p in (RUNS, DATA, CACHE):
    _p.mkdir(parents=True, exist_ok=True)
