"""Filesystem layout shared by every script in `arretrieval.retrieval`.

  runs/retrieval/data       passages / queries parquet, translations (git-ignored, rebuilt by the data stage)
  runs/retrieval/cache      embedding caches for the off-the-shelf models
  runs/retrieval/mined      hard-negative candidate files
  runs/retrieval/models     fine-tuned weights
  runs/retrieval/results    one json per evaluated system (+ stored top-100 rankings, ~850 MB; git-ignored)
  results/retrieval         the small curated snapshot that is committed (see `publish_results.py`)

`ARR_RUNS_DIR` moves the whole `runs/` tree; `ARR_RESULTS_DIR` points the *report* tools (make_report, chart) at a
different results directory (default: the committed snapshot, so `make report` works on a fresh clone).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import ROOT, RUNS_DIR

REPO = ROOT
OUT = RUNS_DIR / "retrieval"
DATA = OUT / "data"  # passages.parquet, queries.parquet, translations
CACHE = OUT / "cache"  # embedding caches (baseline models only)
MINED = OUT / "mined"  # hard-negative files
MODELS = OUT / "models"  # fine-tuned weights
RESULTS = OUT / "results"  # one json per evaluated system / experiment (written by the pipeline)
SNAPSHOT = ROOT / "results" / "retrieval"  # curated small copies that ARE committed
REPORT_RESULTS = Path(os.getenv("ARR_RESULTS_DIR", SNAPSHOT))  # where the report tools read from
CARD = RESULTS / "data_card"  # small derived artefacts of the data stage (manifest, leakage audit, id index)
REPORT_CARD = REPORT_RESULTS / "data_card"

SEED = 13


def ensure_dirs() -> None:
    """Create the (git-ignored) output directories."""
    for p in (DATA, CACHE, MINED, MODELS, RESULTS, CARD):
        p.mkdir(parents=True, exist_ok=True)


ensure_dirs()
