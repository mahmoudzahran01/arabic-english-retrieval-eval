#!/usr/bin/env bash
# Import the final v2 result snapshot from a checkout of the v2-embedder branch (read-only: nothing in that checkout is touched or merged).
#   bash report/import_snapshot.sh ../arabic-english-retrieval-eval-v2embedder
# Copies results/retrieval_v2/ (test/dev json + per-query arrays, alpha_curve / confirmatory / leak_exact / mteb_paired / mteb_board / suite_pick json,
# guard_split with the corrected labels, raw MTEB results of the v2 models, the lab notebook PROGRESS.md) and docs/retrieval/RESULTS_v2.md over the snapshot,
# keeps the report-only files (data/protocol.json, data/licences.json, models/*, README.md), and rebuilds the pre-registration trail from the notebook.
# `report/pull_spark_extras.sh` still pulls what the branch does not carry (per-subset MTEB scores of the baselines, training arguments of the final models).
set -euo pipefail
SRC="${1:?path to the v2-embedder checkout}"
cd "$(dirname "$0")/.."
DST=results/retrieval_v2
test -f "$SRC/results/retrieval_v2/alpha_curve.json" || { echo "no v2 snapshot at $SRC/results/retrieval_v2" >&2; exit 1; }
mkdir -p "$DST/data"
rsync -a --exclude 'data/' --exclude 'models/' --exclude 'README.md' --exclude 'stale_*' "$SRC/results/retrieval_v2/" "$DST/"
cp "$SRC/results/retrieval_v2/audit_ood.json" "$SRC/results/retrieval_v2/audit_ood_filtered.json" "$SRC/results/retrieval_v2/diag_xquad_style.json" "$DST/data/"
cp "$SRC/results/retrieval_v2/data_card/manifest_v2.json" "$DST/data/manifest_v2.json"
cp "$SRC/docs/retrieval/RESULTS_v2.md" "$DST/RESULTS_v2.md"
python3 report/extract_prereg.py "$DST/PROGRESS.md" "$DST/data/prereg_trail.json"
