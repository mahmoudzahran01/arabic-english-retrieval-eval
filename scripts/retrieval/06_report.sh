#!/usr/bin/env bash
# Stage 6: publish the small result files, regenerate every table, RESULTS.md and the chart. CPU only.
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.retrieval.publish_results                 # runs/retrieval/results -> results/retrieval (no rankings, no text)
$PY -m arretrieval.retrieval.make_report                     # results/retrieval/tables.md
$PY -m arretrieval.retrieval.build_results                   # docs/retrieval/RESULTS.md
$PY scripts/make_chart.py                                    # docs/img/mrr_by_system.png
