#!/usr/bin/env bash
# RAG-eval stage 5: export the 40-item human-labelling sheet, then (after you filled it in) compute human-vs-judge
# agreement; finally copy the curated, text-free outputs into results/rageval.
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.rageval.export_human                  # -> runs/rageval/human_labeling.csv (+ _key.json: do not open before labelling)
echo "Fill the human_* columns of runs/rageval/human_labeling.csv (see docs/rageval/HUMAN_LABELING.md), then:"
echo "  $PY -m arretrieval.rageval.agreement"
$PY -m arretrieval.rageval.publish_results
