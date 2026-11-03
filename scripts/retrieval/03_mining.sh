#!/usr/bin/env bash
# Stage 3: hard-negative mining and the LLM false-negative audit.
#   *-xl files also mine the translated cross-lingual train queries; the plain files mine the monolingual queries only
#   (the false-negative audit and the monolingual ablations use those).
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.retrieval.mine_cli --model e5-small --mono-only
$PY -m arretrieval.retrieval.mine_cli --model bge-m3 --mono-only
$PY -m arretrieval.retrieval.mine_cli --model e5-small --tag e5-small-xl
$PY -m arretrieval.retrieval.mine_cli --model bge-m3 --tag bge-m3-xl
$PY -m arretrieval.retrieval.judge_false_negatives --model e5-small      # LLM judge, concurrency 2 (GEMMA_* variables)
$PY -m arretrieval.retrieval.analyze_mining --model e5-small
