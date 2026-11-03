#!/usr/bin/env bash
# RAG-eval stage 3: build the planted-error set (labels true by construction), run every judge variant on it, score them.
#   needs: GEMMA_* (generator + main judge) and QWEN_* (second judge) endpoints
#   Gemma was slow and shared (3-5 calls/min), so its chain-of-thought judges run on the TEST split only, as published;
#   the cheaper Qwen judge runs everything.
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.rageval.plant_bases                   # verified-correct base answers (to NEW XQuAD questions)
$PY -m arretrieval.rageval.plant_items                   # -> runs/rageval/data/meta_items.jsonl (+ meta_card.json)

$PY -m arretrieval.rageval.calibrate --model gemma --judges correct_v1,cite_v1,faith_v3 --split test
$PY -m arretrieval.rageval.calibrate --model gemma --judges overall_pipeline_v3 --split test
$PY -m arretrieval.rageval.calibrate --model gemma --judges faith_v0,correct_v0,overall_v0
QWEN_SLOTS="${QWEN_SLOTS:-12}" $PY -m arretrieval.rageval.calibrate --model qwen --judges all
$PY -m arretrieval.rageval.ragas_eval meta --llm qwen --max-per-type 8        # Ragas on the planted set (Qwen evaluator)

for split in test dev all; do
  $PY -m arretrieval.rageval.analyze_judges --split "$split" --md "$RUNS/rageval/calibration_$split.md"
done
