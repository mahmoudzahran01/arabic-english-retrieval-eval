#!/usr/bin/env bash
# RAG-eval stage 4: the before/after experiment (A_baseline -> I_rerank_lang), its fresh-question confirmation, Ragas on the
# RAG answers, pairwise-judge bias checks, judge-vs-judge agreement on real answers.
source "$(dirname "$0")/../lib.sh"
export QWEN_SLOTS="${QWEN_SLOTS:-12}"

$PY -m arretrieval.rageval.rag --config I_rerank_lang
for c in A_baseline I_rerank_lang; do
  $PY -m arretrieval.rageval.calibrate --model qwen  --answers "$c" --judges correct_v1,cite_v1
  $PY -m arretrieval.rageval.calibrate --model gemma --answers "$c" --judges correct_v1
done
$PY -m arretrieval.rageval.ab_test --base A_baseline --others I_rerank_lang --judge-model gemma --faith-judge cite_v1 --faith-model qwen \
  --md "$RUNS/rageval/ab_test.md"
$PY -m arretrieval.rageval.ab_test --base A_baseline --others I_rerank_lang --judge-model qwen --faith-judge cite_v1 \
  --md "$RUNS/rageval/ab_test_qwenjudge.md"

# 48 fresh questions, never used to diagnose or tune anything
$PY -m arretrieval.rageval.heldout_ab --n 48
for c in heldout_A_baseline heldout_I_rerank_lang; do
  $PY -m arretrieval.rageval.calibrate --model qwen  --answers "$c" --judges correct_v1,cite_v1
  $PY -m arretrieval.rageval.calibrate --model gemma --answers "$c" --judges correct_v1
done
$PY -m arretrieval.rageval.ab_test --base heldout_A_baseline --others heldout_I_rerank_lang --judge-model gemma --faith-judge cite_v1 \
  --faith-model qwen --md "$RUNS/rageval/ab_test_heldout.md"

# Gemma vs Qwen on real answers (no planted labels): a 24-answer Gemma cross-check per config, then the agreement report
for c in A_baseline I_rerank_lang; do $PY -m arretrieval.rageval.calibrate --model gemma --answers "$c" --judges correct_v1 --sample 24; done
$PY -m arretrieval.rageval.real_agreement --configs A_baseline,I_rerank_lang --judge correct_v1

# Ragas on a stratified 20-question subset (+ 6 targeted retrieval misses) of the baseline answers
$PY -m arretrieval.rageval.ragas_eval rag --config A_baseline --per-cell 4 --n-unans 4 --add-misses 6
$PY -m arretrieval.rageval.ragas_report --config A_baseline --md "$RUNS/rageval/ragas_A.md"

# pairwise-judge biases: position, verbosity, self-preference
$PY -m arretrieval.rageval.bias_checks --model gemma --n 8
$PY -m arretrieval.rageval.bias_checks --model qwen --n 24
$PY -m arretrieval.rageval.bias_report > "$RUNS/rageval/bias_table.md"
