#!/usr/bin/env bash
# Stage 1: build the benchmark from the public datasets, translate the cross-lingual train/dev queries, audit leakage.
#   needs: network (Hugging Face Hub), an OpenAI-compatible LLM for the translation step (GEMMA_* variables, see .env.example)
#   time:  build_data ~2 min on CPU; translation ~2 h for 5,000 questions at concurrency 2 on a shared server; the rest minutes
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.retrieval.build_data                  # -> passages.parquet, queries_mono.parquet, xling_candidates.jsonl
$PY -m arretrieval.retrieval.translate                   # -> xling_translations.jsonl (resumable)
$PY -m arretrieval.retrieval.finalize_data               # -> queries.parquet, results/data_card/{manifest.json,test_query_index.csv}
$PY -m arretrieval.retrieval.audit_leakage               # -> results/data_card/{leakage_audit.json,test_pids_near_train.json}
