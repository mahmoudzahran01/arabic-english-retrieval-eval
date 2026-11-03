#!/usr/bin/env bash
# RAG-eval stage 2: the baseline RAG system (config A: dense top-3 + Gemma, prompt v1) and the hand-checked Ragas toy cases.
#   needs: GEMMA_* endpoint (generator); the toy Ragas check also needs the `rageval` extra (ragas) and bge-m3 (~2 GB)
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.rageval.rag --config A_baseline       # -> runs/rageval/answers_A_baseline.jsonl
$PY -m arretrieval.rageval.toy_ragas
