#!/usr/bin/env bash
# Stage 2: off-the-shelf baselines (BM25 x3 tokenizations, e5-small, MiniLM, bge-m3, +Arabic-normalized dense, RRF hybrids)
# and the language-gated hybrids (the typing-noise test runs in stage 5, once the fine-tuned model exists). Uses the GPU if there is one (CPU works, but bge-m3 over 31k texts is slow).
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.retrieval.eval_baselines --splits test
$PY -m arretrieval.retrieval.eval_baselines --splits dev --dense-tags e5-small --skip-arnorm --bm25-modes stem
$PY -m arretrieval.retrieval.eval_hybrid --dense e5-small --bm25 bm25-stem --gate-language
$PY -m arretrieval.retrieval.eval_hybrid --dense bge-m3 --bm25 bm25-stem --gate-language
