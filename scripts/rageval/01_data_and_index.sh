#!/usr/bin/env bash
# RAG-eval stage 1: XQuAD -> 220-passage corpus + 180 eval questions + 932-question pool, index, retrieval-only metrics.
#   needs: network (GitHub raw, Hugging Face Hub for the two small models); CPU is enough
source "$(dirname "$0")/../lib.sh"

$PY -m arretrieval.rageval.download_data                 # xquad.{ar,en}.json, checksum-verified
$PY -m arretrieval.rageval.build_data                    # -> runs/rageval/data/{corpus,eval_questions,all_qas,heldout}.jsonl
$PY -m arretrieval.rageval.ingest                        # whole-passage index
$PY -m arretrieval.rageval.ingest --chunks               # sentence-window chunks (the "chunking" hypothesis)
$PY -m arretrieval.rageval.retrieval_eval                # hit@k / MRR per config -> runs/rageval/retrieval_eval.json
