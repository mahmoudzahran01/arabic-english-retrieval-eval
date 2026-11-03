#!/usr/bin/env bash
# Stage 5: cross-encoder rerankers (fine-tune + evaluate), the gated hybrids built on the fine-tuned embedder, the 568M
# reference reranker on a 1,050-query subset, depth sweep, latency. Needs stage 4 (the kept final-recipe model) and a GPU.
# Idempotent: finished items are skipped.
source "$(dirname "$0")/../lib.sh"

MM=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
FT=e5s_hard1_xl_e1_s13

rer() {      # rer <name> <train_reranker args...>
  local name="$1"; shift
  retry "$RES/${name}.train.json" "$LOGS/${name}.log" $PY -m arretrieval.retrieval.train_reranker --name "$name" "$@"
}
rerank() {   # rerank <first stage> <reranker path|id> <name> [eval_rerank args...]
  local first="$1" model="$2" name="$3"; shift 3
  retry "$RES/$(fname "$first>>$name").test.json" "$LOGS/rerank_$(fname "$first>>$name").log" \
    $PY -m arretrieval.retrieval.eval_rerank --first-stage "$first" --reranker "$model" --name "$name" --splits test "$@"
}

# first stages that are not plain dense runs: language-gated RRF(dense, BM25-stem)
$PY -m arretrieval.retrieval.eval_hybrid --dense "$FT" --bm25 bm25-stem --gate-language
H_FT="hybrid-gated-$FT+bm25-stem"
H_BGE="hybrid-gated-bge-m3+bm25-stem"

# off-the-shelf reranker, then the fine-tuned one (mined negatives, + translated queries) and its ablations
rerank e5-small "$MM" rr_mmarco_off
rer rr_ft_bce_s13 --neg-model e5-small-xl --n-neg 4 --xling --seed 13
rerank e5-small "$MODELS/rr_ft_bce_s13" rr_ft_bce_s13
rer rr_ft_randneg_s13 --neg-model e5-small-xl --n-neg 4 --xling --pick random_corpus --seed 13
rerank e5-small "$MODELS/rr_ft_randneg_s13" rr_ft_randneg_s13
rer rr_ft_bce_noxl_s13 --neg-model e5-small-xl --n-neg 4 --seed 13
rerank e5-small "$MODELS/rr_ft_bce_noxl_s13" rr_ft_bce_noxl_s13
rerank e5-small "$MODELS/rr_ft_bce_s13" rr_ft_bce_s13_top10 --top-n 10
rerank e5-small "$MODELS/rr_ft_bce_s13" rr_ft_bce_s13_top20 --top-n 20

# other first stages behind the same rerankers
rerank "$FT" "$MM" rr_mmarco_off
rerank "$FT" "$MODELS/rr_ft_bce_s13" rr_ft_bce_s13
rerank "$H_BGE" "$MM" rr_mmarco_off
rerank "$H_BGE" "$MODELS/rr_ft_bce_s13" rr_ft_bce_s13
rerank "$H_FT" "$MODELS/rr_ft_bce_s13" rr_ft_bce_s13

# 568M reference reranker: ~15x the compute, so it scores a seeded 150-queries-per-slice subset (1,050 queries); the other
# systems are re-scored on exactly the same subset
rerank e5-small BAAI/bge-reranker-v2-m3 rr_bge_v2m3_off_sub150 --bs 32 --sample-per-slice 150
$PY -m arretrieval.retrieval.eval_subset --n 150 --systems e5-small "e5-small>>rr_mmarco_off" "e5-small>>rr_ft_bce_s13" \
  "e5-small>>rr_ft_randneg_s13" "e5-small>>rr_bge_v2m3_off_sub150"

# typing-noise robustness of the fine-tuned embedder, and latency (run with nothing else on the GPU)
$PY -m arretrieval.retrieval.eval_robustness --dense e5-small bge-m3 --model-dir "$FT"
$PY -m arretrieval.retrieval.latency --first-stage e5-small
