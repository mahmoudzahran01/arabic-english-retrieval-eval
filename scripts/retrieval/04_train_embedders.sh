#!/usr/bin/env bash
# Stage 4: every embedder run behind the tables in docs/retrieval/RESULTS.md (fine-tune multilingual-e5-small, then
# evaluate on dev + test). Needs a CUDA GPU; ~35 runs of 1-30 min each on one NVIDIA GB10. Idempotent: finished runs are skipped.
source "$(dirname "$0")/../lib.sh"

emb() {   # emb <name> <train_embedder args...>
  local name="$1"; shift
  retry "$RES/${name}.train.json" "$LOGS/${name}.log" $PY -m arretrieval.retrieval.train_embedder --name "$name" "$@"
}
HARD="--neg-model e5-small --n-neg 1 --margin 0.02"          # one mined hard negative, 0.02 cosine false-negative margin

# (a) in-batch only vs +1 mined hard negative: 1 epoch, 3 seeds; +3 negatives (plain MNRL peaks at ~16 GB)
for s in 13 14 15; do emb "e5s_inb_e1_s$s" --neg-model none --epochs 1 --seed $s; done
for s in 13 14 15; do emb "e5s_hard1_e1_s$s" $HARD --epochs 1 --seed $s; done
emb e5s_hard3_e1_s13 --neg-model e5-small --n-neg 3 --margin 0.02 --epochs 1 --seed 13
# (c) epochs 1 vs 3, with a dev curve after every epoch
emb e5s_inb_e3_s13   --neg-model none --epochs 3 --seed 13 --dev-curve
emb e5s_hard1_e3_s13 $HARD --epochs 3 --seed 13 --dev-curve
# (b) data size: 25% / 50% of the training queries, and 25% for 4 epochs (same number of steps as 100% x 1 epoch)
emb e5s_inb_f25_e1_s13   --neg-model none --frac 0.25 --epochs 1 --seed 13
emb e5s_hard1_f25_e1_s13 $HARD --frac 0.25 --epochs 1 --seed 13
emb e5s_inb_f25_e4_s13   --neg-model none --frac 0.25 --epochs 4 --seed 13
emb e5s_hard1_f25_e4_s13 $HARD --frac 0.25 --epochs 4 --seed 13
emb e5s_inb_f50_e1_s13   --neg-model none --frac 0.5 --epochs 1 --seed 13
emb e5s_hard1_f50_e1_s13 $HARD --frac 0.5 --epochs 1 --seed 13
# (d) mining policy: false-negative handling and who mines
emb e5s_hard1_nomargin_e1_s13 --neg-model e5-small --n-neg 1 --no-margin --epochs 1 --seed 13
emb e5s_hard1_m05_e1_s13      --neg-model e5-small --n-neg 1 --margin 0.05 --epochs 1 --seed 13
emb e5s_hard1_notitle_e1_s13  --neg-model e5-small --n-neg 1 --margin 0.02 --same-title --epochs 1 --seed 13
emb e5s_hard1_rank10_e1_s13   --neg-model e5-small --n-neg 1 --margin 0.02 --rank-lo 10 --epochs 1 --seed 13
emb e5s_randneg1_e1_s13       --neg-model e5-small --n-neg 1 --pick random_corpus --epochs 1 --seed 13
emb e5s_hard1_bgem3_e1_s13    --neg-model bge-m3 --n-neg 1 --margin 0.02 --epochs 1 --seed 13
# (e) loss temperature / learning rate / batch size (batch >= 128 uses CachedMNRL: same gradient, 5.4 GB instead of OOM)
emb e5s_hard1_sc50_e1_s13     $HARD --scale 50 --epochs 1 --seed 13
emb e5s_hard1_lr1e5_e1_s13    $HARD --lr 1e-5 --epochs 1 --seed 13
emb e5s_hard1_lr6e5_e1_s13    $HARD --lr 6e-5 --epochs 1 --seed 13
emb e5s_hard1_bs32_e1_s13     $HARD --bs 32 --epochs 1 --seed 13
emb e5s_hard1_bs128c_e1_s13   $HARD --bs 128 --loss cached --mini-batch 64 --epochs 1 --seed 13
emb e5s_hard1_cached512_e1_s13 $HARD --bs 512 --loss cached --mini-batch 64 --epochs 1 --seed 13

# cross-lingual: + the translated queries (negatives mined over mono + translated train queries: *-xl files)
XL="--neg-model e5-small-xl --n-neg 1 --margin 0.02 --xling"
emb e5s_inb_xl_e1_s13 --neg-model none --xling --epochs 1 --seed 13
for s in 13 14 15; do emb "e5s_hard1_xl_e1_s$s" $XL --epochs 1 --seed $s --keep-model; done      # FINAL RECIPE, 3 seeds
emb e5s_hard3c_xl_e1_s13      --neg-model e5-small-xl --n-neg 3 --margin 0.02 --xling --epochs 1 --seed 13 --loss cached --mini-batch 64
emb e5s_hard1_xl_e2_s13       $XL --epochs 2 --seed 13
emb e5s_hard1_xl_bgem3_e1_s13 --neg-model bge-m3-xl --n-neg 1 --margin 0.02 --xling --epochs 1 --seed 13
