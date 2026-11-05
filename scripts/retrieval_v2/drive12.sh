#!/usr/bin/env bash
# Stage 12: (1) contamination-split diagnostic REDONE with passage titles (mteb embeds "title text"; the first version dropped the titles and under-scored every model),
# (2) the selected recipe (m3 + WiSE a.75): seed 15, WiSE blends of seeds 14/15, TEST + mixed of the blends, (3) guards of raw / a.50 for the alpha curve.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --test --keep-model"
M3="--base bge-m3 --data v2 --negs same:1 --epochs 1 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive12.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive12.log; }
while ! grep -q DRIVE11_DONE $L/queues.done 2>/dev/null; do sleep 30; done
# --- coordinator priority 1: score the selected v2-m3 (WiSE a.75, seed 13) on all MTEB(ara) tasks BEFORE anything else (same resource rules: behind the gate, fp16, batch 32, max_seq 512)
if [ ! -d $HOME/projects/qubit-ai-lab/runs/retrieval/v2/models/final_m3_wa75_s13 ]; then cp -r $HOME/projects/qubit-ai-lab/runs/retrieval/v2/models/wm_a75 $HOME/projects/qubit-ai-lab/runs/retrieval/v2/models/final_m3_wa75_s13; fi
echo "- $(date +%H:%M) FINAL CANDIDATE saved: runs/retrieval/v2/models/final_m3_wa75_s13 (v2-m3 = bge-m3 fine-tuned on the full filtered v2 data, seed 13, WiSE alpha 0.75 against bge-m3). Scoring on MTEB(ara) now." >> runs/retrieval/v2/PROGRESS.md
step sc1_m3 bash -c 'cd $HOME/projects/mteb-arabic && EXTRA_DOCKER_ARGS="-v $HOME/projects/qubit-ai-lab/runs/retrieval/v2/models:/v2models:ro -v $HOME/projects/qubit-ai-lab/runs/retrieval/v2/scripts:/myv2:ro" RESULT_CACHE=/work/results_cache_v2 ./run.sh python /myv2/score_local.py /v2models/final_m3_wa75_s13 final_m3_wa75_s13 32 BAAI/bge-m3 >> logs/eval_v2_final_m3_wa75_s13.log 2>&1'
# cheap student-recipe ablation right after the scoring (decides the final e5-base / e5-small recipe)
step k4_b_kdonly_e2 $D -m lab.retrieval.v2.train_v2 --name f5_b_kdonly_e2_s13 --base e5-base --negs same:1 --seed 13 --data v2 --epochs 2 --kd 1 --kd-only --t-scale 20 --teacher m3b --exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model
# --- coordinator priority 2: parameter-limited anti-forgetting runs (same data and rule): (a) LoRA r=32 lr 1e-4, (b) frozen embeddings + bottom 12 layers
LORA="--base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
step lo_lora $D -m lab.retrieval.v2.train_v2 --name m3_lo_s13 $LORA --lr 1e-4 --lora 32 --exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model
step lo_freeze $D -m lab.retrieval.v2.train_v2 --name m3_fr_s13 $LORA --lr 1e-5 --freeze-bottom 12 --exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model
for r in m3_lo_s13 m3_fr_s13; do
  for a in 75 50; do
    step w_${r}_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/$r --alpha 0.$a --out $M/w_${r}_a$a
    step w_${r}_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/w_${r}_a$a --name w_${r}_a$a --prefix none --splits dev devM --bs 32
  done
done
step lo_proxy $D -m lab.retrieval.v2.ood_dev_ar --models m3_lo_s13=$M/m3_lo_s13:none m3_fr_s13=$M/m3_fr_s13:none
# --- priority 3: seed 15 of the selected recipe, WiSE a.75 blends of seeds 14 / 15 with TEST + mixed
step m3_b_s15 $D -m lab.retrieval.v2.train_v2 --name m3_b_s15 $M3 --seed 15 $X
for s in 14 15; do
  step wm_s${s}_merge $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m3_b_s$s --alpha 0.75 --out $M/wm_a75_s$s
  step wm_s${s}_test $D -m lab.retrieval.v2.eval_v2 --model $M/wm_a75_s$s --name wm_a75_s$s --prefix none --splits dev test --mixed --bs 32
done
echo DRIVE12_DONE >> $L/queues.done
