#!/usr/bin/env bash
# Stage 14 (coordinator re-prioritisation 2026-11-04 ~19:00): validation-suite selection of the final v2-m3.
# Order: seed 15 -> soup + WiSE blends -> suite for the non-LoRA candidates (interim table) -> LoRA / freeze + blends -> suite for them -> final table + pick (suite_pick.json).
# Pre-registration: PROGRESS.md "PRE-REGISTRATION: validation suite" (rule, components, pool).  No MTEB, no TEST number is read here.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models; H=runs/retrieval/v2/models
EXCL="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json"
X="$EXCL --test --keep-model"
M3="--base bge-m3 --data v2 --negs same:1 --epochs 1 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive14.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive14.log; }
merge() { out=$1; shift; step merge_$out $D -m lab.retrieval.v2.merge_v2 "$@" --out $M/$out; }
devs() { for n in "$@"; do step dev_$n $D -m lab.retrieval.v2.eval_v2 --model $M/$n --name $n --prefix none --splits dev devM --bs 32; done; }
SOUP="$M/m3_b_s13 $M/m3_b_s14 $M/m3_b_s15"
P1="bge-m3=BAAI/bge-m3:none m3_b_s13=$M/m3_b_s13:none"
for n in wm_a50 wm_a75; do P1="$P1 $n=$M/$n:none"; done
# 1. seed 15 of the m3_b recipe (needed for the soup)
step m3_b_s15 $D -m lab.retrieval.v2.train_v2 --name m3_b_s15 $M3 --seed 15 $X
# 2. WiSE of seed 13 at the two new alphas, soup raw, soup + WiSE
merge w_m3b_a40 --base BAAI/bge-m3 --ft $M/m3_b_s13 --alpha 0.40
merge w_m3b_a60 --base BAAI/bge-m3 --ft $M/m3_b_s13 --alpha 0.60
merge soup_m3 --soup $SOUP
for a in 40 50 60 75; do merge soup_a$a --base BAAI/bge-m3 --soup $SOUP --alpha 0.$a; done
devs w_m3b_a40 w_m3b_a60 soup_m3 soup_a40 soup_a50 soup_a60 soup_a75
NEW1=""
for n in w_m3b_a40 w_m3b_a60 soup_m3 soup_a40 soup_a50 soup_a60 soup_a75; do NEW1="$NEW1 $n=$M/$n:none"; done
step proxy1 $D -m lab.retrieval.v2.ood_dev_ar --models $P1 $NEW1
step suite1 $D -m lab.retrieval.v2.suite_v2 --models $P1 $NEW1
step select1 env NO_GPU=1 $D -m lab.retrieval.v2.suite_select
echo "$(date +%T) INTERIM suite table ready (logs/select1.log)" >> $L/drive14.log
# 3. parameter-limited runs (coordinator priority 2) + their WiSE blends
LORA="--base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
step lo_lora $D -m lab.retrieval.v2.train_v2 --name m3_lo_s13 $LORA --lr 1e-4 --lora 32 $EXCL --keep-model
step lo_freeze $D -m lab.retrieval.v2.train_v2 --name m3_fr_s13 $LORA --lr 1e-5 --freeze-bottom 12 $EXCL --keep-model
NEW2="m3_lo_s13=$M/m3_lo_s13:none m3_fr_s13=$M/m3_fr_s13:none"
for r in m3_lo_s13 m3_fr_s13; do
  for a in 75 50; do
    merge w_${r}_a$a --base BAAI/bge-m3 --ft $M/$r --alpha 0.$a
    devs w_${r}_a$a
    NEW2="$NEW2 w_${r}_a$a=$M/w_${r}_a$a:none"
  done
done
step proxy2 $D -m lab.retrieval.v2.ood_dev_ar --models $NEW2
step suite2 $D -m lab.retrieval.v2.suite_v2 --models $NEW2
step select2 env NO_GPU=1 $D -m lab.retrieval.v2.suite_select --write
echo DRIVE14_DONE >> $L/queues.done
