#!/usr/bin/env bash
# Stage 9: the preservation run (labels now 4x smaller after memguard killed the first attempts at 7 GB available), its WiSE blends; then the baseline guards.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model"
M3="--base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive9.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive9.log; }
rm -rf $M/m3_p_s13 $M/m3_rp_s13 $M/w_m3_p_s13_a75 $M/w_m3_p_s13_a50 $M/w_m3_rp_s13_a75 $M/w_m3_rp_s13_a50 2>/dev/null
step r2_pres $D -m lab.retrieval.v2.train_v2 --name m3_p_s13 $M3 --pres 0.5 --teacher bgem3 $X
for a in 75 50; do
  step w_m3_p_s13_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m3_p_s13 --alpha 0.$a --out $M/w_m3_p_s13_a$a
  step w_m3_p_s13_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/w_m3_p_s13_a$a --name w_m3_p_s13_a$a --prefix none --splits dev devM --bs 32
done
step b1_guard_bgem3 $D -m lab.retrieval.v2.mteb_eval --models bge-m3 --suite guard --bs 32
step b2_guard_e5s_v1 $D -m lab.retrieval.v2.mteb_eval --models e5-small v1=/lab/runs/retrieval/models/e5s_hard1_xl_e1_s13 --suite guard --bs 32
step b3_guard_e5base $D -m lab.retrieval.v2.mteb_eval --models e5-base --suite guard --bs 32
echo DRIVE9_DONE >> $L/queues.done
