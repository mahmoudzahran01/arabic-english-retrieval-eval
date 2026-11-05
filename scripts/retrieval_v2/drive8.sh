#!/usr/bin/env bash
# Stage 8 (coordinator: fix the MIRACL forgetting properly). Order: replay run -> preservation run -> replay+preservation run -> WiSE sweep of each, all judged on
# DEV + devM (500 held-out MIRACL-ar TRAIN queries) + the mMARCO proxy, never on the MTEB guards.
# Pre-registered rule (written to PROGRESS.md BEFORE any of these runs): a candidate qualifies if (i) devM MRR@10 >= bge-m3's devM MRR@10 - 1.0, (ii) dev ar->en >= 79.0
# (~ TEST 72), (iii) dev macro >= (m3_b dev macro) - 1.5; among the qualifying ones take the highest dev macro (ties: fewer moving parts).
# One GPU job at a time via gate.sh.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model"
M3="--base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive8.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive8.log; }
while [ ! -f runs/retrieval/v2/data/manifest_miracl.json ]; do sleep 30; done
# contamination filter over the new rows (CPU, no gate) - rewrites exclude_qids.json (v2 rows keep their decisions)
echo "$(date +%T) START g1_contam_replay" >> $L/drive8.log; env NO_GPU=1 $D -m lab.retrieval.v2.contam_v2 > $L/g1_contam_replay.log 2>&1; echo "$(date +%T) END g1_contam_replay rc=$?" >> $L/drive8.log
step t1_teacher_bgem3 $D -m lab.retrieval.v2.teacher_v2 --tag bgem3 --prefix none --data v2
for m in bge-m3=BAAI/bge-m3 m3_b_s13=$M/m3_b_s13 wm_a75=$M/wm_a75 wm_a50=$M/wm_a50; do
  n=${m%%=*}; pth=${m#*=}
  step e0_devM_$n $D -m lab.retrieval.v2.eval_v2 --model $pth --name $n --prefix none --splits devM --bs 32
done
step r1_replay $D -m lab.retrieval.v2.train_v2 --name m3_r_s13 $M3 --replay --replay-x 3 $X
step r2_pres $D -m lab.retrieval.v2.train_v2 --name m3_p_s13 $M3 --pres 0.5 --teacher bgem3 $X
step r3_rp $D -m lab.retrieval.v2.train_v2 --name m3_rp_s13 $M3 --replay --replay-x 3 --pres 0.5 --teacher bgem3 $X
for r in m3_r_s13 m3_p_s13 m3_rp_s13; do
  for a in 75 50; do
    step w_${r}_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/$r --alpha 0.$a --out $M/w_${r}_a$a
    step w_${r}_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/w_${r}_a$a --name w_${r}_a$a --prefix none --splits dev devM --bs 32
  done
done
echo DRIVE8_DONE >> $L/queues.done
