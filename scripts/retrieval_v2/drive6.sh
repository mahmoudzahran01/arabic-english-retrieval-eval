#!/usr/bin/env bash
# Stage 6 (priorities after the m3 result): the selected final (v2-m3 + WiSE alpha 0.75, chosen on dev + mMARCO proxy only) gets test/mixed, reranker pairing and the guards;
# then seeds 14/15 of the m3 recipe (3-seed CIs). One GPU job at a time via gate.sh.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --test --keep-model"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive6.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive6.log; }
step wm75_test $D -m lab.retrieval.v2.eval_v2 --model $M/wm_a75 --name wm_a75 --prefix none --splits test --mixed --bs 32
step wm75_rerank $D -m lab.retrieval.v2.rerank_v2 --first-stage wm_a75
step wm75_guard $D -m lab.retrieval.v2.mteb_eval --models m3w75=$M/wm_a75 --suite guard --bs 32
step teacher_m3b $D -m lab.retrieval.v2.teacher_v2 --model $M/m3_b_s13 --tag m3b --prefix none --data v2
step f5_b_kd $D -m lab.retrieval.v2.train_v2 --name f5_b_kd_s13 --base e5-base --data v2 --negs same:1 --epochs 1 --seed 13 --kd 1 --t-scale 20 --teacher m3b $X
step f5_s_kd $D -m lab.retrieval.v2.train_v2 --name f5_s_kd_s13 --data v2 --negs same:1 --epochs 1 --seed 13 --kd 1 --t-scale 20 --teacher m3b $X
step m3_b_s14 $D -m lab.retrieval.v2.train_v2 --name m3_b_s14 --base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 14 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32 $X
bash runs/retrieval/v2/scripts/drive8.sh
echo DRIVE6_DONE >> $L/queues.done
