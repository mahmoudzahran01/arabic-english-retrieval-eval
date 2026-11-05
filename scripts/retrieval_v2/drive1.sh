#!/usr/bin/env bash
# Stage 1 of the resumed plan (priorities 1 and 2): guards for the raw v2 finals, proxy for all models, WiSE blends of v2 (dev + proxy only).
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; M=/lab/runs/retrieval/v2/models; G=runs/retrieval/v2/scripts/gate.sh
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive1.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive1.log; }
step g1_guard_v2s $D -m lab.retrieval.v2.mteb_eval --models v2s_s13=$M/p3_s_e1_s13 --suite guard --bs 32
step g2_guard_v2b $D -m lab.retrieval.v2.mteb_eval --models v2b_s13=$M/p3_b_e1_s13 --suite guard --bs 32
step p1_proxy $D -m lab.retrieval.v2.ood_dev_ar --models base_e5-small=intfloat/multilingual-e5-small:e5 v1_s13=/lab/runs/retrieval/models/e5s_hard1_xl_e1_s13:e5 p3_s_e1_s13=$M/p3_s_e1_s13:e5 p3_s_lr1e5_s13=$M/p3_s_lr1e5_s13:e5 p3_s_lr2e5_s13=$M/p3_s_lr2e5_s13:e5 base_e5-base=intfloat/multilingual-e5-base:e5 p3_b_e1_s13=$M/p3_b_e1_s13:e5 bge-m3=BAAI/bge-m3:none
for a in 75 50; do
  step w_s_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base intfloat/multilingual-e5-small --ft $M/p3_s_e1_s13 --alpha 0.$a --out $M/w_s_a$a
  step w_s_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/w_s_a$a --name w_s_a$a --prefix e5 --splits dev
  step w_s_a${a}_proxy $D -m lab.retrieval.v2.ood_dev_ar --models w_s_a$a=$M/w_s_a$a:e5
done
for a in 75 50; do
  step w_b_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base intfloat/multilingual-e5-base --ft $M/p3_b_e1_s13 --alpha 0.$a --out $M/w_b_a$a
  step w_b_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/w_b_a$a --name w_b_a$a --prefix e5 --splits dev
  step w_b_a${a}_proxy $D -m lab.retrieval.v2.ood_dev_ar --models w_b_a$a=$M/w_b_a$a:e5
done
echo DRIVE1_DONE >> $L/queues.done
