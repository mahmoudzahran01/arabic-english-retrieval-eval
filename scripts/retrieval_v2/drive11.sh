#!/usr/bin/env bash
# Stage 11: preservation run with gradient checkpointing (the non-cached loss ran out of memory: memguard killed it at 7 GB available), its WiSE blends; then the KD ablations that
# crashed on a padding-negative without a teacher embedding (fixed). One GPU job at a time.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model"
M3="--base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
B="--base e5-base --negs same:1 --seed 13"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive11.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive11.log; }
rm -rf $M/m3_p_s13 $M/w_m3_p_s13_a75 $M/w_m3_p_s13_a50 2>/dev/null
step r2_pres $D -m lab.retrieval.v2.train_v2 --name m3_p_s13 $M3 --pres 0.5 --teacher bgem3 --grad-ckpt $X
for a in 75 50; do
  step w_m3_p_s13_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m3_p_s13 --alpha 0.$a --out $M/w_m3_p_s13_a$a
  step w_m3_p_s13_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/w_m3_p_s13_a$a --name w_m3_p_s13_a$a --prefix none --splits dev devM --bs 32
done
for spec in bge-m3=BAAI/bge-m3:none e5-base=intfloat/multilingual-e5-base:e5 wm_a75=$M/wm_a75:none m3_b_s13=$M/m3_b_s13:none v1=/lab/runs/retrieval/models/e5s_hard1_xl_e1_s13:e5; do
  step gs_${spec%%=*} $D -m lab.retrieval.v2.guard_split --models $spec
done
echo "$(date +%T) START gs_report" >> $L/drive11.log; env NO_GPU=1 $D -m lab.retrieval.v2.guard_split --report > $L/gs_report.log 2>&1; echo "$(date +%T) END gs_report rc=$?" >> $L/drive11.log
step k1_b_kd3 $D -m lab.retrieval.v2.train_v2 --name f5_b_kd3_s13 $B --data v2 --epochs 1 --kd 3 --t-scale 20 --teacher m3b $X
step k2_b_kdonly $D -m lab.retrieval.v2.train_v2 --name f5_b_kdonly_s13 $B --data v2 --epochs 1 --kd 1 --kd-only --t-scale 20 --teacher m3b $X
step k3_b_kd_e2 $D -m lab.retrieval.v2.train_v2 --name f5_b_kd_e2_s13 $B --data v2 --epochs 2 --kd 1 --t-scale 20 --teacher m3b $X
echo DRIVE11_DONE >> $L/queues.done
