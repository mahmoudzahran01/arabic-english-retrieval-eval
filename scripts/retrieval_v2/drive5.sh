#!/usr/bin/env bash
# After drive2b (mine -> f4_b -> f4_s): v2-m3 on the FULL v2 data (filtered), test + keep; then WiSE blends of it against bge-m3 (dev + mMARCO proxy, no guards).
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --test --keep-model"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive5.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive5.log; }
while ! grep -q DRIVE2_DONE $L/queues.done 2>/dev/null; do sleep 30; done
step m3_b_s13 $D -m lab.retrieval.v2.train_v2 --name m3_b_s13 --base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32 $X
for a in 75 50; do
  step wm_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m3_b_s13 --alpha 0.$a --out $M/wm_a$a
  step wm_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/wm_a$a --name wm_a$a --prefix none --splits dev --bs 32
done
step wm_proxy $D -m lab.retrieval.v2.ood_dev_ar --models bge-m3=BAAI/bge-m3:none m3_b_s13=$M/m3_b_s13:none wm_a75=$M/wm_a75:none wm_a50=$M/wm_a50:none base_e5-small=intfloat/multilingual-e5-small:e5 base_e5-base=intfloat/multilingual-e5-base:e5
echo DRIVE5_DONE >> $L/queues.done
