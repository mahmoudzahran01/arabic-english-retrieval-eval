#!/usr/bin/env bash
# Stage 10 (after drive9): KD ablations for the <=300M students with the v2-m3 teacher (m3b = m3_b_s13 embeddings), dev-selected; v3 data as the last lever.
# One GPU job at a time via gate.sh.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model"
B="--base e5-base --negs same:1 --seed 13"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive10.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive10.log; }
while ! grep -q DRIVE9_DONE $L/queues.done 2>/dev/null; do sleep 30; done
step k1_b_kd3 $D -m lab.retrieval.v2.train_v2 --name f5_b_kd3_s13 $B --data v2 --epochs 1 --kd 3 --t-scale 20 --teacher m3b $X
step k2_b_kdonly $D -m lab.retrieval.v2.train_v2 --name f5_b_kdonly_s13 $B --data v2 --epochs 1 --kd 1 --kd-only --t-scale 20 --teacher m3b $X
step k3_b_kd_e2 $D -m lab.retrieval.v2.train_v2 --name f5_b_kd_e2_s13 $B --data v2 --epochs 2 --kd 1 --t-scale 20 --teacher m3b $X
echo DRIVE10_DONE >> $L/queues.done
