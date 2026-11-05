#!/usr/bin/env bash
# Early read on v2-m3 (coordinator): fine-tune bge-m3 itself on the CURRENT filtered data (11.9k-translation set), dev only, as soon as the GPU gate clears.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh
echo "$(date +%T) START m3_a_s13" >> $L/drive3.log
$G $D -m lab.retrieval.v2.train_v2 --name m3_a_s13 --base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32 --exclude /lab/runs/retrieval/v2/data/exclude_qids.json --keep-model > $L/m3_a_s13.log 2>&1
echo "$(date +%T) END m3_a_s13 rc=$?" >> $L/drive3.log
