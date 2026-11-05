#!/usr/bin/env bash
# Stage 2 (translation of the 22.3k planned questions is complete): merge, contamination filter, mine, retrain base and small on the FULL v2 set. One GPU job at a time via gate.sh.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --test --keep-model"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive2.log; "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive2.log; }
step f0_data env NO_GPU=1 $D -m lab.retrieval.v2.data_v2
step f0_contam env NO_GPU=1 $D -m lab.retrieval.v2.contam_v2
step f1_mine $G $D -m lab.retrieval.v2.mine_v2 --part xl2
step f4_b_e1_s13 $G $D -m lab.retrieval.v2.train_v2 --name f4_b_e1_s13 --base e5-base --data v2 --negs same:1 --epochs 1 --seed 13 $X
step f4_s_e1_s13 $G $D -m lab.retrieval.v2.train_v2 --name f4_s_e1_s13 --data v2 --negs same:1 --epochs 1 --seed 13 $X
echo DRIVE2_DONE >> $L/queues.done
