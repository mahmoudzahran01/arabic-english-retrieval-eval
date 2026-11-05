#!/usr/bin/env bash
# Stage 13 (coordinator priority 4): the <=300M students. Recipe = distillation from the fine-tuned v2-m3 (teacher embeddings tag m3b = m3_b_s13), KD-ONLY (best on dev among w=1 / w=3 / KD-only;
# epochs 1 or 2 chosen from the dev macro of f5_b_kdonly_s13 vs f5_b_kdonly_e2_s13: 2 only if it gains > 0.3). 3 seeds each (TEST + keep), base-model TEST baseline for the CIs, then WiSE blends of seed 13.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive13.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name rc=$?" >> $L/drive13.log; }
while ! grep -q DRIVE12_DONE $L/queues.done 2>/dev/null; do sleep 30; done
KDARGS=$(python3 - <<'PY'
import json
cands = {   # run name (dev-evaluated in the KD ablation) -> the student-recipe arguments it stands for
    "f5_b_kd_s13": "--epochs 1 --kd 1",
    "f5_b_kd3_s13": "--epochs 1 --kd 3",
    "f5_b_kdonly_s13": "--epochs 1 --kd 1 --kd-only",
    "f5_b_kd_e2_s13": "--epochs 2 --kd 1",
    "f5_b_kdonly_e2_s13": "--epochs 2 --kd 1 --kd-only",
}
best, arg = -1, ""
for n, a in cands.items():
    try:
        v = json.load(open(f"runs/retrieval/v2/results/{n}.dev.json"))["directions"]["macro"]["mrr@10"] * 100
    except Exception:
        continue
    if v > best + 0.2 or (abs(v - best) <= 0.2 and a.count("--") < arg.count("--")):   # ties within 0.2: fewer moving parts
        best, arg = v, a
print(arg)
PY
)
echo "$(date +%T) student recipe (best dev macro among the KD ablations): $KDARGS" >> $L/drive13.log
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --test --keep-model"
KD="--data v2 --negs same:1 --t-scale 20 --teacher m3b $KDARGS"
step e0_base_e5base $D -m lab.retrieval.v2.eval_v2 --model intfloat/multilingual-e5-base --name base_e5-base --prefix e5 --splits dev test --mixed
for s in 13 14 15; do
  step st_b_s$s $D -m lab.retrieval.v2.train_v2 --name st_b_s$s --base e5-base $KD --seed $s $X
  step st_s_s$s $D -m lab.retrieval.v2.train_v2 --name st_s_s$s $KD --seed $s $X
done
for k in b s; do
  base=$([ $k = b ] && echo intfloat/multilingual-e5-base || echo intfloat/multilingual-e5-small)
  for a in 75 50; do
    step w_st_${k}_a${a}_merge $D -m lab.retrieval.v2.merge_v2 --base $base --ft $M/st_${k}_s13 --alpha 0.$a --out $M/w_st_${k}_a$a
    step w_st_${k}_a${a}_dev $D -m lab.retrieval.v2.eval_v2 --model $M/w_st_${k}_a$a --name w_st_${k}_a$a --prefix e5 --splits dev devM --bs 64
  done
done
step e0_devM_e5 $D -m lab.retrieval.v2.eval_v2 --model intfloat/multilingual-e5-base --name base_e5-base --prefix e5 --splits devM
step e0_devM_e5s $D -m lab.retrieval.v2.eval_v2 --model intfloat/multilingual-e5-small --name base_e5-small --prefix e5 --splits devM
step e0_devM_st $D -m lab.retrieval.v2.eval_v2 --model $M/st_b_s13 --name st_b_s13 --prefix e5 --splits devM
step e0_devM_sts $D -m lab.retrieval.v2.eval_v2 --model $M/st_s_s13 --name st_s_s13 --prefix e5 --splits devM
echo DRIVE13_DONE >> $L/queues.done
