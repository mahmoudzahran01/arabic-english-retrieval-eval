#!/usr/bin/env bash
# v2.1 SELECTION queue (PRE-REGISTERED in PROGRESS.md 2026-11-05 ~08:50). No TEST, no MTEB(ara) score of any candidate. Sources lib_drive.sh (step logs rc; failures -> logs/failed_steps).
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; LOGF=$L/drive18.log
M=/lab/runs/retrieval/v2/models; H=runs/retrieval/v2/models; R=runs/retrieval/v2/results; MC=$HOME/projects/mteb-arabic/results_cache_v2/results; P=runs/retrieval/v2/PROGRESS.md
. runs/retrieval/v2/scripts/lib_drive.sh
EX=/lab/runs/retrieval/v2/data/exclude_qids_v21.json
need runs/retrieval/v2/data/exclude_qids_v21.json
# ---- 1. mining: data-v3 translations with the baseline miner (as every other part); Arabic-monolingual rows with v2-m3 (lever b)
step mine_v3 $D -m lab.retrieval.v2.mine_v2 --part v3 --slices train_xl3_ar2en --modes same
need runs/retrieval/v2/mined/raw_same_e5-small_v3.parquet
step mine_ar $D -m lab.retrieval.v2.mine_v2 --part v1 --slices train_tydi_ar,train_arcd_ar --miner $M/final_m3 --tag m3f --prefix none --modes same
need runs/retrieval/v2/mined/raw_same_m3f_v1.parquet
# ---- 2. Namaa-like proxy on the existing models with known NamaaMrTydi scores = the validity gate; also gives the incumbent rows
ARRS="bge-m3=BAAI/bge-m3:none e5-base=intfloat/multilingual-e5-base:e5 e5-small=intfloat/multilingual-e5-small:e5 v1=/lab/runs/retrieval/models/e5s_hard1_xl_e1_s13:e5"
for n in w_m3b_a40 wm_a75 w_m3b_a30 wm_a50 w_m3b_a60 m3_b_s13 w_m3b_a40_s16 w_m3b_a40_s17; do ARRS="$ARRS $n=$M/$n:none"; done
ARRS="$ARRS final_base=$M/final_base:e5 final_small=$M/final_small:e5"
step arr0 $D -m lab.retrieval.v2.ar_rerank_proxy --models $ARRS
need $R/arr/bge-m3.json $R/arr/final_small.json $R/arr/w_m3b_a40.json
python3 lab/retrieval/v2/arr_validate.py > $L/arr_validate.log 2>&1; need $R/arr/_gate.json
echo "- $(date +%H:%M) v2.1 ar_rerank validity gate: $(tail -1 $L/arr_validate.log)" >> $P
# incumbent family + reference rows need the v2 suite components only if they are missing (they exist from v2): nothing to do
# ---- 3. the three trainings (seed 13), identical m3_b recipe except for the levers; v2.1 exclusion list
COMMON="--base bge-m3 --negs same:1 --epochs 1 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32 --exclude $EX --keep-model --seed 13"
echo "- $(date +%H:%M) v2.1 trainings start (pre-registration 08:50): R_A / R_B / R_AB, seed 13, exclusion list exclude_qids_v21.json" >> $P
step tr_A  $D -m lab.retrieval.v2.train_v2 --name m21_A_s13  $COMMON --data v3 --drop-slices train_squad_en_x
need $H/m21_A_s13/config.json $R/m21_A_s13.dev.json
step tr_B  $D -m lab.retrieval.v2.train_v2 --name m21_B_s13  $COMMON --data v2 --ar-neg-model m3f
need $H/m21_B_s13/config.json $R/m21_B_s13.dev.json
step tr_AB $D -m lab.retrieval.v2.train_v2 --name m21_AB_s13 $COMMON --data v3 --drop-slices train_squad_en_x --ar-neg-model m3f
need $H/m21_AB_s13/config.json $R/m21_AB_s13.dev.json
# ---- 4. blends alpha 0.3 / 0.4 / 0.5 / 0.6 of each, then dev + devM
ALL=""
for rec in A B AB; do for a in 30 40 50 60; do n=w_m21_${rec}_a$a
  step mg_$n $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m21_${rec}_s13 --alpha 0.$a --out $M/$n; need $H/$n/config.json
  step dv_$n $D -m lab.retrieval.v2.eval_v2 --model $M/$n --name $n --prefix none --splits dev devM --bs 32; need $R/$n.dev.json $R/$n.devM.json
  ALL="$ALL $n=$M/$n:none"
done; done
# ---- 5. proxy, suite, ar_rerank for the 12 candidates
step px_v21 $D -m lab.retrieval.v2.ood_dev_ar --models $ALL
step su_v21 $D -m lab.retrieval.v2.suite_v2 --models $ALL
step ar_v21 $D -m lab.retrieval.v2.ar_rerank_proxy --models $ALL
for rec in A B AB; do for a in 30 40 50 60; do need $R/suite/w_m21_${rec}_a$a.json $R/arr/w_m21_${rec}_a$a.json; done; done
# ---- 6. selection (pre-registered decision rule)
step sel_v21 env NO_GPU=1 $D -m lab.retrieval.v2.suite_select_v21 --write
need $R/suite_v21_pick.json
echo "- $(date +%H:%M) v2.1 SELECTION DONE: $(python3 -c "import json;d=json.load(open('$R/suite_v21_pick.json'));print(d['decision'], d['pick'], '|', d['reason'])")" >> $P
echo DRIVE18_DONE >> $L/queues.done
