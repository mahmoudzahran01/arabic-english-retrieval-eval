#!/usr/bin/env bash
# v2.1 CONFIRMATION queue (runs only if the pre-registered selection says "confirm"): seeds 18, 19 of the pick's recipe, blended with the pick's alpha, TEST + mixed, ONE full MTEB(ara) scoring per seed, adoption decision.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; LOGF=$L/drive19.log; S=runs/retrieval/v2/scripts
M=/lab/runs/retrieval/v2/models; H=runs/retrieval/v2/models; R=runs/retrieval/v2/results; MC=$HOME/projects/mteb-arabic/results_cache_v2/results; P=runs/retrieval/v2/PROGRESS.md
. $S/lib_drive.sh
while ! grep -q DRIVE18_DONE $L/queues.done 2>/dev/null; do sleep 30; done
need $R/suite_v21_pick.json
J() { python3 -c "import json;print(json.load(open('$R/suite_v21_pick.json'))$1)"; }
if [ "$(J "['decision']")" != "confirm" ]; then
  echo "- $(date +%H:%M) v2.1: selection decision = stay -> no confirmation run; v2 stays (v2.1 reported as a negative result)." >> $P; echo DRIVE19_DONE >> $L/queues.done; exit 0; fi
REC=$(J "['recipe']"); ALPHA=$(J "['alpha']"); PICK=$(J "['pick']")
case "$REC" in A) DATA="--data v3 --drop-slices train_squad_en_x" ;; B) DATA="--data v2 --ar-neg-model m3f" ;; AB) DATA="--data v3 --drop-slices train_squad_en_x --ar-neg-model m3f" ;; *) echo "bad recipe $REC" >> $LOGF; exit 1 ;; esac
EX=/lab/runs/retrieval/v2/data/exclude_qids_v21.json
echo "- $(date +%H:%M) v2.1 CONFIRMATION starts: pick $PICK (recipe $REC, alpha $ALPHA); fresh seeds 18, 19; data args: $DATA; the seed-13 pick is never scored on TEST/MTEB." >> $P
for s in 18 19; do
  step cf_tr_$s $D -m lab.retrieval.v2.train_v2 --name m21_${REC}_s$s --base bge-m3 --negs same:1 --epochs 1 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32 --exclude $EX --keep-model --seed $s $DATA
  need $H/m21_${REC}_s$s/config.json
  step cf_mg_$s $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m21_${REC}_s$s --alpha $ALPHA --out $M/v21_s$s; need $H/v21_s$s/config.json
  step cf_te_$s $D -m lab.retrieval.v2.eval_v2 --model $M/v21_s$s --name v21_s$s --prefix none --splits dev test --mixed --bs 32; need $R/v21_s$s.test.json $R/v21_s$s.test.perq.npz
  step cf_sc_$s bash $S/mteb_score.sh v21_s$s v21_s$s BAAI/bge-m3 ""
  n=$(count_mteb v21_s$s); echo "$(date +%T) MTEB result files for v21_s$s: $n" >> $LOGF; [ "$n" -ge 19 ] || { echo "$(date +%T) FAIL: only $n MTEB files for v21_s$s" >> $LOGF; exit 1; }
done
PYTHONPATH=lab/retrieval/v2 python3 lab/retrieval/v2/v21_adopt.py > $L/v21_adopt.log 2>&1; need $R/v21_adoption.json
echo "- $(date +%H:%M) v2.1 ADOPTION (judged once, seeds 18/19): $(python3 -c "import json;d=json.load(open('$R/v21_adoption.json'));print(d['decision'], d['criteria'])")" >> $P
echo DRIVE19_DONE >> $L/queues.done
