#!/usr/bin/env bash
# Stage 16 (coordinator: steps 4, A, D, B, C):  final select -> ONE MTEB(ara) scoring of the pick (final_m3) -> TEST of the pick (+ blends of seeds 14/15)
#  -> A confirmatory replication (new seeds 16, 17; recipe frozen = the pick) -> D teacher=final_m3, students (KD-only, 2 epochs, 3 seeds) + final_base / final_small
#  -> B post-hoc blend curve -> C MTEB paired statistics -> guard split (titles) + reranker pairing.  Output files are verified after every step (run_docker.sh masks exit codes).
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models; H=runs/retrieval/v2/models; R=runs/retrieval/v2/results; P=runs/retrieval/v2/PROGRESS.md
S=runs/retrieval/v2/scripts; MC=$HOME/projects/mteb-arabic/results_cache_v2/results
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive16.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name" >> $L/drive16.log; }
need() { for f in "$@"; do [ -s "$f" ] || { echo "$(date +%T) FAIL: missing $f" >> $L/drive16.log; exit 1; }; done; }
jget() { python3 -c "import json;print(json.load(open('$1'))$2)"; }
mteb() { dir=$1; name=$2; base=$3; tasks=$4; step sc_$name bash $S/mteb_score.sh $dir $name $base $tasks
  n=$(ls $MC/$name/*/*.json 2>/dev/null | wc -l); echo "$(date +%T) MTEB result files for $name: $n" >> $L/drive16.log; }
while ! grep -q DRIVE15_DONE $L/queues.done 2>/dev/null; do sleep 30; done
need $R/suite_pick.json
PICK=$(jget $R/suite_pick.json "['pick']"); FAM=$(jget $R/suite_pick.json "['family']"); ALPHA=$(jget $R/suite_pick.json "['alpha']"); RULE=$(jget $R/suite_pick.json "['rule']")
RECIPE=$(echo "$PICK" | sed 's/_s13//')      # per-seed file naming of the blended models: <recipe>_s14 / _s15 / _s16 / _s17 (+ .test.json and .test.perq.npz)
FROZEN_AT=$(date -Is)
rm -rf $H/final_m3; cp -r $H/$PICK $H/final_m3; rm -rf $H/final_m3/ckpt $H/final_m3/lora_adapter; need $H/final_m3/config.json
echo "- $(date +%H:%M) RECIPE FROZEN at $FROZEN_AT (before any confirmatory seed is trained): family $FAM, alpha $ALPHA, pick $PICK, per-seed file recipe name $RECIPE. Seeds 16/17 are trained only after this line." >> $P
echo "- $(date +%H:%M) SUITE PICK (pre-registered rule, full pool): $PICK (family $FAM, alpha $ALPHA). $RULE. Saved as runs/retrieval/v2/models/final_m3 (clean sentence-transformers folder). ONE MTEB(ara) scoring now; reported whatever it is." >> $P
# ---- step 4: ONE MTEB(ara) scoring of the pick
mteb final_m3 final_m3 BAAI/bge-m3 ""
echo "FINAL_M3_SCORED" >> $L/queues.done
# ---- TEST / mixed of the pick; per-seed blends of seeds 14 / 15 when the family is single-seed WiSE (the 3-seed reporting rule)
step te_final_m3 $D -m lab.retrieval.v2.eval_v2 --model $M/final_m3 --name final_m3 --prefix none --splits dev test --mixed --bs 32; need $R/final_m3.test.json
if [ "$FAM" = "wise" ]; then
  for s in 14 15; do n=${RECIPE}_s$s
    step mg_$n $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m3_b_s$s --alpha $ALPHA --out $M/$n; need $H/$n/config.json
    step te_$n $D -m lab.retrieval.v2.eval_v2 --model $M/$n --name $n --prefix none --splits dev test --mixed --bs 32; need $R/$n.test.json
  done
fi
# ---- A: CONFIRMATORY REPLICATION (recipe frozen = the pick; seeds 16, 17 never used before; pre-registration 3 in PROGRESS.md)
X="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json --test --keep-model"
case "$FAM" in
  wise|soup) RAW=m3_b;  SEEDS="16 17"; TRAIN="--base bge-m3 --data v2 --negs same:1 --epochs 1 --lr 1e-5 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"; [ "$FAM" = soup ] && SEEDS="16 17 18" ;;
  freeze)    RAW=m3_fr; SEEDS="16 17"; TRAIN="--base bge-m3 --data v2 --negs same:1 --epochs 1 --scale 50 --loss cached --mini-batch 16 --eval-bs 32 --lr 1e-5 --freeze-bottom 12" ;;
  lora)      RAW=m3_lo; SEEDS="16 17"; TRAIN="--base bge-m3 --data v2 --negs same:1 --epochs 1 --scale 50 --loss cached --mini-batch 16 --eval-bs 32 --lr 1e-4 --lora 32" ;;
esac
echo "- $(date +%H:%M) CONFIRMATORY REPLICATION starts (recipe frozen at $FROZEN_AT): frozen recipe = family $FAM, alpha $ALPHA (pick $PICK); raw run name prefix $RAW; NEW seeds: $SEEDS (never used in any selection or training so far). Training arguments: $TRAIN $X" >> $P
for s in $SEEDS; do step conf_train_s$s $D -m lab.retrieval.v2.train_v2 --name ${RAW}_s$s $TRAIN --seed $s $X; need $R/${RAW}_s$s.test.json $H/${RAW}_s$s/config.json; done
if [ "$FAM" = "soup" ]; then
  CONF="${RECIPE}_s16"; MEM=""; for s in $SEEDS; do MEM="$MEM $M/${RAW}_s$s"; done
  step mg_$CONF $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --soup $MEM --alpha $ALPHA --out $M/$CONF; need $H/$CONF/config.json
  CONFS="$CONF"
else
  CONFS=""
  for s in $SEEDS; do n=${RECIPE}_s$s
    if [ "$ALPHA" = "1.0" ]; then rm -rf $H/$n; cp -r $H/${RAW}_s$s $H/$n; rm -rf $H/$n/ckpt $H/$n/lora_adapter
    else step mg_$n $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/${RAW}_s$s --alpha $ALPHA --out $M/$n; fi
    need $H/$n/config.json; CONFS="$CONFS $n"
  done
fi
for n in $CONFS; do
  step te_$n $D -m lab.retrieval.v2.eval_v2 --model $M/$n --name $n --prefix none --splits dev test --mixed --bs 32; need $R/$n.test.json
  mteb $n $n BAAI/bge-m3 ""
done
python3 - "$FROZEN_AT" "$PICK" "$FAM" "$ALPHA" "$RECIPE" "$RAW" "$SEEDS" "$CONFS" <<'EOP'
import json, sys
fz, pick, fam, alpha, recipe, raw, seeds, confs = sys.argv[1:9]
files = lambda n: {"test": f"{n}.test.json", "test_perq": f"{n}.test.perq.npz", "dev": f"{n}.dev.json", "mteb": f"results_cache_v2/results/{n}/"}
out = {"recipe_frozen_at": fz, "pick": pick, "family": fam, "alpha": float(alpha), "recipe": recipe, "new_seeds": seeds.split(),
       "blended_models": {n: files(n) for n in confs.split()}, "raw_models": {f"{raw}_s{s}": files(f"{raw}_s{s}") for s in seeds.split()},
       "note": "Confirmatory replication (coordinator addition A): seeds never used in any selection step; reported whatever they show."}
json.dump(out, open("runs/retrieval/v2/results/confirmatory.json", "w"), indent=1)
EOP
echo "CONF_DONE" >> $L/queues.done
# ---- D: teacher = final_m3, students (KD-only, 2 epochs, 3 seeds, e5-base and e5-small)
step teacher_m3f $D -m lab.retrieval.v2.teacher_v2 --model $M/final_m3 --tag m3f --prefix none --data v2
need runs/retrieval/v2/data/teacher_m3f_q.npz runs/retrieval/v2/data/teacher_m3f_p.npz
echo "- $(date +%H:%M) teacher embeddings re-computed from final_m3 (tag m3f); students distil from it: KD-only, 2 epochs, 3 seeds, e5-base and e5-small; final student = 3-seed soup unless its dev macro is > 0.5 below the mean of its seeds (pre-registration 3)." >> $P
KD="--data v2 --negs same:1 --epochs 2 --kd 1 --kd-only --t-scale 20 --teacher m3f"
step e0_base_e5base $D -m lab.retrieval.v2.eval_v2 --model intfloat/multilingual-e5-base --name base_e5-base --prefix e5 --splits dev test --mixed
for s in 13 14 15; do
  step st_b_s$s $D -m lab.retrieval.v2.train_v2 --name st_b_s$s --base e5-base $KD --seed $s $X; need $R/st_b_s$s.test.json $H/st_b_s$s/config.json
  step st_s_s$s $D -m lab.retrieval.v2.train_v2 --name st_s_s$s $KD --seed $s $X; need $R/st_s_s$s.test.json $H/st_s_s$s/config.json
done
for k in b s; do
  step st_${k}_soup_merge $D -m lab.retrieval.v2.merge_v2 --soup $M/st_${k}_s13 $M/st_${k}_s14 $M/st_${k}_s15 --out $M/st_${k}_soup; need $H/st_${k}_soup/config.json
  step st_${k}_soup_eval $D -m lab.retrieval.v2.eval_v2 --model $M/st_${k}_soup --name st_${k}_soup --prefix e5 --splits dev test devM --bs 64; need $R/st_${k}_soup.test.json
done
pickst() { python3 - $1 <<'PY'
import json, sys
k = sys.argv[1]
g = lambda n: json.load(open(f"runs/retrieval/v2/results/{n}.dev.json"))["directions"]["macro"]["mrr@10"] * 100
mean = sum(g(f"st_{k}_s{s}") for s in (13, 14, 15)) / 3
print(f"st_{k}_s13" if g(f"st_{k}_soup") < mean - 0.5 else f"st_{k}_soup")
PY
}
FB=$(pickst b); FS=$(pickst s)
for pair in "b $FB final_base intfloat/multilingual-e5-base" "s $FS final_small intfloat/multilingual-e5-small"; do set -- $pair
  rm -rf $H/$3; cp -r $H/$2 $H/$3; rm -rf $H/$3/ckpt; need $H/$3/config.json
  echo "- $(date +%H:%M) STUDENT PICK (pre-registered rule): $3 = $2" >> $P
  mteb $3 $3 $4 ""
done
echo "STUDENTS_SCORED" >> $L/queues.done
step e0_devM_e5 $D -m lab.retrieval.v2.eval_v2 --model intfloat/multilingual-e5-base --name base_e5-base --prefix e5 --splits devM
step e0_devM_e5s $D -m lab.retrieval.v2.eval_v2 --model intfloat/multilingual-e5-small --name base_e5-small --prefix e5 --splits devM
# ---- B: post-hoc blend curve (reuse what exists; run only the missing cheap evals: TEST of the blends, guard-only MTEB runs)
for n in w_m3b_a30 w_m3b_a40 wm_a50 w_m3b_a60; do
  [ -s $R/$n.test.json ] || { step bt_$n $D -m lab.retrieval.v2.eval_v2 --model $M/$n --name $n --prefix none --splits test --bs 32; need $R/$n.test.json; }
done
GT="MIRACLRetrievalHardNegatives.v2,NamaaMrTydiReranking,SemRel24STS"
for n in w_m3b_a30 w_m3b_a40 wm_a50 w_m3b_a60 m3_b_s13; do
  if [ "$n" = "$PICK" ]; then continue; fi        # the pick's guards come from the full final_m3 scoring
  [ -s $MC/alpha_$n/*/SemRel24STS.json ] 2>/dev/null || mteb $n alpha_$n BAAI/bge-m3 $GT
done
python3 lab/retrieval/v2/alpha_curve.py > $L/alpha_curve.log 2>&1; need $R/alpha_curve.json
# ---- C: MTEB(ara) paired statistics (CPU)
PAIRS="final_m3:BAAI__bge-m3 final_m3_wa75_s13:BAAI__bge-m3"; for n in $CONFS; do PAIRS="$PAIRS $n:BAAI__bge-m3"; done
python3 lab/retrieval/v2/mteb_paired.py $PAIRS \
  final_base:intfloat__multilingual-e5-base final_small:intfloat__multilingual-e5-small > $L/mteb_paired.log 2>&1; need $R/mteb_paired.json
# ---- titled contamination-split diagnostic and reranker pairing
step guard_split $D -m lab.retrieval.v2.guard_split --models bge-m3=BAAI/bge-m3:none final_m3=$M/final_m3:none m3_b_s13=$M/m3_b_s13:none wm_a75=$M/wm_a75:none v1=/lab/runs/retrieval/models/e5s_hard1_xl_e1_s13:e5 e5-base=intfloat/multilingual-e5-base:e5 e5-small=intfloat/multilingual-e5-small:e5 final_base=$M/final_base:e5 final_small=$M/final_small:e5
step guard_split_report env NO_GPU=1 $D -m lab.retrieval.v2.guard_split --report
step rerank $D -m lab.retrieval.v2.rerank_v2 --first-stage base_bge-m3 final_m3 st_b_soup st_s_soup base_e5-small base_e5-base
echo DRIVE16_DONE >> $L/queues.done
