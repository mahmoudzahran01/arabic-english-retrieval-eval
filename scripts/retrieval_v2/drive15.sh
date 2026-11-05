#!/usr/bin/env bash
# Stage 15 = AMENDMENT 2 of the validation-suite selection (PROGRESS.md "AMENDMENT 2", written before this ran).
#  (1) freeze proxy + suite  (2) select on the pool without LoRA -> winning family F  (3) ONE extra pair of alphas for F only: 0.30 (+0.35 only if 0.30 is ineligible)
#  (4) LoRA merge self-test, LoRA rerun, blends, proxy, suite  (5) FINAL select on the full pool -> results/suite_pick.json.
# Every step's OUTPUT FILES are verified (run_docker.sh masks exit codes, which hid the LoRA crash in drive14).
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models; H=runs/retrieval/v2/models; R=runs/retrieval/v2/results
EXCL="--exclude /lab/runs/retrieval/v2/data/exclude_qids.json"
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive15.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name" >> $L/drive15.log; }
need() { for f in "$@"; do [ -s "$f" ] || { echo "$(date +%T) FAIL: missing $f" >> $L/drive15.log; exit 1; }; done; }
proxy_has() { python3 - "$@" <<'PY' || { echo "$(date +%T) FAIL: proxy result missing" >> runs/retrieval/v2/logs/drive15.log; exit 1; }
import json, sys
d = json.load(open("runs/retrieval/v2/results/ood_dev_ar.json"))
miss = [n for n in sys.argv[1:] if n not in d]
sys.exit(1 if miss else 0)
PY
}
SOUP="$M/m3_b_s13 $M/m3_b_s14 $M/m3_b_s15"
# ---- (1) freeze proxy + suite
FR="m3_fr_s13 w_m3_fr_s13_a75 w_m3_fr_s13_a50"; SPEC=""
for n in $FR; do need $H/$n/config.json $R/$n.dev.json $R/$n.devM.json; SPEC="$SPEC $n=$M/$n:none"; done
step proxy3 $D -m lab.retrieval.v2.ood_dev_ar --models $SPEC
proxy_has $FR
step suite3 $D -m lab.retrieval.v2.suite_v2 --models $SPEC
need $R/suite/m3_fr_s13.json $R/suite/w_m3_fr_s13_a75.json $R/suite/w_m3_fr_s13_a50.json
# ---- (2) select on the pool without LoRA
step select3 env NO_GPU=1 $D -m lab.retrieval.v2.suite_select
FAM=$(grep -o "WINNING_FAMILY=[a-z]*" $L/select3.log | tail -1 | cut -d= -f2)
echo "$(date +%T) winning family (rule as registered) = $FAM" >> $L/drive15.log
case "$FAM" in
  wise)   NM=w_m3b;          MARGS="--ft $M/m3_b_s13" ;;
  soup)   NM=soup;           MARGS="--soup $SOUP" ;;
  freeze) NM=w_m3_fr_s13;    MARGS="--ft $M/m3_fr_s13" ;;
  *) echo "$(date +%T) FAIL: family '$FAM' has no alpha extension (LoRA is not in this pool yet)" >> $L/drive15.log; exit 1 ;;
esac
# ---- (3) alpha 0.30 for the winning family; 0.35 only if 0.30 fails the eligibility (dev ar->en >= bge-m3 + 5)
ext() { a=$1; n=${NM}_a$a
  step merge_$n $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 $MARGS --alpha 0.$a --out $M/$n; need $H/$n/config.json
  step dev_$n $D -m lab.retrieval.v2.eval_v2 --model $M/$n --name $n --prefix none --splits dev devM --bs 32; need $R/$n.dev.json $R/$n.devM.json; }
ext 30
EL=$(python3 - ${NM}_a30 <<'PY'
import json, sys
g = lambda n: json.load(open(f"runs/retrieval/v2/results/{n}.dev.json"))["directions"]["ar->en"]["mrr@10"] * 100
print("yes" if g(sys.argv[1]) >= g("base_bge-m3") + 5.0 else "no")
PY
)
EXTN="${NM}_a30"
echo "$(date +%T) ${NM}_a30 eligible=$EL" >> $L/drive15.log
if [ "$EL" = "no" ]; then ext 35; EXTN="$EXTN ${NM}_a35"; fi
SPEC=""; for n in $EXTN; do SPEC="$SPEC $n=$M/$n:none"; done
step proxy4 $D -m lab.retrieval.v2.ood_dev_ar --models $SPEC; proxy_has $EXTN
step suite4 $D -m lab.retrieval.v2.suite_v2 --models $SPEC; for n in $EXTN; do need $R/suite/$n.json; done
step select4 env NO_GPU=1 $D -m lab.retrieval.v2.suite_select
echo "$(date +%T) INTERIM2 (pool without LoRA, with extension) in logs/select4.log" >> $L/drive15.log
# ---- (4) LoRA: gated self-test of the merge, then the identical rerun
step lora_selftest $D -m lab.retrieval.v2.lora_v2
grep -q "LORA SELFTEST OK" $L/lora_selftest.log || { echo "$(date +%T) FAIL: LoRA merge self-test" >> $L/drive15.log; exit 1; }
LORA="--base bge-m3 --data v2 --negs same:1 --epochs 1 --seed 13 --scale 50 --loss cached --mini-batch 16 --eval-bs 32"
step lo_lora $D -m lab.retrieval.v2.train_v2 --name m3_lo_s13 $LORA --lr 1e-4 --lora 32 $EXCL --keep-model
need $H/m3_lo_s13/config.json $R/m3_lo_s13.dev.json $R/m3_lo_s13.devM.json
LSPEC="m3_lo_s13=$M/m3_lo_s13:none"; LREQ="m3_lo_s13"
for a in 75 50; do n=w_m3_lo_s13_a$a
  step merge_$n $D -m lab.retrieval.v2.merge_v2 --base BAAI/bge-m3 --ft $M/m3_lo_s13 --alpha 0.$a --out $M/$n; need $H/$n/config.json
  step dev_$n $D -m lab.retrieval.v2.eval_v2 --model $M/$n --name $n --prefix none --splits dev devM --bs 32; need $R/$n.dev.json $R/$n.devM.json
  LSPEC="$LSPEC $n=$M/$n:none"; LREQ="$LREQ,$n"; done
step proxy5 $D -m lab.retrieval.v2.ood_dev_ar --models $LSPEC; proxy_has ${LREQ//,/ }
step suite5 $D -m lab.retrieval.v2.suite_v2 --models $LSPEC; for n in ${LREQ//,/ }; do need $R/suite/$n.json; done
# ---- (5) FINAL select on the full pool
REQ="bge-m3,m3_b_s13,soup_m3,wm_a50,wm_a75,w_m3b_a40,w_m3b_a60,soup_a40,soup_a50,soup_a60,soup_a75,m3_fr_s13,w_m3_fr_s13_a75,w_m3_fr_s13_a50,$LREQ,${EXTN// /,}"
step select5 env NO_GPU=1 $D -m lab.retrieval.v2.suite_select --write --require $REQ
need $R/suite_pick.json
echo DRIVE15_DONE >> $L/queues.done
