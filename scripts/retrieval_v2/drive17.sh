#!/usr/bin/env bash
# Stage 17 (coordinator-approved correction of the drive16 step "guard_split"): the cache skipped 5 models whose JSONs came from the FIRST (untitled) guard-split run.
# Move them to stale_untitled/, rerun them with titles (bge-m3 FIRST as the sanity check: MIRACL-HN must come out ~78.9 = the mteb number), then the other 4.
cd ~/projects/qubit-ai-lab
D=runs/retrieval/v2/scripts/dk.sh; L=runs/retrieval/v2/logs; G=runs/retrieval/v2/scripts/gate.sh; M=/lab/runs/retrieval/v2/models; R=runs/retrieval/v2/results; P=runs/retrieval/v2/PROGRESS.md
step() { name=$1; shift; echo "$(date +%T) START $name" >> $L/drive17.log; $G "$@" > $L/$name.log 2>&1; echo "$(date +%T) END $name" >> $L/drive17.log; }
mkdir -p $R/guard_split/stale_untitled
for n in bge-m3 e5-base m3_b_s13 wm_a75 v1; do [ -f $R/guard_split/$n.json ] && mv $R/guard_split/$n.json $R/guard_split/stale_untitled/$n.json; done
echo "- $(date +%H:%M) CORRECTION to the drive16 step guard_split (coordinator-approved): its cache skipped bge-m3, e5-base, m3_b_s13, wm_a75, v1 because their JSONs came from the FIRST guard-split run, which had no passage titles (mixed conditions in one table). The 5 files were moved to results/guard_split/stale_untitled/ (kept) and the 5 models are recomputed WITH titles in one gated job each (bge-m3 first as the sanity check against mteb 78.9)." >> $P
step gs_bge $D -m lab.retrieval.v2.guard_split --models bge-m3=BAAI/bge-m3:none
[ -s $R/guard_split/bge-m3.json ] || { echo "$(date +%T) FAIL: bge-m3.json missing" >> $L/drive17.log; exit 1; }
V=$(grep -o "MIRACL-HN nDCG@10 [0-9.]*" $L/gs_bge.log | tail -1 | awk '{print $3}')
echo "$(date +%T) bge-m3 titled MIRACL-HN nDCG@10 = $V (mteb: 78.9)" >> $L/drive17.log
python3 -c "import sys; v=float('$V'); sys.exit(0 if 78.4 <= v <= 79.4 else 1)" || { echo "$(date +%T) STOP: sanity check failed ($V not in 78.4..79.4)" >> $L/drive17.log; echo "- $(date +%H:%M) guard_split sanity check FAILED: titled bge-m3 MIRACL-HN = $V (mteb 78.9). Stopped; remaining 4 models not run." >> $P; exit 1; }
echo "- $(date +%H:%M) guard_split sanity check passed: titled bge-m3 MIRACL-HN nDCG@10 = $V (mteb 78.9)." >> $P
step gs_rest $D -m lab.retrieval.v2.guard_split --models e5-base=intfloat/multilingual-e5-base:e5 m3_b_s13=$M/m3_b_s13:none wm_a75=$M/wm_a75:none v1=/lab/runs/retrieval/models/e5s_hard1_xl_e1_s13:e5
for n in bge-m3 e5-base m3_b_s13 wm_a75 v1 final_m3 e5-small final_base final_small; do [ -s $R/guard_split/$n.json ] || { echo "$(date +%T) FAIL: $n.json missing" >> $L/drive17.log; exit 1; }; done
step gs_report env NO_GPU=1 $D -m lab.retrieval.v2.guard_split --report
echo DRIVE17_DONE >> $L/queues.done
