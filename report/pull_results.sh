#!/usr/bin/env bash
# Refresh the committed v2 result snapshot from the Spark (read-only: rsync, nothing on the Spark is modified).
#   results/retrieval_v2/   per-system JSONs + per-query MRR arrays of the v2 runs (TEST for the final candidates, DEV for the ablations)
#   results/mteb_ara/       MTEB(ara, v1) leaderboard + the MTEB task JSONs of the guard tasks
set -euo pipefail
HOST="${SPARK_HOST:-mzahran1@100.76.25.77}"
LAB="projects/qubit-ai-lab/runs/retrieval/v2"
MTEB="projects/mteb-arabic"
cd "$(dirname "$0")/.."
DST=results/retrieval_v2
mkdir -p "$DST/models" "$DST/data" "$DST/guard_split" results/mteb_ara/guards

# test results (json + per-query arrays + mixed-index diagnostic) of the systems used in the report
TEST="base_e5-small base_bge-m3 v1_s13 m3_a_s13 m3_b_s13 m3_b_s14 m3_b_s15 wm_a75 f4_b_e1_s13 f4_s_e1_s13 f5_b_kd_s13 f5_s_kd_s13 p3_b_e1_s13 p3_s_e1_s13"
for n in $TEST; do
  rsync -a "$HOST:$LAB/results/$n.test.json" "$HOST:$LAB/results/$n.test.perq.npz" "$DST/"
  rsync -a "$HOST:$LAB/results/$n.test.mixed.json" "$DST/" 2>/dev/null || true
done
# dev results used for the ablation / selection tables
DEV="v1_s13 p1_r0 f4_b_e1_s13 f5_b_kd_s13 f5_b_kd3_s13 f5_b_kdonly_s13 m3_b_s13 wm_a50 wm_a75 w_m3_p_s13_a50 w_m3_p_s13_a75 w_m3_r_s13_a50 w_m3_r_s13_a75 m3_p_s13 m3_r_s13 base_bge-m3 base_e5-small"
for n in $DEV; do
  rsync -a "$HOST:$LAB/results/$n.dev.json" "$DST/" 2>/dev/null || true
  rsync -a "$HOST:$LAB/results/$n.devM.json" "$DST/" 2>/dev/null || true
done
rsync -a "$HOST:$LAB/results/bge-m3.devM.json" "$DST/"
# data / recipe manifests (small) and the contamination split of the guard tasks
rsync -a "$HOST:projects/qubit-ai-lab/runs/retrieval/data/manifest.json" "$DST/data/manifest_v1.json"
for f in manifest_v2.json manifest_v3.json manifest_miracl.json; do rsync -a "$HOST:$LAB/data/$f" "$DST/data/"; done
ssh "$HOST" "python3 -c \"import json;d=json.load(open('\$HOME/$LAB/data/exclude_qids.json'));print(json.dumps(d['summary']))\"" > "$DST/data/exclude_summary.json"
for m in m3_b_s13 f5_b_kd_s13 f5_s_kd_s13; do
  mkdir -p "$DST/models/$m"; rsync -a "$HOST:$LAB/models/$m/lab_retrieval.json" "$DST/models/$m/"
done
mkdir -p "$DST/models/wm_a75"; rsync -a "$HOST:$LAB/models/wm_a75/merge.json" "$DST/models/wm_a75/"
rsync -a "$HOST:$LAB/results/guard_split/" "$DST/guard_split/"
rsync -a "$HOST:$LAB/results/audit_ood.json" "$HOST:$LAB/results/audit_ood_filtered.json" "$HOST:$LAB/results/diag_xquad_style.json" "$DST/data/"
# MTEB(ara, v1)
rsync -a "$HOST:$MTEB/leaderboard.json" "$HOST:$MTEB/LEADERBOARD.md" "$HOST:$MTEB/TASK_SELECTION.md" results/mteb_ara/
rsync -a --include='*/' --include='NamaaMrTydiReranking.json' --include='SemRel24STS.json' --include='MIRACLRetrievalHardNegatives.v2.json' --exclude='*' \
  "$HOST:$LAB/mteb/" results/mteb_ara/guards/
