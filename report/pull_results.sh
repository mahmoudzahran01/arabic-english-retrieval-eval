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
# ablations behind the scaling curve and the negative-results table (dev json + training-set sizes), and the later runs (freeze, LoRA, alpha extension, soups)
ABL="p1_r0 p1_oth1 p1_same2 p1_bplang p1_sym p1_kd1 p1_kd3 p1_kdonly p1_base_r0 p2_xl2_a5k p2_s_f25 p2_s_f50 p2_s_full p2_s_e2 p2_s_kdxl p2_b_full f4_s_e1_s13 f4_b_e1_s13 f5_s_kd_s13 f5_b_kd_s13"
for n in $ABL; do
  rsync -a "$HOST:$LAB/results/$n.dev.json" "$HOST:$LAB/results/$n.train.json" "$DST/" 2>/dev/null || true
done
LATE="m3_fr_s13 w_m3_fr_s13_a75 w_m3_fr_s13_a50 m3_lo_s13 w_m3_lo_s13_a75 w_m3_lo_s13_a50 w_m3b_a30 w_m3b_a35 w_m3b_a40 w_m3b_a60 soup_m3 soup_a30 soup_a35 soup_a40 soup_a50 soup_a60 soup_a75 w_m3_fr_s13_a30 w_m3_fr_s13_a35"
for n in $LATE; do
  rsync -a "$HOST:$LAB/results/$n.dev.json" "$HOST:$LAB/results/$n.devM.json" "$HOST:$LAB/results/$n.train.json" "$DST/" 2>/dev/null || true
done
# optional files written by the v2 agent at the end (absent until then): WiSE curve, paired MTEB(ara) statistics, the final write-up, mMARCO proxy, selection suite
for f in alpha_curve.json mteb_paired.json ood_dev_ar.json; do rsync -a "$HOST:$LAB/results/$f" "$DST/" 2>/dev/null || true; done
rsync -a "$HOST:$LAB/RESULTS_v2.md" "$DST/" 2>/dev/null || true
rsync -a "$HOST:$LAB/results/suite/" "$DST/suite/" 2>/dev/null || true
# confirmatory seeds (16, 17) trained after the recipe was frozen, whatever the recipe is called
ssh "$HOST" "cd $LAB/results && ls | grep -E '_s1[67]\.(test|dev)\.(json|perq\.npz)$|_s1[67]\.test\.mixed\.json$'" 2>/dev/null | while read -r f; do rsync -a "$HOST:$LAB/results/$f" "$DST/"; done || true
# per-subset MTEB scores (cross-lingual vs Arabic-only) from the raw result caches: read-only, the extractor runs on the Spark
ssh "$HOST" "python3 - projects/mteb-arabic/results_cache/results projects/mteb-arabic/results_cache_v2/results" < report/extract_mteb_subsets.py > "$DST/mteb_subsets.json" 2>/dev/null || rm -f "$DST/mteb_subsets.json"
# the pre-registration trail is extracted from the lab notebook (read-only copy)
TMP="$(mktemp)"; rsync -a "$HOST:$LAB/PROGRESS.md" "$TMP" && python3 report/extract_prereg.py "$TMP" "$DST/data/prereg_trail.json"; rm -f "$TMP"

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
