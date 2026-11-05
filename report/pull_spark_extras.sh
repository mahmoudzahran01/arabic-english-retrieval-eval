#!/usr/bin/env bash
# The few inputs the v2-embedder snapshot does not carry, read-only from the Spark:
#   models/<name>/{lab_retrieval.json,merge.json}  training arguments and merge recipes of the final models
#   exact_overlap.json                             exact-copy and flagged counts of MIRACL-HN ar dev against the TyDi-ar / MIRACL-ar training questions
#   mteb_subsets.json                              per-subset MTEB scores (cross-lingual vs Arabic-only) of all scored models, baselines included
set -euo pipefail
HOST="${SPARK_HOST:-mzahran1@100.76.25.77}"
LAB="projects/qubit-ai-lab/runs/retrieval/v2"
cd "$(dirname "$0")/.."
DST=results/retrieval_v2
for m in final_m3 w_m3b_a40 final_base final_small st_b_s13 st_s_s13 m3_b_s13 m3_b_s16 soup_m3; do
  mkdir -p "$DST/models/$m"
  rsync -a "$HOST:$LAB/models/$m/lab_retrieval.json" "$HOST:$LAB/models/$m/merge.json" "$DST/models/$m/" 2>/dev/null || true
done
ssh "$HOST" "python3 - projects/mteb-arabic/results_cache/results projects/mteb-arabic/results_cache_v2/results" < report/extract_mteb_subsets.py > "$DST/mteb_subsets.json.tmp" && mv "$DST/mteb_subsets.json.tmp" "$DST/mteb_subsets.json"
rsync -a "$HOST:$LAB/results/exact_overlap.json" "$DST/exact_overlap.json"
