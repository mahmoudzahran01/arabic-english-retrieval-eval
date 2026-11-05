#!/usr/bin/env bash
# Copy the small result files (json + per-query arrays; no models) from the Spark work dir into deploy/results/.
set -euo pipefail
HOST="${SPARK_HOST:-mzahran1@100.76.25.77}"
SRC="projects/v2-deploy"
RES_DIR="${RES_DIR:-results}"      # results_final for the round on the final models
ART_DIR="${ART_DIR:-artifacts}"    # artifacts_final
cd "$(dirname "$0")"
mkdir -p results/quality results/bench results/export
rsync -a "$HOST:$SRC/$RES_DIR/quality/" results/quality/
rsync -a "$HOST:$SRC/$RES_DIR/bench/" results/bench/ 2>/dev/null || true  # absent until the benchmarks have run
for m in $(ssh "$HOST" "ls $SRC/$ART_DIR"); do
  rsync -a "$HOST:$SRC/$ART_DIR/$m/export_meta.json" "results/export/$m.json" 2>/dev/null || true
done
