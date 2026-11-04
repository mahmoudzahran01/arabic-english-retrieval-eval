#!/usr/bin/env bash
# Copy the small result files (json + per-query arrays; no models) from the Spark work dir into deploy/results/.
set -euo pipefail
HOST="${SPARK_HOST:-mzahran1@100.76.25.77}"
SRC="projects/v2-deploy"
cd "$(dirname "$0")"
mkdir -p results/quality results/bench results/export
rsync -a "$HOST:$SRC/results/quality/" results/quality/
rsync -a "$HOST:$SRC/results/bench/" results/bench/ 2>/dev/null || true  # absent until the benchmarks have run
for m in $(ssh "$HOST" "ls $SRC/artifacts"); do
  rsync -a "$HOST:$SRC/artifacts/$m/export_meta.json" "results/export/$m.json" 2>/dev/null || true
done
