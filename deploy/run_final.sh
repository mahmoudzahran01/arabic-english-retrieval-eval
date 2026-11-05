#!/usr/bin/env bash
# Second deployment round, on the FINAL models (v2-small = final_small, v2-base = final_base), CPU only, one job at a time.
# Results go next to the first round, not over it: ~/projects/v2-deploy/{artifacts_final,results_final}.
# Quality runs are niced (a GPU training round may be running); the latency benchmarks run last, at normal priority.
#   bash deploy/run_final.sh              # quality for both models, then the benchmarks
set -euo pipefail
export ART="$HOME/projects/v2-deploy/artifacts_final" RES="$HOME/projects/v2-deploy/results_final"
source "$(dirname "$0")/spark_env.sh"
mkdir -p "$ART" "$RES" "$HOME/projects/v2-deploy/logs"
for spec in "v2-small:final_small" "v2-base:final_base"; do
  nice -n 10 bash deploy/run_all.sh "${spec%%:*}" "$V2_MODELS/${spec##*:}" e5 quality
done
for spec in "v2-small:final_small" "v2-base:final_base"; do
  bash deploy/run_all.sh "${spec%%:*}" "$V2_MODELS/${spec##*:}" e5 bench
done
echo "[run_final] done"
