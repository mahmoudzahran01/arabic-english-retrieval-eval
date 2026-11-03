#!/usr/bin/env bash
# Shared helpers for the stage scripts. Source it; do not execute it.
#   PYTHON         interpreter to use (default: python)
#   ARR_RUNS_DIR   where large outputs go (default: <repo>/runs)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"
RUNS="${ARR_RUNS_DIR:-$ROOT/runs}"
RES="$RUNS/retrieval/results"        # one json per evaluated system
MODELS="$RUNS/retrieval/models"      # fine-tuned weights
LOGS="$RUNS/retrieval/logs"
mkdir -p "$RES" "$LOGS"

# retry <done-file> <log-file> <command...>
# Runs the command until <done-file> exists (max 3 attempts, 60 s apart); skips it if the file is already there.
# Training / evaluation steps are idempotent, so a stage can be re-launched after a crash (e.g. CUDA out-of-memory on a
# shared GPU) and only the missing items run.
retry() {
  local done_file="$1" log="$2"; shift 2
  if [ -f "$done_file" ]; then echo "skip   $(basename "$done_file")"; return 0; fi
  local attempt
  for attempt in 1 2 3; do
    echo "run    $* (attempt $attempt, $(date +%T))"
    "$@" > "$log" 2>&1
    [ -f "$done_file" ] && return 0
    echo "FAILED (attempt $attempt); last log lines:"; tail -n 3 "$log" | cut -c1-200
    sleep 60
  done
  return 1
}

# Result-file name of a pipeline system ("first-stage>>reranker" is stored as "first-stage__reranker").
fname() { echo "${1//>>/__}"; }
