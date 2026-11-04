#!/usr/bin/env bash
# Copy the code (not data, not models) to the Spark work dir and make sure the data link exists.
set -euo pipefail
HOST="${SPARK_HOST:-mzahran1@100.76.25.77}"
DEST="${SPARK_DIR:-projects/v2-deploy}"
cd "$(dirname "$0")/.."
rsync -a --delete --exclude '__pycache__' --exclude 'results' --exclude 'artifacts' --exclude 'retrieval/v2' \
  src deploy demo "$HOST:$DEST/repo/"
ssh "$HOST" "mkdir -p $DEST/runs/retrieval $DEST/results $DEST/logs $DEST/artifacts && \
  ln -sfn \$HOME/projects/qubit-ai-lab/runs/retrieval/data $DEST/runs/retrieval/data"
