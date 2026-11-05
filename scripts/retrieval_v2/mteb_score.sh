#!/usr/bin/env bash
# mteb_score.sh MODEL_DIR_NAME RESULT_NAME [BASE_ID] [TASKS_CSV]   - score a local v2 checkpoint (runs/retrieval/v2/models/MODEL_DIR_NAME) with score_local.py inside the mteb-arabic container
dir=$1; name=$2; base=${3:-BAAI/bge-m3}; tasks=${4:-}
cd $HOME/projects/mteb-arabic
EXTRA_DOCKER_ARGS="-v $HOME/projects/qubit-ai-lab/runs/retrieval/v2/models:/v2models:ro -v $HOME/projects/qubit-ai-lab/runs/retrieval/v2/scripts:/myv2:ro" RESULT_CACHE=/work/results_cache_v2 \
  ./run.sh python /myv2/score_local.py /v2models/$dir $name 32 $base $tasks >> logs/eval_v2_$name.log 2>&1
