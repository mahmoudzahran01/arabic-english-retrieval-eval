# Source this on the Spark: CPU-only, pinned threads, paths for the deploy scripts.
cd "$HOME/projects/v2-deploy/repo"
source "$HOME/projects/v2-deploy/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=""            # never touch the GPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="$PWD/src:$PWD"
export ARR_RUNS_DIR="$HOME/projects/v2-deploy/runs"
export V2_MODELS="$HOME/projects/qubit-ai-lab/runs/retrieval/v2/models"   # read in place, never modified
export ART="${ART:-$HOME/projects/v2-deploy/artifacts}"       # preset ART / RES to keep a second round apart (artifacts_final, results_final)
export RES="${RES:-$HOME/projects/v2-deploy/results}"
