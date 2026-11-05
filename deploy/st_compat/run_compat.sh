#!/usr/bin/env bash
# Sentence-transformers 5.x compatible copies of the three final models, plus the check that they embed like the originals. Spark, CPU only.
#   final_m3 -> upload/v2-m3, final_base -> upload/v2-base, final_small -> upload/v2-small   (under runs/retrieval/v2/, model cards copied)
# ST 6.2 side: the lab docker image through mteb-arabic/run.sh with the GPU hidden (CUDA_VISIBLE_DEVICES empty, device="cpu").
# ST 5.3 side: the venv of runs/st5test/v5. Each job waits for MemAvailable >= 25 GB. Run from anywhere:  bash run_compat.sh
set -euo pipefail
LAB="$HOME/projects/qubit-ai-lab/runs/retrieval/v2"
WORK="$HOME/projects/v2-deploy/st_compat"
V5="$HOME/projects/qubit-ai-lab/runs/st5test/v5/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="v2-m3:final_m3 v2-base:final_base v2-small:final_small"
mkdir -p "$WORK/out" "$LAB/upload"
cp "$HERE"/build_compat.py "$HERE"/check_compat.py "$HERE"/compare_compat.py "$WORK/"

memgate() { until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 25 ]; do echo "[memgate] MemAvailable < 25 GB, waiting" >&2; sleep 60; done; }
st6() {  # label, then NAME=/v2/... specs
  memgate
  NO_GPU=1 CUDA_LIBS_ONLY=1 EXTRA_DOCKER_ARGS="-v $WORK:/compat -v $LAB:/v2:ro" "$HOME/projects/mteb-arabic/run.sh" \
    python /compat/check_compat.py --label st6 --out /compat/out "$@"
}
field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }

specs=(); for m in $MODELS; do specs+=("${m%%:*}.orig=/v2/models/${m##*:}"); done
st6 "${specs[@]}"                                                    # 1. what ST 6.2 sees in the originals

for m in $MODELS; do                                                 # 2. rebuild with the ST 5.3 classes
  up="${m%%:*}"; src="$LAB/models/${m##*:}"; j="$WORK/out/st6__$up.orig.json"
  memgate
  "$V5" "$WORK/build_compat.py" --src "$src" --dst "$LAB/upload/$up" \
    --pooling "$(field "$j" pooling)" --max-seq-length "$(field "$j" max_seq_length)"
done

specs5=(); specs6=()                                                 # 3. ST 5.3 on the originals (expected to fail) and on the compat copies; ST 6.2 on the compat copies
for m in $MODELS; do up="${m%%:*}"; specs5+=("$up.orig=$LAB/models/${m##*:}" "$up.compat=$LAB/upload/$up"); specs6+=("$up.compat=/v2/upload/$up"); done
memgate; "$V5" "$WORK/check_compat.py" --label st5 --out "$WORK/out" "${specs5[@]}"
st6 "${specs6[@]}"

"$V5" "$WORK/compare_compat.py" "$WORK/out" v2-m3 v2-base v2-small --json "$WORK/out/compat_check.json" | tee "$WORK/out/compat_check.md"
