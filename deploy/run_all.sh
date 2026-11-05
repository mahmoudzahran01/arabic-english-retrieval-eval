#!/usr/bin/env bash
# Export, evaluate and benchmark every deployment variant of one model. Run on the Spark (CPU only), one step at a time:
# each step waits for MemAvailable >= 25 GiB first (deploy.common.wait_for_memory), and no step exceeds ~6 GB RSS.
#
#   source deploy/spark_env.sh
#   bash deploy/run_all.sh v2-small "$V2_MODELS/f5_s_kd_s13" e5 quality
#   bash deploy/run_all.sh v2-base  "$V2_MODELS/f5_b_kd_s13" e5 quality
#   bash deploy/run_all.sh v2-small "$V2_MODELS/f5_s_kd_s13" e5 bench     # run the benches alone on the machine
#
# args: NAME  MODEL_DIR  PREFIX(e5|none)  STAGE(quality|bench|all)
# The benchmarks measure latency, so run them when nothing else of yours is running; quality runs may overlap.
set -euo pipefail
NAME="$1"; MODEL="$2"; PREFIX="${3:-e5}"; STAGE="${4:-all}"
OUT="${ART:?source deploy/spark_env.sh first}/$NAME"
RES="${RES:?}"
LOG="$HOME/projects/v2-deploy/logs"
mkdir -p "$LOG"

if [ ! -f "$OUT/w8/model.onnx" ]; then
  python -m deploy.export_onnx --model "$MODEL" --out "$OUT" --variants int8 int8-mm int8-pc int8-pc-mm w8 2>&1 | grep -E '^\[export\]|Error' | tee -a "$LOG/export_$NAME.log"
fi

if [ "$STAGE" = quality ] || [ "$STAGE" = all ]; then
for V in torch-fp32 onnx-fp32 onnx-int8 onnx-int8-mm onnx-int8-pc onnx-int8-pc-mm onnx-w8; do
  [ -f "$RES/quality/${NAME}__${V}.test.json" ] || \
    python -m deploy.eval_quality --name "$NAME" --variant "$V" --model "$MODEL" --onnx-root "$OUT" --prefix "$PREFIX" --out "$RES/quality" 2>&1 | grep -E '^\[' | tee -a "$LOG/quality_$NAME.log"
done
# quantised queries against an fp32-encoded index (what the demo does); the fp32 corpus embeddings are cached by the fp32 run above
for V in onnx-w8 onnx-int8-pc onnx-int8; do
  [ -f "$RES/quality/${NAME}__${V}+q-only.test.json" ] || \
    python -m deploy.eval_quality --name "$NAME" --variant "$V" --corpus-variant onnx-fp32 --model "$MODEL" --onnx-root "$OUT" --prefix "$PREFIX" --out "$RES/quality" 2>&1 | grep -E '^\[' | tee -a "$LOG/quality_$NAME.log"
done
fi

if [ "$STAGE" = bench ] || [ "$STAGE" = all ]; then
for T in 2 8; do
  for V in torch-fp32 onnx-fp32 onnx-int8 onnx-int8-pc onnx-w8; do
    [ -f "$RES/bench/${NAME}__${V}__t${T}.bench.json" ] || \
      python -m deploy.bench --name "$NAME" --variant "$V" --model "$MODEL" --onnx-root "$OUT" --prefix "$PREFIX" --threads "$T" --out "$RES/bench" 2>&1 | grep -E '^\[' | tee -a "$LOG/bench_$NAME.log"
  done
done
fi
echo "[run_all] $NAME $STAGE done"
