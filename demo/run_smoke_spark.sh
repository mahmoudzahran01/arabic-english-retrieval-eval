#!/usr/bin/env bash
# Smoke test of the demo on the Spark CPU (2 threads, like a free Space). Source deploy/spark_env.sh first.
#   DEMO_INDEX     folder with passages.parquet, examples.json, emb_*.npy   (default ~/projects/v2-deploy/demo_index)
#   V2_ONNX_DIR    folder with <variant>/model.onnx, tokenizer.json, config.json
#   V2_ONNX_VARIANT  int8 | int8-pc | fp32
# HF_HOME points at the cache that already holds BAAI/bge-m3 and the v1 model (downloaded by build_index).
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/projects/v2-deploy/hf_cache}"
export DEMO_INDEX="${DEMO_INDEX:-$HOME/projects/v2-deploy/demo_index}"
export V2_ONNX_DIR="${V2_ONNX_DIR:?set V2_ONNX_DIR}"
export V2_ONNX_VARIANT="${V2_ONNX_VARIANT:-int8-pc}"
export OMP_NUM_THREADS=2 GRADIO_ANALYTICS_ENABLED=False
python - <<PY
from deploy.common import wait_for_memory
wait_for_memory(15.0)
PY
python demo/smoke_test.py --out "${1:-demo/smoke_test.md}" --threads 2
