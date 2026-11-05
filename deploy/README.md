# Deployment: ONNX export, int8 quantisation, CPU benchmark

Export of the v2 retrievers to ONNX (fp32 and dynamic int8), and what each variant costs and saves on a CPU: size on disk, latency (p50 / p95 at batch 1 and 32),
throughput, peak memory, and the **quality cost** measured as MRR@10 on the published TEST split against the PyTorch fp32 model.

Everything here is CPU-only and takes the model folder as an argument, so it applies to any later candidate (`--model PATH`).
Large artefacts (ONNX files, embeddings) are **not in git**: they live on the machine that ran the benchmark (see "Where things are").

## Where things are

| what | where |
|---|---|
| code | this folder (`export_onnx.py`, `eval_quality.py`, `bench.py`, `summarize.py`, `serve.py`, `run_all.sh`) |
| raw results (json + per-query arrays) | `deploy/results/{quality,bench,export}/` (committed, small) |
| ONNX files | Spark: `~/projects/v2-deploy/artifacts/<model>/{fp32,int8,int8-mm,int8-pc,int8-pc-mm,w8}/model.onnx` (+ `tokenizer.json`, `config.json`, `export_meta.json`) |
| source models (read in place, never modified) | Spark: `~/projects/qubit-ai-lab/runs/retrieval/v2/models/{f5_b_kd_s13 (v2-base), f5_s_kd_s13 (v2-small), wm_a75 (v2-m3)}` |
| benchmark data | the v1 benchmark parquet files (`runs/retrieval/data`): test split, 9,817 passages and 6,823 queries |

## Method

* **Export**: `torch.onnx.export` (legacy exporter, opset 17, dynamic batch and sequence axes) of the encoder **plus pooling and L2 normalisation**, so the graph maps
  `(input_ids, attention_mask)` to the unit-length sentence embedding and a deployment needs only `onnxruntime` + `tokenizers`.
  optimum was not used: its ONNX exporter pinned `transformers<5` when this was written. The export script checks the fp32 graph against sentence-transformers
  on AR/EN strings including a 512-token input (minimum cosine >= 0.99999).
* **Quantisation**: `onnxruntime.quantization.quantize_dynamic`, int8 weights, activations quantised on the fly. Four variants: per-tensor or per-channel weight scales, with the
  embedding table (`Gather`) quantised or left in fp32 (`-mm` = MatMul weights only). Plus `w8`: weight-only block-wise 8-bit quantisation of the MatMul weights
  (`MatMulNBits`, block size 64, symmetric) with an int8 embedding table; the activations stay fp32, which avoids the activation-outlier error of dynamic int8.
* **Quality**: `eval_quality.py` runs the repository's evaluation protocol (per-slice MRR@10 over the split's own pools, direction macros) on the CPU, max length 512. The reference is
  PyTorch fp32 on CPU, which reproduces the GPU fp16 numbers of the training runs to the second decimal (v2-small: 80.8 / 87.6 / 60.2 / 81.1). Deltas carry a 95% paired bootstrap
  over queries (2,000 resamples, within slices). `int8 queries vs fp32 index` encodes the corpus with the fp32 graph and the queries with int8 (the setup of `demo/`).
* **Latency**: `bench.py`, one process per variant (clean peak RSS), texts sampled from the test split, 20 / 3 / 5 / 2 warm-up calls and 200 / 20 / 60 / 8 timed calls for the
  query b=1 / query b=32 / passage b=1 / passage b=32 workloads. Latency is end-to-end per call (tokenisation + model + pooling); throughput = sentences / wall time of the timed calls.
  Passages are truncated at 512 tokens, queries are short.
* **Hardware**: the NVIDIA GB10 machine (20-core ARM CPU, 121 GB unified memory), **CPU only** (`CUDA_VISIBLE_DEVICES=""`), 2 threads (the CPU of a free Hugging Face Space)
  and 8 threads. That machine is shared with a training job, so the load average during the runs is stored in each benchmark json: treat latencies as indicative, not as x86 numbers.
  Quality numbers do not depend on the hardware.

## Results

<!-- RESULTS:START -->
**v2-base**: test MRR@10 x100 (CPU, max length 512). Delta = variant minus PyTorch fp32, with a 95% paired-bootstrap CI over queries.

| variant | ar->ar | en->en | ar->en | en->ar | macro | delta macro [95% CI] | delta ar->en [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 81.76 | 88.57 | 64.40 | 84.03 | 79.69 | reference | reference |
| ONNX fp32 | 81.76 | 88.57 | 64.40 | 84.03 | 79.69 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| ONNX int8 (dynamic) | 77.86 | 84.54 | 56.13 | 78.58 | 74.28 | -5.42 [-6.01, -4.86] | -8.27 [-9.82, -6.72] |
| ONNX int8, per-channel | 78.58 | 85.33 | 57.91 | 78.74 | 75.14 | -4.56 [-5.14, -4.00] | -6.50 [-8.05, -5.05] |
| ONNX weight-only 8-bit (MatMulNBits) | 81.59 | 88.54 | 64.23 | 83.97 | 79.58 | -0.11 [-0.23, +0.01] | -0.18 [-0.51, +0.13] |

**v2-base** (Spark, ARM server CPU), 8 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 1061 | 1114 / 2447 | 66.4 / 94.9 | 36 | 188 / 289 | 10545 / 11634 | 3.3 |
| ONNX fp32 | 1059 | 2114 / 3089 | 22.8 / 30.1 | 94 | 54 / 88 | 9093 / 10141 | 3.9 |
| ONNX int8 (dynamic) | 265 | 876 / 2708 | 8.6 / 10.8 | 262 | 20 / 35 | 3262 / 3774 | 10.8 |
| ONNX int8, per-channel | 266 | 905 / 2736 | 9.2 / 11.0 | 220 | 23 / 43 | 2937 / 4020 | 10.7 |
| ONNX weight-only 8-bit (MatMulNBits) | 270 | 901 / 2226 | 38.6 / 46.6 | 98 | 67 / 99 | 9556 / 10338 | 3.7 |

**v2-base** (Spark, ARM server CPU), 2 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 1061 | 1042 / 2384 | 48.1 / 59.6 | 35 | 138 / 241 | 20322 / 21923 | 1.8 |
| ONNX fp32 | 1059 | 2046 / 3021 | 31.8 / 45.0 | 34 | 125 / 237 | 20079 / 22433 | 1.8 |
| ONNX int8 (dynamic) | 265 | 836 / 2668 | 7.4 / 12.6 | 138 | 38 / 73 | 5766 / 7289 | 5.9 |
| ONNX int8, per-channel | 266 | 834 / 2665 | 9.0 / 12.2 | 156 | 36 / 68 | 4800 / 7059 | 7.1 |
| ONNX weight-only 8-bit (MatMulNBits) | 270 | 830 / 2155 | 96.6 / 109.2 | 25 | 188 / 285 | 38680 / 41164 | 0.9 |

**v2-small**: test MRR@10 x100 (CPU, max length 512). Delta = variant minus PyTorch fp32, with a 95% paired-bootstrap CI over queries.

| variant | ar->ar | en->en | ar->en | en->ar | macro | delta macro [95% CI] | delta ar->en [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 80.81 | 87.61 | 60.18 | 81.10 | 77.43 | reference | reference |
| ONNX fp32 | 80.81 | 87.61 | 60.18 | 81.10 | 77.43 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| ONNX int8 (dynamic) | 79.35 | 86.94 | 55.10 | 78.57 | 74.99 | -2.44 [-2.88, -1.98] | -5.09 [-6.37, -3.88] |
| ONNX int8, MatMul only | 79.34 | 87.01 | 55.06 | 77.67 | 74.77 | -2.66 [-3.10, -2.18] | -5.12 [-6.38, -3.81] |
| ONNX int8, per-channel | 80.36 | 87.50 | 58.50 | 80.48 | 76.71 | -0.72 [-1.02, -0.39] | -1.68 [-2.55, -0.83] |
| ONNX int8, per-channel, MatMul only | 80.38 | 87.58 | 58.17 | 80.70 | 76.71 | -0.72 [-1.06, -0.41] | -2.01 [-2.92, -1.09] |
| ONNX weight-only 8-bit (MatMulNBits) | 80.76 | 87.59 | 60.23 | 81.28 | 77.46 | +0.04 [-0.07, +0.15] | +0.05 [-0.26, +0.37] |
| int8 queries vs fp32 index | 79.78 | 87.37 | 57.02 | 79.40 | 75.89 | -1.54 [-1.90, -1.17] | -3.17 [-4.21, -2.11] |

**v2-small** (Spark, ARM server CPU), 8 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 449 | 1104 / 1818 | 11.9 / 21.6 | 213 | 18 / 47 | 2361 / 2691 | 14.4 |
| ONNX fp32 | 448 | 1147 / 2613 | 5.4 / 7.9 | 198 | 27 / 48 | 1701 / 2186 | 19.7 |
| ONNX int8 (dynamic) | 113 | 732 / 2249 | 3.1 / 4.3 | 580 | 12 / 20 | 1082 / 1234 | 34.1 |
| ONNX int8, per-channel | 113 | 740 / 2258 | 2.5 / 3.3 | 581 | 13 / 21 | 1041 / 1331 | 32.7 |
| ONNX weight-only 8-bit (MatMulNBits) | 114 | 745 / 2266 | 10.4 / 12.8 | 183 | 31 / 54 | 1454 / 2241 | 19.8 |

**v2-small** (Spark, ARM server CPU), 2 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 449 | 1045 / 1715 | 18.4 / 26.2 | 142 | 43 / 73 | 6529 / 7058 | 5.6 |
| ONNX fp32 | 448 | 1084 / 2550 | 6.2 / 8.8 | 145 | 33 / 63 | 7448 / 7783 | 5.0 |
| ONNX int8 (dynamic) | 113 | 682 / 2198 | 4.0 / 5.9 | 337 | 18 / 33 | 2375 / 3546 | 14.1 |
| ONNX int8, per-channel | 113 | 683 / 2200 | 4.4 / 6.2 | 344 | 17 / 33 | 3496 / 3857 | 10.6 |
| ONNX weight-only 8-bit (MatMulNBits) | 114 | 674 / 2194 | 18.3 / 21.4 | 115 | 49 / 82 | 4913 / 7749 | 5.7 |

**v2-small** (Apple M3 Pro laptop CPU, the two 8-bit variants of the small model only), 2 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX int8, per-channel | 113 | 865 / 1836 | 2.4 / 4.0 | 312 | 17 / 34 | 2559 / 2842 | 14.4 |
| ONNX weight-only 8-bit (MatMulNBits) | 114 | 785 / 1971 | 17.0 / 20.2 | 121 | 48 / 81 | 5179 / 5592 | 7.1 |

**v2-small** (Apple M3 Pro laptop CPU, the two 8-bit variants of the small model only), 8 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX int8, per-channel | 113 | 883 / 2017 | 2.5 / 4.0 | 602 | 11 / 19 | 1152 / 1354 | 31.5 |
| ONNX weight-only 8-bit (MatMulNBits) | 114 | 874 / 1901 | 10.9 / 14.2 | 244 | 26 / 42 | 2150 / 2429 | 16.8 |
<!-- RESULTS:END -->

### Findings

* **ONNX fp32 is exact.** Metrics are identical to PyTorch fp32 on the whole test split (delta 0.00, CI [0.00, 0.00]); PyTorch fp32 on the CPU also reproduces the GPU/fp16 numbers of the
  training runs to within 0.1 (v2-base: macro 79.7 here vs 79.75 there). Single-query latency drops from 18 to 6 ms (v2-small) and from 48 to 32 ms (v2-base) at 2 threads.
* **Dynamic int8 is not acceptable on these models.** It is the fastest variant (v2-base: 7 ms per query, about 2.8x the passage throughput of ONNX fp32 at 8 threads) and 4x smaller, but costs
  **-2.44 macro / -5.09 ar->en points on v2-small and -5.42 / -8.27 on v2-base**, far beyond the 1-point budget. Quantising only the MatMuls (leaving the embedding table in fp32) does not help,
  so the loss comes from the transformer layers (activation outliers). Per-channel weight scales recover part of it (-0.72 on v2-small, -4.56 on v2-base): still too much for the base model.
  Do not ship dynamic int8 without calibration (static quantisation) or quantisation-aware training; neither was attempted.
* **Weight-only 8-bit (`w8`) is lossless but slow.** +0.04 (v2-small) and -0.11 (v2-base, CI [-0.23, +0.01]) macro points, 4x smaller (114 and 270 MiB) and about 2x less resident memory after
  loading; but ONNX Runtime's `MatMulNBits` CPU kernel dequantises on the fly, so a query costs 18 ms (v2-small) and 97 ms (v2-base) at 2 threads, 3x slower than fp32.
* **Recommendation**: ONNX fp32 when latency matters (the demo uses it), `w8` when download size or memory matter more than latency, and per-channel int8 only for v2-small if a 0.7-point loss is acceptable.
  fp16 was not measured: it only pays off on a GPU (off-limits for this task) or Apple's MPS (the Mac had about 3 GB of free disk when the work started, so no model copies were put there).
* **v2-m3 (568M) was not exported or benchmarked.** Exporting and quantising its 2.2 GB weights needs more than the 6 GB per process allowed on the shared Spark (the 278M model already peaked at
  6.8 GB resident during quantisation, 6.05 GB for `w8`), and scoring it on this CPU would take hours. Everything takes the model path as an argument, so it is one command each once a machine allows it.
* **Mac numbers** (last two tables, Apple M3 Pro, small model, 8-bit variants only) are given for orientation: dynamic per-channel int8 answers a query in 2.4 ms, the `w8` graph in 17 ms.
* **Peak RSS** is the peak of the whole benchmark process (python, pandas and the sampled texts, plus the batch-32 x 512-token activations); "RSS after load" is the same process right after loading the model.
* **Latency caveats**: shared 20-core ARM server (load average stored in every json, typically 6 to 12 during the runs), single run per cell, not an x86 CPU; tokenisation is included.

## Sentence-transformers 5.x compatibility folders

Models saved with sentence-transformers 6 write `sentence_transformers.base.*` module paths into `modules.json`, and sentence-transformers 5.x cannot import them
(`ModuleNotFoundError: No module named 'sentence_transformers.base'`). `st_compat/` rebuilds each final model as a folder that loads on both: the weights and the tokenizer are plain
Hugging Face files, so only the three modules (Transformer, Pooling, Normalize) are re-created with the 5.x classes and saved again (`build_compat.py`, run with sentence-transformers 5.3).
The Spark folders are `runs/retrieval/v2/upload/{v2-m3,v2-base,v2-small}` (from `final_m3`, `final_base`, `final_small`; the model cards are copied unchanged). The `model.safetensors` files are
byte-identical to the originals.

`check_compat.py` loads a folder with whatever sentence-transformers is installed (CPU only) and embeds 13 fixed texts: Arabic and English queries and passages with and without
prefixes, diacritics, numerals, an emoji, code, and two inputs far beyond 256 tokens. `compare_compat.py` compares the two sides: the originals under sentence-transformers 6.2 (the lab docker image, GPU hidden)
against the compatibility folders under 5.3. Criteria: the 5.3 folder loads, the largest absolute difference of any embedding component is below 1e-5, and `max_seq_length` and the pooling mode are equal.
Run `bash deploy/st_compat/run_compat.sh` on the Spark; the raw output is in `deploy/results/st_compat/`.

| model | ST 6.2 loads original | ST 5.3 loads original | ST 5.3 loads compat | ST 6.2 loads compat | max_seq_length (6 orig / 5 compat) | pooling (6 orig / 5 compat) | max abs diff (6 orig vs 5 compat) | min cosine | check |
|---|---|---|---|---|---|---|---|---|---|
| v2-m3 | yes | no | yes | yes | 256 / 256 | cls / cls | 4.25e-07 | 0.99999982 | PASS |
| v2-base | yes | no | yes | yes | 256 / 256 | mean / mean | 7.60e-07 | 0.99999988 | PASS |
| v2-small | yes | no | yes | yes | 256 / 256 | mean / mean | 3.07e-07 | 0.99999988 | PASS |

Loading the compatibility folders under 6.2 as well gives embeddings identical to the originals (largest difference 0.0 for all three).

## Reproduce

```bash
# on the machine that has the models (Spark): code only is synced from the Mac checkout
bash deploy/sync_to_spark.sh                      # rsync src/ deploy/ demo/ to ~/projects/v2-deploy/repo
source deploy/spark_env.sh                        # CPU only, 8 threads, paths
bash deploy/run_all.sh v2-small "$V2_MODELS/f5_s_kd_s13" e5 quality
bash deploy/run_all.sh v2-base  "$V2_MODELS/f5_b_kd_s13" e5 quality
# latency: run alone on the machine (nothing else of yours running)
bash deploy/run_all.sh v2-small "$V2_MODELS/f5_s_kd_s13" e5 bench
bash deploy/run_all.sh v2-base  "$V2_MODELS/f5_b_kd_s13" e5 bench
# v2-m3 (bge-m3 family: CLS pooling, no prefixes)
bash deploy/run_all.sh v2-m3 "$V2_MODELS/wm_a75" none quality

# back on the Mac: copy the small result files, rebuild the tables
bash deploy/pull_results.sh
python -m deploy.summarize --write                # refreshes the table above
python report/make_tables.py                      # the report tables
```

Single steps: `python -m deploy.export_onnx --model DIR --out OUT`, `python -m deploy.eval_quality --name NAME --variant onnx-int8 --model DIR --onnx-root OUT --out RES`,
`python -m deploy.bench --name NAME --variant onnx-int8 --model DIR --onnx-root OUT --threads 2 --out RES`. Every step first waits for `MemAvailable >= 15 GiB`
(`deploy/common.py: wait_for_memory`) because the Spark is shared; nothing here exceeds about 6 GB resident.

## Serving

`serve.py` is a small FastAPI server over any exported variant with a text-embeddings-inference-style `POST /embed` route (see the docstring):

```bash
ONNX_ROOT=artifacts/v2-base VARIANT=int8 THREADS=2 python -m uvicorn deploy.serve:app --port 8080
```

Not tested here: serving the sentence-transformers folder with TEI itself.
