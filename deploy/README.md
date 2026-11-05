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
| ONNX files | Spark: `~/projects/v2-deploy/artifacts_final/<model>/{fp32,int8,int8-mm,int8-pc,int8-pc-mm,w8}/model.onnx` (+ `tokenizer.json`, `config.json`, `export_meta.json`) |
| source models (read in place, never modified) | Spark: `~/projects/qubit-ai-lab/runs/retrieval/v2/models/{final_base (v2-base), final_small (v2-small), final_m3 (v2-m3)}` |
| first round (earlier student candidates) | `deploy/results/round1_candidates/` (models `f5_b_kd_s13`, `f5_s_kd_s13`; superseded by the final models) |
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
| PyTorch fp32 (reference) | 82.16 | 89.74 | 66.54 | 82.18 | 80.16 | reference | reference |
| ONNX fp32 | 82.16 | 89.74 | 66.54 | 82.18 | 80.16 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| ONNX int8 (dynamic) | 79.40 | 88.14 | 61.73 | 78.92 | 77.05 | -3.11 [-3.58, -2.63] | -4.81 [-6.05, -3.53] |
| ONNX int8, MatMul only | 79.75 | 88.28 | 60.68 | 78.65 | 76.84 | -3.32 [-3.78, -2.86] | -5.87 [-7.19, -4.58] |
| ONNX int8, per-channel | 80.63 | 88.91 | 62.17 | 79.27 | 77.74 | -2.41 [-2.85, -1.99] | -4.37 [-5.57, -3.21] |
| ONNX int8, per-channel, MatMul only | 80.49 | 88.45 | 62.02 | 79.19 | 77.54 | -2.62 [-3.09, -2.17] | -4.53 [-5.81, -3.27] |
| ONNX weight-only 8-bit (MatMulNBits) | 82.15 | 89.57 | 66.42 | 82.40 | 80.14 | -0.02 [-0.12, +0.08] | -0.12 [-0.38, +0.14] |
| int8 queries vs fp32 index | 81.33 | 89.46 | 63.61 | 81.81 | 79.05 | -1.10 [-1.45, -0.77] | -2.94 [-3.96, -1.94] |
| int8 per-channel queries vs fp32 index | 81.96 | 89.57 | 64.89 | 81.61 | 79.51 | -0.65 [-0.98, -0.32] | -1.66 [-2.59, -0.73] |
| w8 queries vs fp32 index | 82.13 | 89.74 | 66.36 | 82.31 | 80.14 | -0.02 [-0.11, +0.07] | -0.18 [-0.44, +0.08] |

**v2-base** (Spark, ARM server CPU), 8 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 1061 | 1100 / 2440 | 50.4 / 79.9 | 39 | 168 / 260 | 5867 / 8967 | 4.9 |
| ONNX fp32 | 1059 | 2118 / 3093 | 16.9 / 24.2 | 121 | 41 / 74 | 6168 / 6560 | 5.9 |
| ONNX int8 (dynamic) | 265 | 905 / 2305 | 5.9 / 8.2 | 240 | 26 / 51 | 2266 / 2548 | 16.1 |
| ONNX int8, per-channel | 266 | 901 / 2302 | 4.8 / 6.8 | 237 | 22 / 44 | 1402 / 1502 | 26.4 |
| ONNX weight-only 8-bit (MatMulNBits) | 270 | 899 / 2225 | 25.6 / 32.6 | 139 | 41 / 71 | 8339 / 9339 | 4.2 |

**v2-base** (Spark, ARM server CPU), 2 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 1061 | 1046 / 2330 | 67.7 / 87.8 | 27 | 166 / 274 | 25902 / 27204 | 1.4 |
| ONNX fp32 | 1059 | 2056 / 3030 | 49.4 / 69.4 | 28 | 159 / 280 | 28209 / 29552 | 1.3 |
| ONNX int8 (dynamic) | 265 | 839 / 2238 | 14.2 / 22.0 | 109 | 44 / 77 | 10618 / 10899 | 3.5 |
| ONNX int8, per-channel | 266 | 829 / 2228 | 13.6 / 19.0 | 108 | 50 / 86 | 10385 / 10860 | 3.6 |
| ONNX weight-only 8-bit (MatMulNBits) | 270 | 837 / 2160 | 86.1 / 100.8 | 27 | 185 / 296 | 33819 / 35624 | 1.1 |

**v2-small**: test MRR@10 x100 (CPU, max length 512). Delta = variant minus PyTorch fp32, with a 95% paired-bootstrap CI over queries.

| variant | ar->ar | en->en | ar->en | en->ar | macro | delta macro [95% CI] | delta ar->en [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 80.35 | 88.01 | 59.70 | 79.74 | 76.95 | reference | reference |
| ONNX fp32 | 80.35 | 88.01 | 59.70 | 79.74 | 76.95 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| ONNX int8 (dynamic) | 78.74 | 86.86 | 56.06 | 77.39 | 74.76 | -2.19 [-2.62, -1.75] | -3.64 [-4.81, -2.47] |
| ONNX int8, MatMul only | 79.21 | 87.06 | 55.75 | 77.15 | 74.79 | -2.16 [-2.60, -1.72] | -3.95 [-5.18, -2.78] |
| ONNX int8, per-channel | 79.95 | 87.59 | 58.21 | 79.47 | 76.30 | -0.65 [-0.96, -0.34] | -1.49 [-2.30, -0.64] |
| ONNX int8, per-channel, MatMul only | 79.92 | 87.70 | 59.00 | 78.98 | 76.40 | -0.55 [-0.87, -0.25] | -0.69 [-1.52, +0.07] |
| ONNX weight-only 8-bit (MatMulNBits) | 80.40 | 87.99 | 59.69 | 79.68 | 76.94 | -0.01 [-0.12, +0.10] | -0.01 [-0.36, +0.32] |
| int8 queries vs fp32 index | 79.77 | 87.63 | 57.51 | 78.30 | 75.80 | -1.15 [-1.54, -0.77] | -2.19 [-3.27, -1.11] |
| int8 per-channel queries vs fp32 index | 80.26 | 87.92 | 59.00 | 79.34 | 76.63 | -0.32 [-0.59, -0.06] | -0.70 [-1.48, +0.04] |
| w8 queries vs fp32 index | 80.37 | 87.96 | 60.03 | 79.87 | 77.05 | +0.10 [+0.01, +0.21] | +0.33 [+0.04, +0.63] |

**v2-small** (Spark, ARM server CPU), 8 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 449 | 1100 / 1813 | 18.8 / 36.0 | 131 | 50 / 106 | 2591 / 3277 | 12.7 |
| ONNX fp32 | 448 | 1134 / 2600 | 7.2 / 9.6 | 228 | 18 / 33 | 2885 / 2982 | 13.1 |
| ONNX int8 (dynamic) | 113 | 728 / 2245 | 3.3 / 4.3 | 724 | 11 / 21 | 1326 / 1518 | 26.6 |
| ONNX int8, per-channel | 113 | 738 / 2255 | 3.6 / 5.2 | 643 | 10 / 18 | 1453 / 1875 | 24.7 |
| ONNX weight-only 8-bit (MatMulNBits) | 114 | 749 / 2269 | 10.0 / 11.9 | 199 | 36 / 65 | 2761 / 2868 | 13.6 |

**v2-small** (Spark, ARM server CPU), 2 CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.

| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 449 | 1034 / 1715 | 20.7 / 31.1 | 132 | 49 / 75 | 8113 / 8514 | 4.6 |
| ONNX fp32 | 448 | 1088 / 2553 | 14.2 / 20.6 | 115 | 47 / 81 | 9627 / 10226 | 3.8 |
| ONNX int8 (dynamic) | 113 | 675 / 2193 | 4.7 / 6.5 | 319 | 22 / 39 | 4683 / 5074 | 8.0 |
| ONNX int8, per-channel | 113 | 677 / 2194 | 4.9 / 6.9 | 308 | 18 / 33 | 4712 / 5245 | 7.9 |
| ONNX weight-only 8-bit (MatMulNBits) | 114 | 680 / 2200 | 24.2 / 29.4 | 110 | 54 / 87 | 10167 / 10663 | 3.6 |
<!-- RESULTS:END -->

### Findings

Measured on the **final** v2-small and v2-base (the three-seed student soups). A first round on single-seed student candidates (`deploy/results/round1_candidates/`) ordered the variants the same way;
its dynamic-int8 losses were -2.44 (v2-small) and -5.42 (v2-base) macro points, so those deltas depend on the model and should be re-measured for any later one.

* **ONNX fp32 is exact.** Metrics are identical to PyTorch fp32 on the whole test split (delta 0.00, CI [0.00, 0.00]); PyTorch fp32 on the CPU also reproduces the GPU/fp16 numbers of the
  training runs to within 0.1 (v2-base: macro 80.16 here vs 80.2 in the report's test table; v2-small 76.95 vs 76.9). Single-query latency at 2 threads drops from 21 to 14 ms (v2-small) and from 68 to 49 ms (v2-base);
  that is a smaller gain than in the first round (18 to 6 ms and 48 to 32 ms), because the machine was busier (see the caveats).
* **Dynamic int8 is not acceptable on these models.** It is the fastest variant (v2-base: 6 ms per query at 8 threads, 2.7x the passage throughput of ONNX fp32) and 4x smaller, but costs
  **-2.19 macro / -3.64 ar->en points on v2-small and -3.11 / -4.81 on v2-base**, beyond the 1-point budget. Quantising only the MatMuls (leaving the embedding table in fp32) does not help (-2.16 and -3.32),
  so the loss comes from the transformer layers (activation outliers). Per-channel weight scales recover part of it (-0.65 on v2-small, -2.41 on v2-base): still too much for the base model.
  Do not ship dynamic int8 without calibration (static quantisation) or quantisation-aware training; neither was attempted.
* **Weight-only 8-bit (`w8`) is lossless but slow.** -0.01 (v2-small, CI [-0.12, +0.10]) and -0.02 (v2-base, CI [-0.12, +0.08]) macro points, 4x smaller (114 and 270 MiB) and about 2x less resident memory after
  loading; but ONNX Runtime's `MatMulNBits` CPU kernel dequantises on the fly, so a query costs 24 ms (v2-small) and 86 ms (v2-base) at 2 threads, about 1.7x slower than ONNX fp32.
* **Quantised queries against an fp32 index** (the setup of the demo) lose less than quantising both sides: dynamic int8 -1.15 (v2-small) and -1.10 (v2-base), per-channel int8 -0.32 and -0.65, `w8` +0.10 and -0.02 macro points.
* **Recommendation**: ONNX fp32 when latency matters (the demo uses it), `w8` when download size or memory matter more than latency, and per-channel int8 only for v2-small if a 0.65-point loss (0.32 with an fp32 index) is acceptable.
  fp16 was not measured: it only pays off on a GPU (off-limits for this task) or Apple's MPS.
* **v2-m3 (568M) was not exported or benchmarked.** Exporting and quantising its 2.2 GB weights needs more than the 6 GB per process allowed on the shared Spark (in the first round the 278M model already peaked at
  6.8 GB resident during quantisation, 6.05 GB for `w8`), and scoring it on this CPU would take hours. Everything takes the model path as an argument, so it is one command each once a machine allows it.
* **Mac numbers** (Apple M3 Pro, small candidate, 8-bit variants) exist only for the first round (`round1_candidates/bench_mac`) and were not repeated.
* **Peak RSS** is the peak of the whole benchmark process (python, pandas and the sampled texts, plus the batch-32 x 512-token activations); "RSS after load" is the same process right after loading the model.
* **Latency caveats**: shared 20-core ARM server with a GPU training job running (load average stored in every json, 6 to 12 during the final round), single run per cell, not an x86 CPU; tokenisation is included.
  Quality numbers do not depend on the load.

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
# the whole round on the final models (CPU only; quality jobs niced; latency benchmarks last, alone on the machine):
#   export, quality and bench for v2-small = final_small and v2-base = final_base -> ~/projects/v2-deploy/{artifacts_final,results_final}
bash deploy/run_final.sh
# or one model, one stage:
source deploy/spark_env.sh                        # CPU only, 8 threads, paths (preset ART / RES to keep rounds apart)
bash deploy/run_all.sh v2-small "$V2_MODELS/final_small" e5 quality
bash deploy/run_all.sh v2-small "$V2_MODELS/final_small" e5 bench
# v2-m3 (bge-m3 family: CLS pooling, no prefixes); not run, see the findings
bash deploy/run_all.sh v2-m3 "$V2_MODELS/final_m3" none quality

# back on the Mac: copy the small result files, rebuild the tables
RES_DIR=results_final ART_DIR=artifacts_final bash deploy/pull_results.sh
python -m deploy.summarize --write                # refreshes the table above
python report/make_tables.py                      # the report tables
```

Single steps: `python -m deploy.export_onnx --model DIR --out OUT`, `python -m deploy.eval_quality --name NAME --variant onnx-int8 --model DIR --onnx-root OUT --out RES`,
`python -m deploy.bench --name NAME --variant onnx-int8 --model DIR --onnx-root OUT --threads 2 --out RES`. Every step first waits for `MemAvailable >= 25 GiB` (`V2_MIN_MEM_GB`)
(`deploy/common.py: wait_for_memory`) because the Spark is shared; nothing here exceeds about 6 GB resident.

## Serving

`serve.py` is a small FastAPI server over any exported variant with a text-embeddings-inference-style `POST /embed` route (see the docstring):

```bash
ONNX_ROOT=artifacts/v2-base VARIANT=int8 THREADS=2 python -m uvicorn deploy.serve:app --port 8080
```

Not tested here: serving the sentence-transformers folder with TEI itself.
