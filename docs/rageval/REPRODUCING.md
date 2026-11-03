# Reproducing the RAG-evaluation experiments

Two levels of reproduction:

1. **Re-score the committed outputs** (CPU, seconds, no LLM, no dataset text needed). `results/rageval/` holds the planted-error set without
   passages, every judge verdict, and the generated answers without passage text. `tests/rageval/test_committed_results.py` re-computes the
   calibration tables and the before/after comparison from it and requires them to equal the committed reports exactly:

   ```bash
   ARR_RAGEVAL_DIR=$(mktemp -d)/rageval ; cp -R results/rageval "$ARR_RAGEVAL_DIR" ; export ARR_RAGEVAL_DIR
   python -m arretrieval.rageval.analyze_judges --split test --md /tmp/calibration_test.md       # == results/rageval/calibration_test.md
   python -m arretrieval.rageval.ab_test --base A_baseline --others I_rerank_lang --judge-model gemma \
       --faith-judge cite_v1 --faith-model qwen --md /tmp/ab_test.md                               # == results/rageval/ab_test.md
   ```

2. **From scratch**: the numbered scripts in `scripts/rageval/` against two OpenAI-compatible LLM servers.

## Environment

```bash
pip install -e ".[dev,rageval]"                  # adds torch, sentence-transformers, ragas==0.4.3 and the langchain 0.3 line
cp .env.example .env && $EDITOR .env && set -a && source .env && set +a
```

* `GEMMA_*`: generator and main judge. The published run used **Gemma 4 26B-A4B** via llama.cpp's `llama-server`, shared with other jobs and capped at
  **2 concurrent requests**, thinking off.
* `QWEN_*`: second judge. The published run used **Qwen3-4B-Instruct-2507** on a dedicated vLLM server (`vllm serve Qwen/Qwen3-4B-Instruct-2507
  --served-model-name qwen-judge --max-model-len 8192 --max-num-seqs 16 --enable-prefix-caching`), up to 12 concurrent requests.
* Retrieval for the system under test runs in-process on CPU (paraphrase-multilingual-MiniLM-L12-v2 + the mMiniLM cross-encoder); `bge-m3` (~2 GB) is
  downloaded for the Ragas answer-relevancy embedder.
* Exact package versions of the published run: [`requirements/rageval.lock.txt`](../../requirements/rageval.lock.txt) (aarch64 / CUDA 13; provenance only).

Every LLM call goes through `arretrieval/rageval/llm.py`: a cross-process concurrency cap (flock slot files), an on-disk SQLite cache keyed by the full
request (re-runs are free and a crashed job resumes), `enable_thinking=false`, and a seed in the cache key so that "n samples of the same prompt" are distinct.

## Stages

| Stage | Script | What it does | Needs |
|---|---|---|---|
| 1 | `scripts/rageval/01_data_and_index.sh` | download XQuAD (checksum-verified), build corpus / questions / pools, index, retrieval-only metrics | network, CPU |
| 2 | `scripts/rageval/02_baseline_rag.sh` | baseline RAG answers (config A), hand-checked Ragas toy cases | `GEMMA_*` |
| 3 | `scripts/rageval/03_planted_set_and_judges.sh` | planted-error set, every judge variant, Ragas on the planted set, calibration tables | `GEMMA_*`, `QWEN_*` |
| 4 | `scripts/rageval/04_improvement_experiment.sh` | baseline -> reranked + explicit-language prompt, fresh-question confirmation, Ragas on RAG answers, bias checks, judge-vs-judge | `GEMMA_*`, `QWEN_*` |
| 5 | `scripts/rageval/05_human_hook_and_publish.sh` | export the 40-item human-labelling sheet, publish text-free outputs to `results/rageval` | - |

Gemma throughput was the binding constraint of the published run (3-20 calls/min under shared load), so the schedule was trimmed and the report says where:
Gemma's chain-of-thought judges ran on the test split only, Ragas on the RAG answers ran on a stratified 20-question subset (+ 6 targeted retrieval
misses), Ragas on the planted set used the Qwen evaluator only, Gemma bias checks use n = 8 pairs per suite. With a faster endpoint nothing needs trimming.

## Notes

* **Retrieval backend.** The published runs queried a self-hosted embed/rerank HTTP service and a Qdrant collection serving the same two models with exact
  cosine search; this repository ships the in-process path (`rageval/retrieval.py`, an on-disk exact-cosine index) that the runs fell back to
  once the service was scaled down. Vectors were verified identical (cosine 1.000).
* **Non-determinism.** A different LLM (or the same one under another quantisation) gives different generated answers and base answers for the planted set,
  so a from-scratch run will not reproduce the published numbers exactly; the committed outputs do (level 1 above).
* **Human labels.** `export_human.py` writes the 40-row labelling sheet; `agreement.py` computes human-vs-judge agreement. The hook was tested on synthetic
  labels only; nobody has filled it in yet, so no human-agreement number is claimed anywhere.
