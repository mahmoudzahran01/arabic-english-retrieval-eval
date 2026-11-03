# Reproducing the retrieval experiments

Two levels of reproduction:

1. **Tables, RESULTS.md and the chart from the committed results** (CPU, seconds, no downloads): `make report`. The committed
   `results/retrieval/` snapshot holds one json per evaluated system plus the per-query metric arrays that feed the paired bootstrap, so
   every number in [`RESULTS.md`](RESULTS.md) and the README is regenerated, byte for byte, by `python -m arretrieval.retrieval.make_report`
   (`tests/retrieval/test_report.py` checks that on every CI run).
2. **From scratch** (GPU + an LLM endpoint): the numbered stage scripts in `scripts/retrieval/`, below.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"          # torch + sentence-transformers >= 5.0 + datasets; CUDA GPU for stages 2, 4, 5
cp .env.example .env && $EDITOR .env && set -a && source .env && set +a      # only stages 1 and 3 need an LLM endpoint
```

The published runs used a single **NVIDIA GB10 (DGX Spark)**, CUDA 13, torch 2.15, sentence-transformers 6.2, transformers 5.22, bf16 training.
The GPU's unified memory was shared with other jobs, so wall-clock times below are noisy and every stage script is idempotent
(finished items are skipped; a crashed item is retried up to three times).

## Stages

| Stage | Script | What it does | Needs | Time (published run) |
|---|---|---|---|---|
| 1 | `scripts/retrieval/01_data.sh` | download the public datasets, build pools / queries, LLM-translate 5,000 cross-lingual train/dev queries, leakage audit | network, LLM endpoint | ~2 min build; ~2 h translation at concurrency 2 |
| 2 | `scripts/retrieval/02_baselines.sh` | BM25 (raw / normalized / stemmed), e5-small, MiniLM, bge-m3, Arabic-normalized dense, RRF hybrids, language-gated hybrids | GPU | minutes |
| 3 | `scripts/retrieval/03_mining.sh` | hard-negative mining (e5-small and bge-m3 as miners), LLM false-negative audit, mining statistics | GPU, LLM endpoint | minutes |
| 4 | `scripts/retrieval/04_train_embedders.sh` | ~35 embedder fine-tuning runs (negatives, data size, epochs, hyper-parameters, cross-lingual data), each with dev + test evaluation | GPU | 1-30 min per run (5-13 min typical; 3 hard negatives 11 min; cross-lingual run ~10 min) |
| 5 | `scripts/retrieval/05_rerankers.sh` | reranker fine-tuning + ablations, rerank evaluations on five first stages, 568M reference reranker on a 1,050-query subset, typing-noise robustness, latency | GPU | 15-30 min per reranker training / evaluation |
| 6 | `scripts/retrieval/06_report.sh` | publish the small result files to `results/retrieval`, regenerate tables, RESULTS.md, chart | CPU | seconds |

Single runs, e.g. the final recipe and its reranker:

```bash
python -m arretrieval.retrieval.train_embedder --name e5s_hard1_xl_e1_s13 --neg-model e5-small-xl --n-neg 1 --margin 0.02 --xling --epochs 1 --seed 13 --keep-model
python -m arretrieval.retrieval.train_reranker --name rr_ft_bce_s13 --neg-model e5-small-xl --n-neg 4 --xling --seed 13
python -m arretrieval.retrieval.eval_rerank --first-stage e5s_hard1_xl_e1_s13 --reranker runs/retrieval/models/rr_ft_bce_s13 --name rr_ft_bce_s13
```

Outputs go to `runs/` (git-ignored, ~3 GB of weights and ~850 MB of stored top-100 rankings); `publish_results.py` copies only the small
files into `results/retrieval/`. System names like `e5-small>>rr_ft_bce_s13` (first stage >> reranker) are stored as `e5-small__rr_ft_bce_s13.*`
so that file names are valid on every OS.

## Seeds and determinism

* Data split seed `13` (title hash -> pool; every sampling step in `build_data.py` uses `SEED = 13` or `SEED + 1`). The split is a pure function of
  the pinned public datasets; the only non-deterministic ingredient is the LLM translation of the train/dev cross-lingual queries.
* Training seeds `13, 14, 15` for the headline comparisons (in-batch vs hard negatives; final recipe), `13` only for the one-off ablations. A seed fixes
  the data order, the no-duplicates batch sampler (`seed` and `data_seed`) and dropout; fine-tuning has no randomly initialised head.
* bf16 training on a shared GPU is not bit-reproducible, but the same configuration and seed twice gave macro MRR@10 70.67 vs 70.69; seed-to-seed std is ~0.2.
* Test-set numbers are from the *final* checkpoint of each run (no checkpoint selection on test). Dev was used for model selection only.
* Bootstrap: 2,000 resamples, seed 0.

## What actually ran (and what was re-issued)

The published results are the union of what finished; the original job queues were re-planned several times, and
`scripts/retrieval/04_*.sh` / `05_*.sh` are the consolidated, de-duplicated form of them:

* The first embedder queue stopped after `e5s_randneg1` hit a CUDA OOM during a machine-wide memory spike; its tail was re-issued with
  `--loss cached` for batch >= 128 (plain MNRL at batch 128 would exceed the 15 GB budget), and the one run that failed three times
  (`rank window 10-30`) was re-run once the spike had passed.
* `hard3_xl` first ran with plain MNRL (peak ~16 GB), was stopped and re-issued as `hard3c_xl` with CachedMNRL (5.4 GB).
* The reranker queue was re-planned three times (training/eval runs take 15-30 min each on the shared GPU); the strong 568M reranker was moved to a
  1,050-query subset (150 queries per slice) and every other system is re-scored on exactly the same subset (`eval_subset.py`).
* A 2-slot GPU lock was added after OOMs caused by too many concurrent jobs; it is not part of this repository.

## Deviations from the original plan (and why)

* **Data:** mMARCO and MIRACL-ar were not used. mMARCO's Arabic split has 8.8M passages; MIRACL's corpus has 2M passages and its Hugging Face loader is
  script-based (not loadable with `datasets>=4`). TyDi-QA GoldP, ARCD, SQuAD and XQuAD load as plain parquet and give the parallel ar/en test set that the ar->en
  question needs. See [`RESULTS.md`](RESULTS.md) section 7 and [`EXPLAINER.md`](EXPLAINER.md) section 12.
* **GPU budget:** all single runs stayed below 15 GB except one (3 hard negatives at batch 64 with plain MNRL: 15.9 GB peak).
* **bge-reranker-v2-m3** was scored on the 1,050-query subset rather than all 6,823 test queries.
* **LLM:** the translator and the false-negative judge ran one after the other with concurrency <= 2, thinking disabled.
