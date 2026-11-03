# Retrieval results - fine-tuning an embedder and a reranker with hard negatives

Everything below was measured on a single NVIDIA GB10 (DGX Spark) between 2026-11-02 and 2026-11-03. All numbers are **test-set** numbers unless a table is
labelled DEV; "x100" means percentage points (MRR@10 of 0.768 is printed 76.8). Mean +- std is over 3 training seeds (13, 14, 15); rows
without "+-" are a single seed (13) - read differences below about 1 point in those rows as noise (the 3-seed std of the headline recipe is 0.1-0.6).
Commands, seeds and file layout are in [`REPRODUCING.md`](REPRODUCING.md); the methods walk-through is [`EXPLAINER.md`](EXPLAINER.md).
This file is generated from the prose in `RESULTS.template.md` plus `results/retrieval/tables.md` (`python -m arretrieval.retrieval.build_results`), so the tables are never typed by hand.

## 0. TL;DR

* **Setup.** A bilingual (Arabic/English) retrieval benchmark of 24.5k passages and 31.4k queries built from TyDi-QA, ARCD, SQuAD and XQuAD, with **disjoint
  train/dev/test pools** (test = official TyDi dev, ARCD validation, XQuAD) and an independent leakage audit. A parallel XQuAD test set gives clean
  ar->ar, en->en, **ar->en** and en->ar slices over the same facts.
* **Baselines.** Off-the-shelf `multilingual-e5-small` has macro MRR@10 66.9; `bge-m3` 74.8. The ar->en slice is the weak spot of the small model (41.4 vs 82.7 for en->en over the same English corpus).
* **Fine-tuning works, mostly through two things:** mined hard negatives (+2.2 macro over in-batch only, 70.8 -> 73.0, 3 seeds) and **cross-lingual training pairs** (+2.7 macro on top, 73.0 -> 75.7; ar->en 49.4 -> 54.7, en->ar 74.1 -> 80.4).
  The final e5-small (118M params) reaches the 568M `bge-m3` on ar->ar and en->en, beats it by 12.6 points on en->ar, and is still 8.2 points behind it on ar->en.
* **Random negatives are worthless** (70.9, same as in-batch only), and **false-negative filtering matters**: a score-margin filter is worth +1.2 macro (and is direction-dependent: +4.1 on ar->en, -1.3 on en->ar vs no filter);
  an LLM audit found that 8.8% of the top-3 mined negatives actually contain the answer, rising to 19% when the candidate outscores the gold.
* **More epochs and bigger batches do not help** at this scale (3 epochs: 72.4 vs 73.0; a 512-batch with only 32 optimizer steps: 70.3). 25% of the data already gives ~80% of the gain.
* **English did not improve** (87.8 -> 87.7) and XQuAD-en got 1.8 points worse: fine-tuning on a mixed corpus trades a little English for Arabic and cross-lingual quality.
* **Lexical channel:** Arabic normalization + a light stemmer lifts BM25 ar->ar from 67.4 to 76.1; RRF hybrid helps Arabic (+3.6) but **hurts cross-lingual queries badly (ar->en -4.5, en->ar -34.8) unless BM25 is language-gated**.
* **Reranking is the biggest single lever for the cross-lingual directions:** on the baseline retriever, reranking the top-50 lifts macro MRR@10 from 66.9 to 80.5 (off-the-shelf mMiniLM cross-encoder) and 82.7 (fine-tuned), and ar->en from 41.4 to 68.9; the fully fine-tuned 118M stack (e5-small + reranker) reaches 83.3 and gated RRF(bge-m3, BM25) -> fine-tuned reranker 85.0. Reranking the top-50 costs a median 94 ms on a GPU shared with other jobs (39 ms for the top-20).

Headline table (test MRR@10 x100; ar->ar, en->en and en->ar/ar->en are macro-averaged over the slices listed in section 1):

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 (normalized + light stemmer) | 76.1 | 79.5 | 3.2 | 1.2 | 40.0 |
| multilingual-e5-small, off the shelf (118M) | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 |
| bge-m3 dense, off the shelf (568M) | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 |
| e5-small + BM25(stem), language-gated RRF | 80.4 | 86.4 | 41.4 | 61.7 | 67.5 |
| **fine-tuned e5-small** (hard negatives + cross-lingual pairs; 3 seeds) | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 |
| baseline e5-small -> top-50 -> off-the-shelf mMiniLM reranker | 85.3 | 90.9 | 65.7 | 80.0 | 80.5 |
| baseline e5-small -> top-50 -> **fine-tuned reranker** | 87.2 | 91.0 | 68.9 | 83.8 | 82.7 |
| **fine-tuned e5-small -> top-50 -> fine-tuned reranker** | 86.7 | 90.6 | 72.3 | 83.7 | 83.3 |
| gated RRF(bge-m3, BM25) -> top-50 -> fine-tuned reranker | 89.0 | 91.3 | 75.3 | 84.4 | 85.0 |

(Reranked rows use embedder seed 13 and a single reranker training run; every reranker scores the first stage's top-50.)

Same systems, Hit@1 (the fraction of queries whose gold passage is ranked first):

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 (normalized + light stemmer) | 68.3 | 72.3 | 2.4 | 0.7 | 35.9 |
| multilingual-e5-small, off the shelf (118M) | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 |
| bge-m3 dense, off the shelf (568M) | 73.1 | 82.1 | 52.4 | 59.2 | 66.7 |
| e5-small + BM25(stem), language-gated RRF | 72.2 | 80.5 | 31.7 | 51.4 | 59.0 |
| **fine-tuned e5-small** (hard negatives + cross-lingual pairs; 3 seeds) | 72.8 ± 0.2 | 81.4 ± 0.3 | 42.8 ± 0.8 | 72.7 ± 0.5 | 67.4 ± 0.2 |
| baseline e5-small -> top-50 -> off-the-shelf mMiniLM reranker | 80.3 | 86.9 | 58.7 | 74.9 | 75.2 |
| baseline e5-small -> top-50 -> **fine-tuned reranker** | 82.8 | 86.8 | 62.1 | 78.8 | 77.6 |
| **fine-tuned e5-small -> top-50 -> fine-tuned reranker** | 82.3 | 86.4 | 64.2 | 77.7 | 77.7 |
| gated RRF(bge-m3, BM25) -> top-50 -> fine-tuned reranker | 84.3 | 87.0 | 67.6 | 79.3 | 79.6 |

## 1. Data and protocol

**Sources** (all load as plain parquet; licences recorded in `results/retrieval/data_card/manifest.json` and in [`DATA.md`](../../DATA.md)):
TyDi-QA GoldP ar+en (Apache-2.0), ARCD (MIT), SQuAD v1.1 (CC BY-SA 4.0), XQuAD ar+en (CC BY-SA 4.0). Cross-lingual *training/dev* queries are machine translations
(Gemma-4-26B-A4B served by llama.cpp's `llama-server`; 5,000 questions, 100% passed the script check, spot-checked by eye) of real questions; cross-lingual *test* queries are XQuAD's human translations.

| split | passages (ar / en) | queries | what |
|---|---|---:|---|
| train pool | 5,425 / 5,227 | 20,918 (16,419 monolingual + 4,499 translated) | TyDi-ar/en train, ARCD train, SQuAD train (<= 2 questions per paragraph) |
| dev pool | 1,982 / 2,055 | 3,663 | held-out articles of the same sources + 500 translated queries; **model selection only** |
| test pool | 4,325 / 5,492 | 6,823 | TyDi-ar dev (921 q), TyDi-en dev (440), ARCD validation (702), XQuAD x 4 directions (4 x 1,190) |

Each split is searched only inside its own pool (so a test query competes against 4.3k Arabic or 5.5k English passages: gold passages plus distractors that were never used for training). Slices:
`tydi_ar`, `arcd_ar`, `xquad_ar2ar` -> **ar->ar**; `tydi_en`, `xquad_en2en` -> **en->en**; `xquad_ar2en` -> **ar->en** (Arabic question, English passage, gold = the parallel English paragraph);
`xquad_en2ar` -> **en->ar**. A direction's number is the macro-average of its slices; per-slice numbers for every system are in the appendix.

**Leakage controls** (`build_data.py`, checked independently by `audit_leakage.py`; result in `results/retrieval/data_card/leakage_audit.json`):
* whole articles are assigned to a pool by a hash of the title; every article that appears in a test source is removed from train and dev in all sources (3,016 rows dropped);
* identical passages / questions across sources are resolved by priority test > dev > train (a lower-priority duplicate is dropped);
* audit result: 0 queries whose gold is outside its pool; 0 shared passage ids; 0 test-gold article titles in train; 1 question text shared by dev and train (a translated duplicate);
  26 of 9,817 test passages (14 gold queries out of 6,823) have >= 50% 6-gram overlap with a training passage and 8 test questions are near-duplicates (Jaccard >= 0.8) of a training question (generic ones like "Where was he born?").
  Removing the 14 affected queries changes macro MRR@10 by +0.01 to +0.02 points for every system we checked (table in the appendix). Distractor-only articles can share a title with a training article (same Wikipedia article, different paragraphs): that is a property of the distractor pool, not of any gold passage.
* The dev set was used for model selection only (epoch count / early stopping and the final-recipe check, see 4.5); defaults such as the 0.02 margin were fixed from train-pool statistics before any training run, and every ablation is reported on test without selecting on it. Caveat: ~30 embedder configurations were compared on test, so the best row is mildly optimistic.

**Metrics** (`metrics.py`, hand-checked unit tests in `tests/`): Hit@k, MRR@10, nDCG@10 with one gold passage per query; paired bootstrap (2,000 resamples of queries, same indices for both systems) for differences.

## 2. Baselines (off the shelf)

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 raw tokens | 67.4 | 79.5 | 3.2 | 1.2 | 37.8 |
| BM25 + Arabic normalization | 71.3 | 79.5 | 3.1 | 1.2 | 38.8 |
| BM25 + normalization + light stemmer | 76.1 | 79.5 | 3.2 | 1.2 | 40.0 |
| MiniLM-L12 paraphrase-multilingual (dense) | 40.2 | 70.1 | 35.0 | 52.8 | 49.5 |
| multilingual-e5-small (dense) | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 |
| multilingual-e5-small, Arabic-normalized input | 74.2 | 87.8 | 34.8 | 61.7 | 64.6 |
| bge-m3 (dense) | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 |
| bge-m3, Arabic-normalized input | 79.0 | 87.8 | 59.0 | 64.3 | 72.5 |
| RRF e5-small + BM25(stem) | 80.4 | 86.4 | 36.9 | 26.9 | 57.7 |
| RRF e5-small + BM25(stem), language-gated | 80.4 | 86.4 | 41.4 | 61.7 | 67.5 |
| RRF bge-m3 + BM25(raw) | 77.6 | 86.8 | 56.1 | 34.8 | 63.8 |
| RRF bge-m3 + BM25(norm) | 79.7 | 86.8 | 55.8 | 34.8 | 64.3 |
| RRF bge-m3 + BM25(stem) | 82.7 | 86.8 | 54.5 | 34.8 | 64.7 |
| RRF bge-m3 + BM25(stem), language-gated | 82.7 | 86.8 | 63.3 | 67.8 | 75.1 |

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 raw tokens | 60.0 | 72.3 | 2.4 | 0.7 | 33.8 |
| BM25 + Arabic normalization | 63.3 | 72.3 | 2.3 | 0.7 | 34.6 |
| BM25 + normalization + light stemmer | 68.3 | 72.3 | 2.4 | 0.7 | 35.9 |
| MiniLM-L12 paraphrase-multilingual (dense) | 32.7 | 62.3 | 24.9 | 43.9 | 41.0 |
| multilingual-e5-small (dense) | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 |
| multilingual-e5-small, Arabic-normalized input | 65.3 | 81.7 | 26.4 | 50.9 | 56.1 |
| bge-m3 (dense) | 73.1 | 82.1 | 52.4 | 59.2 | 66.7 |
| bge-m3, Arabic-normalized input | 71.7 | 82.1 | 47.8 | 55.4 | 64.3 |
| RRF e5-small + BM25(stem) | 72.2 | 80.5 | 25.0 | 10.6 | 47.1 |
| RRF e5-small + BM25(stem), language-gated | 72.2 | 80.5 | 31.7 | 51.4 | 59.0 |
| RRF bge-m3 + BM25(raw) | 69.4 | 81.0 | 41.4 | 16.2 | 52.0 |
| RRF bge-m3 + BM25(norm) | 71.9 | 81.0 | 41.1 | 16.2 | 52.6 |
| RRF bge-m3 + BM25(stem) | 75.4 | 81.0 | 38.8 | 16.2 | 52.9 |
| RRF bge-m3 + BM25(stem), language-gated | 75.4 | 81.0 | 52.4 | 59.2 | 67.0 |

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 raw tokens | 82.2 | 91.8 | 5.3 | 2.9 | 45.5 |
| BM25 + Arabic normalization | 86.1 | 91.8 | 5.3 | 2.9 | 46.5 |
| BM25 + normalization + light stemmer | 90.2 | 91.8 | 5.5 | 2.9 | 47.6 |
| MiniLM-L12 paraphrase-multilingual (dense) | 55.8 | 86.6 | 59.1 | 70.9 | 68.1 |
| multilingual-e5-small (dense) | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 |
| multilingual-e5-small, Arabic-normalized input | 89.3 | 97.5 | 53.8 | 82.5 | 80.8 |
| bge-m3 (dense) | 91.9 | 97.0 | 84.7 | 84.4 | 89.5 |
| bge-m3, Arabic-normalized input | 90.7 | 97.0 | 81.8 | 81.3 | 87.7 |
| RRF e5-small + BM25(stem) | 93.5 | 96.3 | 63.1 | 73.3 | 81.5 |
| RRF e5-small + BM25(stem), language-gated | 93.5 | 96.3 | 62.9 | 82.7 | 83.8 |
| RRF bge-m3 + BM25(raw) | 91.8 | 96.2 | 83.6 | 78.5 | 87.5 |
| RRF bge-m3 + BM25(norm) | 92.8 | 96.2 | 83.6 | 78.5 | 87.8 |
| RRF bge-m3 + BM25(stem) | 94.3 | 96.2 | 83.6 | 78.5 | 88.2 |
| RRF bge-m3 + BM25(stem), language-gated | 94.3 | 96.2 | 84.7 | 84.4 | 89.9 |

(nDCG@10 and every per-slice number: appendix and `results/tables.md`.)

**Takeaways.**
* *Dense beats BM25 almost everywhere, but not in Arabic-to-Arabic.* BM25 with normalization + light stemming (76.1) ties e5-small (76.8) on ar->ar and is within 4 points of bge-m3 (80.2); on XQuAD-ar (translated, high lexical overlap) stemmed BM25 (85.3) *beats* both dense models (77.0 / 79.4).
  On natural TyDi questions (low lexical overlap) dense wins clearly (90.2 / 92.5 vs 77.8). Which regime your production queries look like decides how much BM25 matters.
* *Cross-lingual retrieval needs a multilingual dense model* (BM25 gets 3.2 / 1.2). `paraphrase-multilingual-MiniLM` is a sentence-similarity model, not a retrieval model: 40.2 on ar->ar. Pick retrieval-trained checkpoints.
* *The ar->en asymmetry is large for the small model:* e5-small gets 41.4 on ar->en vs 61.7 on en->ar (and 82.7 on en->en over the same English corpus); bge-m3 gets 63.3 vs 67.8. The ar->en vs en->ar gap partly reflects corpus differences (English test pool is 27% larger), so the cleaner comparison is the same-corpus one: ar->en vs en->en (-41.3 for e5-small, -17.3 for bge-m3).
* *Hybrid (RRF with BM25-stem):* ar->ar +3.6 (e5-small) / +2.5 (bge-m3); en->en -1.4 / -1.0; and **cross-lingual collapse** (en->ar 61.7 -> 26.9 for e5-small). The language-gated hybrid (RRF only when query and passage languages match) keeps the Arabic gain and removes the collapse (macro 67.5 and 75.1 vs 66.9 and 74.8 for dense alone) - the cross-lingual numbers of the gated rows are simply the dense numbers.
  Why the collapse: BM25 has almost no shared tokens across scripts, but the few it has (digits, Latin names) form a confident-looking rank-1 that rank-based fusion cannot discount.

### 2.1 Arabic normalization and typing-noise robustness

BM25 on ar->ar slices (MRR@10 x100):

| tokenization | tydi_ar | arcd_ar | xquad_ar2ar | ar->ar macro |
|---|---:|---:|---:|---:|
| raw tokens (diacritics, tatweel, hamza variants kept) | 71.0 | 51.5 | 79.8 | 67.4 |
| + normalization (alef/ya/ta-marbuta/diacritics/tatweel/digits) | 72.9 | 61.2 | 79.9 | 71.3 |
| + normalization + light stemmer | 77.8 | 65.1 | 85.3 | 76.1 |

* Normalization gives +3.9 and the stemmer another +4.8 on average; ARCD profits most (+9.6 from normalization alone: its text has more diacritics/tatweel/orthographic variation).
* **Dense models do not want normalized input**: feeding normalized Arabic to e5-small *lowers* ar->ar from 76.8 to 74.2 and ar->en from 41.4 to 34.8; bge-m3 drops 1.2 / 4.3. (Applied to both queries and passages.)
* **Typing-noise test** (63% of Arabic test queries corrupted by random hamza drops, ة->ه, ى->ي; passages untouched):

Fraction of Arabic test queries changed by the corruption: 0.634

| System | clean MRR@10 | corrupted-query MRR@10 | change |
|---|---:|---:|---:|
| bm25-raw | 67.4 | 61.8 | -5.7 |
| bm25-norm | 71.3 | 71.3 | +0.0 |
| bm25-stem | 76.1 | 76.1 | +0.0 |
| e5-small | 76.8 | 75.7 | -1.1 |
| bge-m3 | 80.2 | 79.6 | -0.5 |
| e5s_hard1_xl_e1_s13 | 80.2 | 79.2 | -1.0 |

  Raw BM25 loses 5.7 points; normalized BM25 is unaffected *by construction* (normalization is exactly the inverse of this corruption - so it is a statement about the lexical channel, not a surprise); dense models are fairly robust (e5-small -1.1, bge-m3 -0.5). The practical reading: normalization is cheap insurance for the BM25 channel; it is not needed to protect the dense channel against these particular variants.

## 3. Hard-negative mining and the false-negative problem

How the mining works (`negatives.py`): embed the train pool with the baseline e5-small, retrieve the top-100 for each train query (restricted to the pool of the gold passage's language), drop the gold, apply a policy, keep the hardest survivors.
How good is the miner? The gold is ranked first among its train pool in 75.4% of cases (top-10: 95.7%), so a quarter of the time the baseline already puts *something else* above the gold - exactly where false negatives live.

Score gap between the gold and the k-th best non-gold candidate (cosine; e5 scores live in 0.7-0.9):

| negative rank | median gap | 10th-90th pct | P(candidate scores above the gold) |
|---:|---:|---|---:|
| 1 | 0.026 | -0.023 .. 0.081 | 24.6% |
| 2 | 0.040 | -0.006 .. 0.092 | 13.5% |
| 3 | 0.047 | 0.000 .. 0.098 | 9.8% |
| 10 | 0.063 | 0.017 .. 0.113 | 4.3% |
| 30 | 0.079 | 0.031 .. 0.127 | 2.1% |

**LLM audit of false negatives** (`judge_false_negatives.py`): 160 train queries (80 ar, 80 en), for each the gold plus one non-gold candidate from ranks 1-3, 4-10 and 11-30; Gemma was asked "does this passage contain the answer?" (638 valid judgments).
The judge agrees that the *labelled gold* answers the question 91.2% of the time, so its "yes" rates below are lower bounds on the true false-negative rate.

| non-gold candidate | n | judged to contain the answer |
|---|---:|---:|
| rank 1-3 | 159 | **8.8%** |
| rank 4-10 | 160 | 1.9% |
| rank 11-30 | 160 | 0.6% |
| scores *above* the gold | 31 | **19.4%** |
| gap to gold 0 - 0.02 | 44 | 11.4% |
| gap 0.02 - 0.05 | 127 | 4.7% |
| gap 0.05 - 0.10 | 189 | 0.5% |
| gap > 0.10 | 88 | 0.0% |

So the danger zone is "scores within ~0.05 of the gold, or ranked in the top 3". That is the evidence behind the margin filter (default 0.02: it removes the 11-19% zone; 0.05 removes nearly all). Cost of the filter: with `n_neg = 1`, 6.1% of queries had no survivor at margin 0.02 (we pad with a random same-language negative), 22.9% at margin 0.05, 0% with no filter.
Example of a probable false negative (query "How much did House of 1000 Corpses gross?"): the miner's top candidate is the film's lead paragraph (cosine 0.900), the labelled gold is the "critical reception" paragraph (0.894).

## 4. Embedder fine-tuning (multilingual-e5-small, MultipleNegativesRankingLoss)

Recipe unless stated: batch 64 (no-duplicates sampler), lr 3e-5, warm-up 10% then linear decay, scale 20, bf16, max 256 tokens, 1 epoch over 16,419 monolingual queries (256 steps, 5-13 min on a GPU that was shared with other jobs), mined negatives from the *baseline* e5-small restricted to ranks 1-30 with a 0.02 cosine margin.
Evaluation: all text up to 512 tokens, final checkpoint, fp16.

### 4.1 (a) In-batch negatives vs mined hard negatives (and how we handle false negatives)

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 | 1 |
| fine-tuned, in-batch negatives only | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| + 1 RANDOM negative per query | 78.6 | 86.8 | 46.6 | 71.7 | 70.9 | 1 |
| + 1 mined hard negative, NO false-negative filter | 79.9 | 86.7 | 45.1 | 75.4 | 71.8 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| + 1 mined hard negative, margin 0.05 | 79.4 | 86.8 | 47.5 | 72.4 | 71.5 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 79.3 | 87.2 | 47.3 | 72.5 | 71.6 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 79.7 | 86.8 | 46.6 | 72.8 | 71.5 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 81.2 | 87.9 | 48.2 | 75.1 | 73.1 | 1 |
| + 3 mined hard negatives, margin 0.02 | 80.2 | 87.6 | 49.7 | 73.8 | 72.8 | 1 |

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 | 1 |
| fine-tuned, in-batch negatives only | 70.7 ± 0.3 | 79.6 ± 0.3 | 35.6 ± 1.2 | 61.2 ± 1.3 | 61.8 ± 0.2 | 3 |
| + 1 RANDOM negative per query | 70.2 | 80.5 | 35.7 | 61.8 | 62.0 | 1 |
| + 1 mined hard negative, NO false-negative filter | 72.7 | 80.4 | 34.8 | 66.2 | 63.5 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| + 1 mined hard negative, margin 0.05 | 71.2 | 80.4 | 36.2 | 62.8 | 62.6 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 71.4 | 81.1 | 36.1 | 62.9 | 62.9 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 71.7 | 80.5 | 35.7 | 63.4 | 62.8 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 74.2 | 82.1 | 37.2 | 66.4 | 65.0 | 1 |
| + 3 mined hard negatives, margin 0.02 | 72.3 | 81.6 | 38.9 | 64.4 | 64.3 | 1 |

**Takeaways.**
* In-batch negatives alone already help (+3.9 macro over the baseline, mostly on the Arabic and cross-lingual directions), but **English gets worse** (87.8 -> 86.4): the model forgets a little of what it knew.
* **One mined hard negative per query adds +2.2 macro** (70.8 +- 0.2 -> 73.0 +- 0.2, 3 seeds; paired bootstrap on seed 13: +1.8 / +1.1 / +3.3 / +2.4 points on the four directions, all CIs exclude 0) and repairs English (87.7).
* **Random negatives give nothing** (70.9 vs 70.8): the gain comes from hardness, not from the extra column.
* **The false-negative filter matters, and not uniformly.** Without a filter: 71.8 macro (-1.2). The effect is direction-dependent: en->ar *improves* without the filter (75.4 vs 74.1) while ar->en collapses (45.1 vs 49.4, CI of the difference excludes 0) - an unfiltered false negative on an Arabic-query/English-passage pair is the kind of label noise that hurts the very direction we care about.
  Stricter filters lose points too (margin 0.05: 71.5; dropping same-article candidates: 71.5; both pad many queries with random negatives and throw away informative near-misses). Skipping ranks 1-9 (candidates from ranks 10-30 only, margin 0.02) gave 71.6 (-1.3 vs the default): the hardest negatives are mostly true negatives, and discarding them costs more than the extra protection saves once the margin filter has removed the dangerous ones.
* **Who mines matters little:** mining with the stronger `bge-m3` instead of the baseline e5-small gave 73.1 vs 73.0 (single seed, within noise); 3 negatives per query gave 72.8 (also within noise, and 2x the cost: 15.9 GB peak memory at batch 64, 688 s vs 305 s).

### 4.2 (b) Data size

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 25% of queries | 77.8 | 85.0 | 45.7 | 72.4 | 70.2 | 1 |
| in-batch, 50% | 78.5 | 85.8 | 46.2 | 71.8 | 70.6 | 1 |
| in-batch, 100% | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| in-batch, 25% x 4 epochs (same #steps as 100% x 1) | 77.4 | 85.8 | 47.2 | 71.4 | 70.5 | 1 |
| hard neg, 25% of queries | 79.0 | 86.5 | 48.4 | 73.1 | 71.7 | 1 |
| hard neg, 50% | 80.1 | 87.0 | 49.5 | 73.5 | 72.5 | 1 |
| hard neg, 100% | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| hard neg, 25% x 4 epochs (same #steps as 100% x 1) | 79.1 | 86.5 | 48.9 | 72.6 | 71.8 | 1 |

With mined negatives, 25% / 50% / 100% of the queries give macro 71.7 / 72.5 / 73.0 (baseline 66.9): about 80% of the gain from 4k queries and one minute of training, and a flattening curve. Without hard negatives the curve is flatter still (70.2 / 70.6 / 70.8) - negatives, not volume, carry the signal.
Training 25% of the data for 4 epochs (the same 256 optimizer steps) recovers little (hard: 71.8; in-batch: 70.5), so the missing gain is data diversity, not step count.

### 4.3 (c) Epochs, and overfitting on dev

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 1 epoch | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| in-batch, 3 epochs | 78.3 | 86.7 | 46.5 | 71.7 | 70.8 | 1 |
| hard neg, 1 epoch | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| hard neg, 3 epochs | 80.6 | 87.8 | 48.9 | 72.1 | 72.4 | 1 |

- e5s_hard1_e3_s13: epoch 1: dev macro MRR@10 83.3, epoch 2: dev macro MRR@10 82.8, epoch 3: dev macro MRR@10 83.0
- e5s_inb_e3_s13: epoch 1: dev macro MRR@10 82.0, epoch 2: dev macro MRR@10 81.0, epoch 3: dev macro MRR@10 81.2

Dev MRR@10 peaks after the **first** epoch in both recipes (hard-negative: 83.3 -> 82.8 -> 83.0; in-batch: 82.0 -> 81.0 -> 81.2; mono macro = ar->ar, en->en) and the final 3-epoch test numbers are no better than 1 epoch (72.4 vs 73.0; 70.8 vs 70.8).
The final logged training loss keeps falling (hard: ~0.20 after 1 epoch -> 0.10 after 3; in-batch: ~0.08 -> 0.03) while dev does not move: the model is memorizing its few thousand training passages. With data this small, early stopping on dev at one epoch is the right default.

### 4.4 (d) Hyper-parameters and batch size

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| default: bs 64, lr 3e-5, scale 20 | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| scale 50 (temperature 0.02) | 81.3 | 88.2 | 47.1 | 73.9 | 72.6 | 1 |
| lr 1e-5 | 79.8 | 86.8 | 48.1 | 73.1 | 71.9 | 1 |
| lr 6e-5 | 80.5 | 87.7 | 49.4 | 74.2 | 72.9 | 1 |
| batch 32 | 80.8 | 87.8 | 49.1 | 73.5 | 72.8 | 1 |
| batch 128 (CachedMNRL) | 80.1 | 87.2 | 48.5 | 73.4 | 72.3 | 1 |
| batch 512 (CachedMNRL, 32 steps) | 77.3 | 85.0 | 47.2 | 71.9 | 70.3 | 1 |

The recipe is **insensitive** to the loss scale (20 vs 50), to lr between 3e-5 and 6e-5 (1e-5 is 1 point lower: not enough movement in 256 steps), and to batch size between 32 and 128 (72.8 / 72.9 / 72.3). Two sub-findings: (i) a 512-example batch via CachedMNRL
(the memory stays at 5.4 GB for batch 128; GradCache does what it promises) *under-trains* when the epoch count is fixed - only 32 optimizer steps: 70.3, i.e. 2.6 points worse - bigger batches need more epochs or a higher lr; (ii) at batch 64 with 3 negatives per query plain MNRL peaked at 15.9 GB, whereas the cached version needs 5.4 GB for the same gradient.

### 4.5 (e) Cross-lingual training data (the ar->en question)

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 | 1 |
| in-batch, monolingual data | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| in-batch, + translated cross-lingual queries | 79.0 | 86.1 | 53.3 | 80.1 | 74.6 | 1 |
| hard neg, monolingual data | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 | 3 |
| hard neg x3 (cached), + cross-lingual | 79.9 | 87.2 | 55.2 | 80.3 | 75.6 | 1 |
| hard neg, + cross-lingual, 2 epochs | 80.2 | 87.3 | 54.9 | 80.8 | 75.8 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 80.9 | 87.1 | 54.1 | 80.8 | 75.7 | 1 |
| bge-m3 dense (568M, off the shelf) | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 | 1 |

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 | 1 |
| in-batch, monolingual data | 92.2 ± 0.4 | 96.9 ± 0.1 | 71.0 ± 1.0 | 89.7 ± 0.8 | 87.4 ± 0.1 | 3 |
| in-batch, + translated cross-lingual queries | 91.8 | 96.8 | 77.0 | 94.0 | 89.9 | 1 |
| hard neg, monolingual data | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 92.3 ± 0.2 | 96.9 ± 0.1 | 79.6 ± 0.5 | 94.1 ± 0.3 | 90.7 ± 0.1 | 3 |
| hard neg x3 (cached), + cross-lingual | 92.1 | 96.9 | 80.2 | 93.9 | 90.8 | 1 |
| hard neg, + cross-lingual, 2 epochs | 92.2 | 96.8 | 79.6 | 94.3 | 90.7 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 92.0 | 97.0 | 78.4 | 94.7 | 90.5 | 1 |
| bge-m3 dense (568M, off the shelf) | 91.9 | 97.0 | 84.7 | 84.4 | 89.5 | 1 |

* **Translated cross-lingual pairs are the biggest lever for the cross-lingual directions**: with mined negatives ar->en 49.4 -> 54.7 +- 0.6 and en->ar 74.1 -> 80.4 +- 0.4 (paired bootstrap on seed 13: +5.9 and +6.4 points, CIs exclude 0), at the price of -0.5 on ar->ar and -0.2 on en->en (not significant). For in-batch-only the same data gives +6.6 / +9.0.
* Hit@10 for ar->en rises from 62.9 (baseline) to 79.6; the Hit@50 ceiling for a downstream reranker rises from 83.1 to 91.3.
* Final recipe (hard negatives, margin 0.02, cross-lingual pairs, 1 epoch) vs the 568M `bge-m3` baseline: ar->ar +0.0 [-1.0, +1.1], en->en -0.5 [-1.6, +0.5] (tied), en->ar **+12.6** [+10.5, +14.7], ar->en **-8.2** [-10.4, -6.2]. The training set had twice as many ar-query/en-passage pairs (3,000) as en-query/ar-passage pairs (1,499; translation throughput limited the latter), yet the two directions gained similar amounts (+5.3 ar->en, +6.3 en->ar): ar->en remains the hard direction for a 118M model, while bge-m3, trained with large-scale cross-lingual objectives, is 8 points ahead on it.
* **Choosing the final recipe.** Alternatives tried on top of the same data: 3 cached negatives (dev macro over the four directions 77.4, test 75.6), 2 epochs (dev 78.1, test 75.8) and bge-m3-mined negatives (dev 77.9, test 75.7), against 77.5 on dev and 75.7 on test for the final recipe. By dev alone the 2-epoch variant would have been picked (+0.6, driven by the machine-translated ar->en / en->ar dev slices); we kept the pre-declared 1-epoch recipe because it is the only one with three seeds and the test difference is 0.1 points. Honest flag: the dev set is translated the same way as the training data, so it slightly over-rewards extra cross-lingual training.

### 4.6 Paired bootstrap summary (seed 13, test MRR@10, points [95% CI])

| Comparison | ar->ar | en->en | ar->en | en->ar |
|---|---:|---:|---:|---:|
| hard neg (3 seeds, seed 13) vs in-batch only (seed 13) | +1.8 [+1.2, +2.4]* | +1.1 [+0.5, +1.7]* | +3.3 [+2.3, +4.3]* | +2.4 [+1.5, +3.3]* |
| random neg (seed 13) vs in-batch only (seed 13) | -0.3 [-0.7, +0.1] | +0.4 [-0.0, +0.8] | +0.7 [-0.1, +1.5] | +0.1 [-0.6, +0.6] |
| hard neg margin 0.02 vs no filter (seed 13) | +0.8 [-0.1, +1.7] | +0.8 [-0.0, +1.8] | +4.1 [+2.8, +5.3]* | -1.3 [-2.4, -0.3]* |
| + cross-lingual data vs mono, hard neg (seed 13) | -0.5 [-1.0, +0.1] | -0.2 [-0.6, +0.2] | +5.9 [+4.7, +7.2]* | +6.4 [+5.2, +7.5]* |
| FINAL (hard neg + xling, seed 13) vs e5-small baseline | +3.3 [+2.5, +4.2]* | -0.5 [-1.4, +0.4] | +13.7 [+11.8, +15.7]* | +18.7 [+16.9, +20.6]* |
| FINAL (hard neg + xling, seed 13) vs bge-m3 dense | +0.0 [-1.0, +1.1] | -0.5 [-1.6, +0.5] | -8.2 [-10.4, -6.2]* | +12.6 [+10.5, +14.7]* |

### 4.7 Training cost

Peak GPU memory (PyTorch allocator) and wall-clock on the GB10 (other jobs ran concurrently - times are noisy, memory is exact): in-batch 5.6 GB / ~5 min; 1 hard negative 9.0 GB / ~5-8 min; 3 hard negatives (plain MNRL) 15.9 GB / 11 min; cached losses 5.2-5.4 GB regardless of batch size; cross-lingual run (20.9k examples) ~10 min. Full table: appendix.

| Run | examples | steps | train s | peak GPU GB | first loss | last loss | padded negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| e5s_hard1_bgem3_e1_s13 | 16419 | 256 | 720 | 8.95 | 1.360 | 0.262 | 558 |
| e5s_hard1_bs128c_e1_s13 | 16419 | 128 | 487 | 5.36 | 1.386 | 0.264 | 996 |
| e5s_hard1_bs32_e1_s13 | 16419 | 513 | 393 | 5.22 | 1.299 | 0.150 | 996 |
| e5s_hard1_cached512_e1_s13 | 16419 | 32 | 492 | 5.36 | 1.666 | 1.666 | 996 |
| e5s_hard1_e1_s13 | 16419 | 256 | 305 | 8.99 | 1.348 | 0.199 | 996 |
| e5s_hard1_e1_s14 | 16419 | 256 | 259 | 8.99 | 1.349 | 0.217 | 996 |
| e5s_hard1_e1_s15 | 16419 | 256 | 272 | 8.99 | 1.354 | 0.218 | 996 |
| e5s_hard1_e3_s13 | 16419 | 768 | 852 | 8.99 | 1.864 | 0.099 | 996 |
| e5s_hard1_f25_e1_s13 | 4105 | 64 | 63 | 8.99 | 0.902 | 0.269 | 259 |
| e5s_hard1_f25_e4_s13 | 4105 | 256 | 474 | 8.99 | 1.347 | 0.091 | 259 |
| e5s_hard1_f50_e1_s13 | 8209 | 128 | 268 | 8.99 | 1.095 | 0.228 | 494 |
| e5s_hard1_lr1e5_e1_s13 | 16419 | 256 | 383 | 8.95 | 1.858 | 0.221 | 996 |
| e5s_hard1_lr6e5_e1_s13 | 16419 | 256 | 401 | 8.95 | 1.090 | 0.192 | 996 |
| e5s_hard1_m05_e1_s13 | 16419 | 256 | 514 | 8.95 | 1.299 | 0.132 | 3760 |
| e5s_hard1_nomargin_e1_s13 | 16419 | 256 | 510 | 8.99 | 1.454 | 0.437 | 0 |
| e5s_hard1_notitle_e1_s13 | 16419 | 256 | 400 | 8.95 | 1.326 | 0.153 | 1264 |
| e5s_hard1_rank10_e1_s13 | 16419 | 256 | 405 | 8.95 | 1.307 | 0.153 | 996 |
| e5s_hard1_sc50_e1_s13 | 16419 | 256 | 387 | 8.95 | 0.363 | 0.113 | 996 |
| e5s_hard1_xl_bgem3_e1_s13 | 20918 | 326 | 434 | 8.95 | 1.827 | 0.320 | 965 |
| e5s_hard1_xl_e1_s13 | 20918 | 326 | 611 | 8.99 | 1.817 | 0.267 | 2480 |
| e5s_hard1_xl_e1_s14 | 20918 | 326 | 753 | 8.99 | 1.836 | 0.286 | 2480 |
| e5s_hard1_xl_e1_s15 | 20918 | 326 | 699 | 8.95 | 1.810 | 0.276 | 2480 |
| e5s_hard1_xl_e2_s13 | 20918 | 652 | 889 | 8.95 | 2.176 | 0.183 | 2480 |
| e5s_hard3_e1_s13 | 16419 | 256 | 688 | 15.85 | 1.847 | 0.315 | 3053 |
| e5s_hard3c_xl_e1_s13 | 20918 | 326 | 1764 | 5.36 | 2.386 | 0.422 | 7546 |
| e5s_inb_e1_s13 | 16419 | 256 | 281 | 5.56 | 0.871 | 0.078 | 0 |
| e5s_inb_e1_s14 | 16419 | 256 | 240 | 5.57 | 0.870 | 0.089 | 0 |
| e5s_inb_e1_s15 | 16419 | 256 | 161 | 5.56 | 0.885 | 0.076 | 0 |
| e5s_inb_e3_s13 | 16419 | 768 | 635 | 5.56 | 1.263 | 0.029 | 0 |
| e5s_inb_f25_e1_s13 | 4105 | 64 | 35 | 5.56 | 0.539 | 0.103 | 0 |
| e5s_inb_f25_e4_s13 | 4105 | 256 | 142 | 5.56 | 0.867 | 0.032 | 0 |
| e5s_inb_f50_e1_s13 | 8209 | 128 | 156 | 5.57 | 0.689 | 0.083 | 0 |
| e5s_inb_xl_e1_s13 | 20918 | 326 | 326 | 5.56 | 1.269 | 0.123 | 0 |
| e5s_randneg1_e1_s13 | 16419 | 256 | 546 | 8.95 | 1.238 | 0.123 | 0 |

## 5. Cross-encoder reranker (retrieve top-50 -> rerank -> top-10)

Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M parameters, multilingual MiniLM, trained on mMARCO incl. Arabic). Fine-tuning (`train_reranker.py`): pointwise binary cross-entropy, positives up-weighted x4,
1 gold + 4 negatives per query for the same 20,918 training queries (incl. the translated cross-lingual ones), 1 epoch (3,268 steps, batch 32, lr 2e-5, max 256 tokens), negatives mined by the **baseline** e5-small (rank 1-30, margin 0.02) because that is the distribution a reranker sees from a first stage.
Every reranker below rescored the first stage's **top-50** and the metrics are computed on the re-ordered list (candidates outside the top-50 keep their first-stage order and never reach the top-10 unless the reranker's list is short).

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 |
| (a) + mMiniLM cross-encoder, off the shelf | 85.3 | 90.9 | 65.7 | 80.0 | 80.5 |
| (a) + fine-tuned reranker (mined hard negatives, + cross-lingual) | 87.2 | 91.0 | 68.9 | 83.8 | 82.7 |
| (a) + fine-tuned reranker, RANDOM negatives | 85.3 | 85.9 | 66.2 | 83.1 | 80.1 |
| (a) + fine-tuned reranker, no cross-lingual training data | 86.8 | 90.7 | 67.3 | 82.3 | 81.7 |
| (a) + fine-tuned reranker, rerank only top-10 | 85.9 | 91.7 | 58.0 | 78.1 | 78.4 |
| (a) + fine-tuned reranker, rerank only top-20 | 86.9 | 91.5 | 64.7 | 82.1 | 81.3 |
| (b) first stage fine-tuned e5-small, no reranker | 80.2 | 87.3 | 55.1 | 80.4 | 75.7 |
| (b) + mMiniLM cross-encoder, off the shelf | 85.5 | 90.7 | 69.8 | 79.9 | 81.5 |
| (b) + fine-tuned reranker | 86.7 | 90.6 | 72.3 | 83.7 | 83.3 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 82.7 | 86.8 | 63.3 | 67.8 | 75.1 |
| (c) + mMiniLM cross-encoder, off the shelf | 86.7 | 91.3 | 70.9 | 80.1 | 82.2 |
| (c) + fine-tuned reranker | 89.0 | 91.3 | 75.3 | 84.4 | 85.0 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 81.7 | 86.5 | 55.1 | 80.4 | 75.9 |
| (d) + fine-tuned reranker | 88.4 | 90.9 | 72.3 | 83.7 | 83.8 |

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 |
| (a) + mMiniLM cross-encoder, off the shelf | 80.3 | 86.9 | 58.7 | 74.9 | 75.2 |
| (a) + fine-tuned reranker (mined hard negatives, + cross-lingual) | 82.8 | 86.8 | 62.1 | 78.8 | 77.6 |
| (a) + fine-tuned reranker, RANDOM negatives | 78.5 | 78.2 | 58.0 | 77.1 | 72.9 |
| (a) + fine-tuned reranker, no cross-lingual training data | 82.5 | 86.6 | 60.3 | 76.5 | 76.5 |
| (a) + fine-tuned reranker, rerank only top-10 | 82.1 | 87.8 | 54.6 | 74.8 | 74.8 |
| (a) + fine-tuned reranker, rerank only top-20 | 82.7 | 87.5 | 60.2 | 77.9 | 77.1 |
| (b) first stage fine-tuned e5-small, no reranker | 72.7 | 81.6 | 43.4 | 72.8 | 67.6 |
| (b) + mMiniLM cross-encoder, off the shelf | 80.5 | 86.6 | 61.8 | 73.4 | 75.6 |
| (b) + fine-tuned reranker | 82.3 | 86.4 | 64.2 | 77.7 | 77.7 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 75.4 | 81.0 | 52.4 | 59.2 | 67.0 |
| (c) + mMiniLM cross-encoder, off the shelf | 81.5 | 87.2 | 62.6 | 74.9 | 76.5 |
| (c) + fine-tuned reranker | 84.3 | 87.0 | 67.6 | 79.3 | 79.6 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 74.1 | 80.8 | 43.4 | 72.8 | 67.8 |
| (d) + fine-tuned reranker | 83.7 | 86.8 | 64.2 | 77.7 | 78.1 |

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 |
| (a) + mMiniLM cross-encoder, off the shelf | 92.9 | 97.4 | 78.2 | 90.0 | 89.6 |
| (a) + fine-tuned reranker (mined hard negatives, + cross-lingual) | 94.0 | 97.4 | 80.2 | 93.4 | 91.2 |
| (a) + fine-tuned reranker, RANDOM negatives | 94.4 | 98.3 | 79.9 | 93.0 | 91.4 |
| (a) + fine-tuned reranker, no cross-lingual training data | 93.8 | 97.5 | 79.7 | 92.8 | 90.9 |
| (a) + fine-tuned reranker, rerank only top-10 | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 |
| (a) + fine-tuned reranker, rerank only top-20 | 92.7 | 97.7 | 72.2 | 88.7 | 87.9 |
| (b) first stage fine-tuned e5-small, no reranker | 92.1 | 96.9 | 79.3 | 93.9 | 90.6 |
| (b) + mMiniLM cross-encoder, off the shelf | 93.5 | 97.5 | 84.5 | 92.8 | 92.0 |
| (b) + fine-tuned reranker | 94.1 | 97.2 | 86.1 | 94.8 | 93.0 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 94.3 | 96.2 | 84.7 | 84.4 | 89.9 |
| (c) + mMiniLM cross-encoder, off the shelf | 95.0 | 98.1 | 86.4 | 90.0 | 92.4 |
| (c) + fine-tuned reranker | 96.6 | 97.9 | 89.0 | 92.9 | 94.1 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 94.2 | 96.4 | 79.3 | 93.9 | 91.0 |
| (d) + fine-tuned reranker | 96.3 | 97.4 | 86.1 | 94.8 | 93.6 |

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 95.3 | 99.4 | 83.1 | 94.1 | 93.0 |
| (b) first stage fine-tuned e5-small, no reranker | 96.4 | 99.1 | 91.3 | 98.4 | 96.3 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 98.1 | 99.8 | 93.8 | 94.0 | 96.4 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 98.3 | 99.4 | 91.3 | 98.4 | 96.8 |

| Comparison | ar->ar | en->en | ar->en | en->ar |
|---|---:|---:|---:|---:|
| (a) FT reranker vs off-the-shelf mMiniLM | +1.9 [+1.1, +2.6]* | +0.0 [-0.9, +1.0] | +3.1 [+1.8, +4.5]* | +3.8 [+2.6, +5.1]* |
| (a) FT reranker (mined negs) vs FT reranker (random negs) | +1.8 [+1.0, +2.7]* | +5.0 [+3.7, +6.4]* | +2.7 [+1.4, +4.0]* | +0.8 [-0.3, +1.9] |
| (a) FT reranker vs no reranker | +10.3 [+9.1, +11.5]* | +3.2 [+1.8, +4.6]* | +27.5 [+25.1, +29.9]* | +22.1 [+19.8, +24.6]* |
| (b) FT reranker behind FT embedder vs FT embedder alone | +6.5 [+5.2, +7.6]* | +3.3 [+1.8, +4.8]* | +17.2 [+14.8, +19.7]* | +3.3 [+1.1, +5.4]* |
| (b) vs (a): FT embedder + FT reranker vs baseline embedder + FT reranker | -0.5 [-1.0, +0.0] | -0.4 [-0.8, +0.0] | +3.5 [+1.8, +5.0]* | -0.1 [-1.2, +1.0] |

**Takeaways.**
* **Reranking is the largest single improvement we measured**: baseline e5-small 66.9 -> 80.5 with the *off-the-shelf* cross-encoder and 82.7 after fine-tuning it (Hit@1: 58.3 -> 75.2 -> 77.6). The gains are biggest where the first stage is weakest: ar->en +27.5 and en->ar +22.1 MRR@10 points (fine-tuned reranker vs no reranker; CIs exclude 0), and still +10.3 on ar->ar and +3.2 on en->en.
* **Fine-tuning the reranker on mined negatives adds 2.2 macro points over the off-the-shelf model** (80.5 -> 82.7; ar->en +3.1 [+1.8, +4.5], en->ar +3.8 [+2.6, +5.1], ar->ar +1.9, en->en +0.0). The off-the-shelf model is already a strong multilingual reranker, so there is less headroom than for the bi-encoder.
* **A reranker trained on random negatives is worse than no fine-tuning at all on English** (en->en MRR 85.9 vs 90.9 off-the-shelf vs 91.0 hard-negative; Hit@1 78.2 vs 86.9 vs 86.8): it learned to separate gold from irrelevant passages, not gold from near-misses - exactly the discrimination a reranker exists for. Mined negatives are worth +2.6 macro here (82.7 vs 80.1), +5.0 on en->en.
* **First-stage quality still matters after reranking, mostly through recall.** Fine-tuned first stage + fine-tuned reranker (83.3) vs baseline first stage + the same reranker (82.7): +3.5 on ar->en (the direction where Hit@50 rose from 83.1 to 91.3) and within +-0.5 elsewhere (not significant). A reranker cannot recover a gold passage that is not in the candidate list: for ar->en the baseline retriever misses it 17% of the time.
* **Best combination measured:** language-gated RRF(bge-m3, BM25-stem) -> fine-tuned reranker, macro 85.0 (Hit@10 94.1) - note the reranker was trained with negatives from a *different* first stage (e5-small) and transfers fine. Our fine-tuned small stack (e5-small + FT + FT reranker) gets 83.3, 1.7 points lower, with a ~5x smaller first-stage model.
* Production-shaped pipeline (d) (gated hybrid with the fine-tuned e5-small + BM25 -> FT reranker): macro 83.8 (ar->ar 88.4 vs 86.7 without the BM25 channel, en->en 90.9, ar->en 72.3, en->ar 83.7): BM25 adds +1.7 on ar->ar and nothing elsewhere (it is gated off for cross-lingual queries). It is the best system built from 118M-parameter models only; the only better one (85.0) puts the 568M bge-m3 in the first stage.
* **Depth:** reranking only the top-10 / 20 / 50 candidates of the baseline e5-small with the fine-tuned reranker gives macro MRR@10 78.4 / 81.3 / 82.7 (ar->en 58.0 / 64.7 / 68.9; en->en is flat at 91-92). The curve follows first-stage recall (ar->en Hit@10 = 62.9, Hit@50 = 83.1): depth buys back exactly the gold passages the first stage ranks low. Going from 20 to 50 candidates costs 2.4x the GPU time (39 -> 94 ms) for +1.4 macro and +4.2 on ar->en; depth 100 was not run.
* **Cross-lingual reranker training data:** training the reranker without the translated cross-lingual pairs costs 1.0 macro point (81.7 vs 82.7), concentrated where expected - ar->en 67.3 vs 68.9 and en->ar 82.3 vs 83.8 (ar->ar 86.8 vs 87.2, en->en 90.7 vs 91.0). Translated pairs help the reranker too, though less than they help the embedder.
* **Strong reference reranker** (`BAAI/bge-reranker-v2-m3`, 568M, off the shelf, scored on a 1,050-query subset because it is ~15x the compute): on those queries the 568M reranker gets macro MRR@10 83.7 (ar->ar 86.5, en->en 93.7, ar->en 75.1, en->ar 79.3) against 83.3 for our fine-tuned 118M reranker (87.8 / 91.3 / 69.7 / 84.4) and 81.8 for the off-the-shelf 118M one. A tie overall: the large model is better on en->en (+2.4) and ar->en (+5.4) and worse on en->ar (-5.1), at 6-7x the latency (560 ms vs 94 ms for the top-50 on GPU). In-distribution fine-tuning of the small reranker closes the *macro* gap to a model five times its size, but not the ar->en gap.

| System (same 1050 test queries: 150 per slice) | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| baseline e5-small, no reranker | 77.6 | 87.4 | 41.7 | 61.3 | 67.0 |
| + mMiniLM cross-encoder (118M), off the shelf | 87.0 | 91.8 | 67.2 | 81.2 | 81.8 |
| + mMiniLM cross-encoder, fine-tuned (mined negatives) | 87.8 | 91.3 | 69.7 | 84.4 | 83.3 |
| + mMiniLM cross-encoder, fine-tuned (random negatives) | 87.0 | 87.0 | 66.5 | 84.4 | 81.2 |
| + bge-reranker-v2-m3 (568M), off the shelf | 86.5 | 93.7 | 75.1 | 79.3 | 83.7 |

### 5.1 Latency (per query, one batch of N candidates, tokenization included)

| Component | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|
| mmarco-mMiniLM (off-the-shelf), cuda, top20 | 38.5 | 47.1 | 36.3 |
| mmarco-mMiniLM (off-the-shelf), cuda, top50 | 94.3 | 120.4 | 93.7 |
| mmarco-mMiniLM (off-the-shelf), cpu 1thr, top20 | 3066.6 | 3261.1 | 3068.8 |
| mmarco-mMiniLM (off-the-shelf), cpu 1thr, top50 | 7206.6 | 7406.1 | 7136.3 |
| mmarco-mMiniLM (off-the-shelf), cpu 4thr, top20 | 933.6 | 1280.2 | 986.8 |
| mmarco-mMiniLM (off-the-shelf), cpu 4thr, top50 | 3160.7 | 3459.6 | 3142.3 |
| mmarco-mMiniLM (off-the-shelf), cpu 8thr, top20 | 1242.1 | 1304.8 | 1247.5 |
| mmarco-mMiniLM (off-the-shelf), cpu 8thr, top50 | 3214.7 | 3491.3 | 3237.7 |
| bge-reranker-v2-m3 (off-the-shelf), cuda, top20 | 262.0 | 302.9 | 263.9 |
| bge-reranker-v2-m3 (off-the-shelf), cuda, top50 | 559.9 | 594.6 | 555.3 |
| bge-reranker-v2-m3 (off-the-shelf), cpu 8thr, top20 | 10737.3 | 12102.2 | 10871.8 |
| embed one query, e5-small, cuda | 16.0 | 19.5 | 15.9 |
| embed one query, e5-small, cpu 8thr | 17.2 | 33.8 | 20.4 |
| embed one query, bge-m3, cuda | 39.3 | 55.2 | 38.9 |
| embed one query, bge-m3, cpu 8thr | 172.1 | 218.0 | 167.8 |

Reading the table: scoring the top-50 with the 118M mMiniLM cross-encoder takes a median **94 ms on the GPU (shared with other jobs)** (p95 120 ms; 39 ms for the top-20) - about 6x the cost of embedding the query for the dense first stage (e5-small: 16 ms on GPU, 17 ms on CPU). On CPU it is a different regime: 3.1 s for the top-20 and 7.2 s for the top-50 with 1 thread, 0.9 s / 3.2 s with 4 threads; 8 threads were *not* faster (1.2 s / 3.2 s) because the host's cores were shared with other tenants' jobs, so treat the CPU numbers as pessimistic. The fine-tuned reranker has the same architecture as the off-the-shelf one, so its latency is identical. The 568M `bge-reranker-v2-m3` is 6-7x slower on GPU (262 ms for the top-20, 560 ms for the top-50) and takes ~11 s per query for the top-20 on 8 CPU threads: that is what "a stronger reranker" costs. As a query encoder `bge-m3` costs 39 ms on GPU and 172 ms on CPU vs 16 / 17 ms for e5-small. Rule of thumb from these numbers: CPU-only cross-encoder reranking is only viable for very short candidate lists (top-10/20) or with quantization / ONNX (not tested here); a GPU makes top-50 comfortable (< 0.1 s). Timing method: warm-up, then median / p95 over 60 test queries (GPU) or 12 (CPU), one batch of N (query, passage) pairs per query, max 256 tokens, tokenization included.

## 6. Negative results, surprises and things that did not work

* **Fine-tuning did not improve English** (87.8 -> 87.7 for hard negatives; XQuAD-en 82.7 -> 80.8 for the final recipe; the in-batch-only recipe loses 1.4). The base model is already strong on English retrieval; a mixed-language fine-tune on 16-21k queries mainly trades quality, and hard negatives limit the damage.
* **More epochs (3) and a bigger effective batch (512) did not help**; 3 epochs overfit dev after epoch 1, and the 512-batch had only 32 optimizer steps (underfit). Lesson: with small data, the number of *useful updates* matters more than the number of negatives per update.
* **Random negatives = no negatives.** Same for the reranker, where they are actively harmful on English.
* **Stricter false-negative filtering is not better filtering**: margin 0.05 and "drop same-article" both lost ~1.5 points against margin 0.02 because they forced random padding for 7-23% of queries.
* **The off-the-shelf `paraphrase-multilingual-MiniLM` is not a retrieval model** (ar->ar 40.2).
* **Normalizing Arabic for the dense models hurt** (-2.6 ar->ar, -6.6 ar->en for e5-small).
* **Equal-weight RRF hurts wherever one channel is much weaker, and is destructive for cross-lingual queries** (en->ar 61.7 -> 26.9) unless BM25 is gated by language.
* **Operational:** the GPU's unified memory was shared with other jobs, and several runs failed with CUDA out-of-memory when those jobs spiked (once the whole machine became unreachable for ~15 minutes with a load average above 500; the queues survived it). Before a 2-slot GPU lock was added up to four of our containers ran at once (each under 16 GB, roughly 20-25 GB aggregate), which did not help. The fixes were retry logic, the lock, and CachedMNRL for large batches / many negatives (a plain 3-negative batch-64 run peaked at 15.9 GB, just above the 15 GB budget - the one run that exceeded it; all later 3-negative runs used the cached loss at 5.4 GB). The one ablation that failed three times (`rank window 10-30`) was re-run once the spike had passed.
* **Reproducibility note:** the identical configuration trained with the same seed twice (`inb_e1_s13`, smoke run vs rerun) scored macro 70.67 vs 70.69, i.e. bf16 training is reproducible to ~0.02 points here (same GPU, same software); seed-to-seed std is ~0.2.

## 7. What this does NOT show

See [`EXPLAINER.md` section 12](EXPLAINER.md) for the longer list. In one breath: Wikipedia factoid QA is not an enterprise-document domain; test corpora of 4-5k passages are tiny compared to production; one gold per query understates every system; cross-lingual *training* queries are clean machine translations; XQuAD is parallel content; one small model family; English did not improve; 25+ configurations were compared on the same test set; latencies were taken on a machine shared with other jobs.

## Appendix A. Selection on dev (model-selection views)

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 72.6 | 87.3 | 55.5 | 67.7 | 79.9 | 1 |
| fine-tuned, in-batch negatives only | 76.9 ± 0.5 | 86.5 ± 0.1 | n/a | n/a | 81.7 ± 0.2 | 3 |
| + 1 RANDOM negative per query | 76.9 | 87.0 | 58.0 | 73.9 | 81.9 | 1 |
| + 1 mined hard negative, NO false-negative filter | 76.7 | 87.6 | 59.2 | 72.1 | 82.1 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| + 1 mined hard negative, margin 0.05 | 78.5 | 86.8 | 58.7 | 74.6 | 82.6 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 78.0 | 86.9 | 59.5 | 73.8 | 82.4 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 77.4 | 86.9 | 59.3 | 73.8 | 82.1 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 78.4 | 88.2 | 61.1 | 74.4 | 83.3 | 1 |
| + 3 mined hard negatives, margin 0.02 | 79.3 | 87.3 | n/a | n/a | 83.3 | 1 |

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 72.6 | 87.3 | 55.5 | 67.7 | 79.9 | 1 |
| in-batch, monolingual data | 76.9 ± 0.5 | 86.5 ± 0.1 | n/a | n/a | 81.7 ± 0.2 | 3 |
| in-batch, + translated cross-lingual queries | 76.5 | 86.5 | 65.4 | 78.1 | 81.5 | 1 |
| hard neg, monolingual data | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 78.1 ± 0.5 | 87.1 ± 0.1 | 66.1 ± 0.2 | 78.5 ± 0.4 | 82.6 ± 0.2 | 3 |
| hard neg x3 (cached), + cross-lingual | 78.0 | 87.1 | 66.0 | 78.5 | 82.6 | 1 |
| hard neg, + cross-lingual, 2 epochs | 77.5 | 87.0 | 67.3 | 80.4 | 82.3 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 79.0 | 87.9 | 66.9 | 77.6 | 83.5 | 1 |

## Appendix B. Remaining test metrics for the embedders

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 | 1 |
| in-batch, monolingual data | 70.7 ± 0.3 | 79.6 ± 0.3 | 35.6 ± 1.2 | 61.2 ± 1.3 | 61.8 ± 0.2 | 3 |
| in-batch, + translated cross-lingual queries | 71.0 | 79.6 | 41.7 | 72.4 | 66.2 | 1 |
| hard neg, monolingual data | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 72.8 ± 0.2 | 81.4 ± 0.3 | 42.8 ± 0.8 | 72.7 ± 0.5 | 67.4 ± 0.2 | 3 |
| hard neg x3 (cached), + cross-lingual | 72.3 | 81.4 | 43.1 | 72.4 | 67.3 | 1 |
| hard neg, + cross-lingual, 2 epochs | 72.8 | 81.6 | 43.2 | 73.2 | 67.7 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 73.9 | 81.0 | 42.4 | 73.0 | 67.6 | 1 |
| bge-m3 dense (568M, off the shelf) | 73.1 | 82.1 | 52.4 | 59.2 | 66.7 | 1 |

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 | 1 |
| fine-tuned, in-batch negatives only | 92.2 ± 0.4 | 96.9 ± 0.1 | 71.0 ± 1.0 | 89.7 ± 0.8 | 87.4 ± 0.1 | 3 |
| + 1 RANDOM negative per query | 92.0 | 96.7 | 70.5 | 90.1 | 87.3 | 1 |
| + 1 mined hard negative, NO false-negative filter | 91.9 | 96.6 | 69.2 | 91.4 | 87.3 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| + 1 mined hard negative, margin 0.05 | 92.5 | 96.9 | 72.4 | 90.3 | 88.0 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 92.2 | 97.0 | 71.3 | 90.5 | 87.8 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 92.6 | 96.9 | 70.2 | 90.3 | 87.5 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 92.8 | 97.1 | 72.4 | 91.2 | 88.4 | 1 |
| + 3 mined hard negatives, margin 0.02 | 92.8 | 97.0 | 72.9 | 91.1 | 88.5 | 1 |

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small baseline | 80.2 | 90.2 | 46.5 | 66.8 | 70.9 | 1 |
| e5s_hard1_bgem3_e1 | 84.1 | 90.2 | 54.0 | 79.0 | 76.8 | 1 |
| e5s_hard1_bs128c_e1 | 83.2 | 89.7 | 54.2 | 77.6 | 76.1 | 1 |
| e5s_hard1_bs32_e1 | 83.8 | 90.1 | 54.9 | 77.7 | 76.6 | 1 |
| e5s_hard1_cached512_e1 | 80.8 | 87.7 | 53.1 | 76.3 | 74.5 | 1 |
| e5s_hard1_e1 | 83.8 ± 0.1 | 90.0 ± 0.2 | 55.3 ± 0.5 | 78.3 ± 0.1 | 76.8 ± 0.2 | 3 |
| e5s_hard1_e3 | 83.6 | 90.1 | 54.9 | 76.4 | 76.2 | 1 |
| e5s_hard1_f25_e1 | 82.2 | 89.0 | 54.1 | 77.4 | 75.6 | 1 |
| e5s_hard1_f25_e4 | 82.3 | 89.0 | 54.6 | 76.9 | 75.7 | 1 |
| e5s_hard1_f50_e1 | 83.2 | 89.5 | 55.1 | 77.9 | 76.4 | 1 |
| e5s_hard1_lr1e5_e1 | 82.9 | 89.3 | 53.9 | 77.4 | 75.9 | 1 |
| e5s_hard1_lr6e5_e1 | 83.5 | 90.0 | 55.4 | 78.3 | 76.8 | 1 |
| e5s_hard1_m05_e1 | 82.7 | 89.3 | 53.5 | 76.7 | 75.6 | 1 |
| e5s_hard1_nomargin_e1 | 82.8 | 89.1 | 50.9 | 79.3 | 75.5 | 1 |
| e5s_hard1_notitle_e1 | 82.9 | 89.3 | 52.3 | 77.1 | 75.4 | 1 |
| e5s_hard1_rank10_e1 | 82.5 | 89.6 | 53.1 | 76.9 | 75.5 | 1 |
| e5s_hard1_sc50_e1 | 84.3 | 90.5 | 52.9 | 78.1 | 76.4 | 1 |
| e5s_hard1_xl_bgem3_e1 | 83.6 | 89.5 | 60.0 | 84.2 | 79.3 | 1 |
| e5s_hard1_xl_e1 | 83.2 ± 0.2 | 89.6 ± 0.2 | 60.7 ± 0.5 | 83.8 ± 0.3 | 79.3 ± 0.0 | 3 |
| e5s_hard1_xl_e1_s13>>rr_ft_bce | 88.5 | 92.2 | 75.7 | 86.4 | 85.7 | 1 |
| e5s_hard1_xl_e2 | 83.1 | 89.6 | 60.8 | 84.1 | 79.4 | 1 |
| e5s_hard3_e1 | 83.3 | 90.0 | 55.3 | 78.0 | 76.6 | 1 |
| e5s_hard3c_xl_e1 | 82.9 | 89.6 | 61.2 | 83.6 | 79.3 | 1 |
| e5s_inb_e1 | 82.1 ± 0.0 | 89.0 ± 0.2 | 52.5 ± 1.0 | 75.7 ± 1.0 | 74.8 ± 0.1 | 3 |
| e5s_inb_e3 | 81.8 | 89.3 | 52.6 | 76.2 | 74.9 | 1 |
| e5s_inb_f25_e1 | 81.2 | 87.8 | 51.6 | 76.8 | 74.3 | 1 |
| e5s_inb_f25_e4 | 80.8 | 88.4 | 53.0 | 75.8 | 74.5 | 1 |
| e5s_inb_f50_e1 | 81.8 | 88.5 | 52.0 | 76.3 | 74.7 | 1 |
| e5s_inb_xl_e1 | 82.1 | 88.7 | 59.0 | 83.5 | 78.4 | 1 |
| e5s_randneg1_e1 | 81.9 | 89.3 | 52.3 | 76.1 | 74.9 | 1 |

## Appendix C. Every slice, every metric, every baseline

| System | Slice | n | hit@1 | hit@5 | hit@10 | mrr@10 | ndcg@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 raw tokens | arcd_ar | 702 | 41.2 | 65.2 | 73.1 | 51.5 | 56.7 |
| BM25 raw tokens | tydi_ar | 921 | 64.8 | 79.3 | 83.0 | 71.0 | 73.9 |
| BM25 raw tokens | tydi_en | 440 | 69.3 | 87.3 | 89.1 | 76.7 | 79.8 |
| BM25 raw tokens | xquad_ar2ar | 1190 | 74.1 | 86.8 | 90.4 | 79.8 | 82.4 |
| BM25 raw tokens | xquad_ar2en | 1190 | 2.4 | 4.4 | 5.3 | 3.2 | 3.7 |
| BM25 raw tokens | xquad_en2ar | 1190 | 0.7 | 1.6 | 2.9 | 1.2 | 1.6 |
| BM25 raw tokens | xquad_en2en | 1190 | 75.2 | 91.8 | 94.5 | 82.3 | 85.3 |
| BM25 + Arabic normalization | arcd_ar | 702 | 49.1 | 76.9 | 82.2 | 61.2 | 66.3 |
| BM25 + Arabic normalization | tydi_ar | 921 | 66.3 | 82.3 | 85.8 | 72.9 | 76.0 |
| BM25 + Arabic normalization | tydi_en | 440 | 69.3 | 87.3 | 89.1 | 76.7 | 79.8 |
| BM25 + Arabic normalization | xquad_ar2ar | 1190 | 74.3 | 87.0 | 90.3 | 79.9 | 82.4 |
| BM25 + Arabic normalization | xquad_ar2en | 1190 | 2.3 | 4.4 | 5.3 | 3.1 | 3.6 |
| BM25 + Arabic normalization | xquad_en2ar | 1190 | 0.7 | 1.6 | 2.9 | 1.2 | 1.6 |
| BM25 + Arabic normalization | xquad_en2en | 1190 | 75.2 | 91.8 | 94.5 | 82.3 | 85.3 |
| BM25 + normalization + light stemmer | arcd_ar | 702 | 52.7 | 80.8 | 87.5 | 65.1 | 70.6 |
| BM25 + normalization + light stemmer | tydi_ar | 921 | 71.7 | 86.0 | 89.5 | 77.8 | 80.6 |
| BM25 + normalization + light stemmer | tydi_en | 440 | 69.3 | 87.3 | 89.1 | 76.7 | 79.8 |
| BM25 + normalization + light stemmer | xquad_ar2ar | 1190 | 80.4 | 91.7 | 93.7 | 85.3 | 87.4 |
| BM25 + normalization + light stemmer | xquad_ar2en | 1190 | 2.4 | 4.6 | 5.5 | 3.2 | 3.8 |
| BM25 + normalization + light stemmer | xquad_en2ar | 1190 | 0.7 | 1.6 | 2.9 | 1.2 | 1.6 |
| BM25 + normalization + light stemmer | xquad_en2en | 1190 | 75.2 | 91.8 | 94.5 | 82.3 | 85.3 |
| MiniLM-L12 paraphrase-multilingual (dense) | arcd_ar | 702 | 20.8 | 38.9 | 43.3 | 28.3 | 31.9 |
| MiniLM-L12 paraphrase-multilingual (dense) | tydi_ar | 921 | 36.2 | 51.5 | 57.7 | 42.8 | 46.4 |
| MiniLM-L12 paraphrase-multilingual (dense) | tydi_en | 440 | 76.8 | 89.8 | 92.5 | 82.3 | 84.8 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_ar2ar | 1190 | 41.1 | 60.6 | 66.4 | 49.5 | 53.6 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_ar2en | 1190 | 24.9 | 48.2 | 59.1 | 35.0 | 40.8 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_en2ar | 1190 | 43.9 | 65.1 | 70.9 | 52.8 | 57.1 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_en2en | 1190 | 47.8 | 72.4 | 80.7 | 58.0 | 63.4 |
| multilingual-e5-small (dense) | arcd_ar | 702 | 50.3 | 79.3 | 84.0 | 63.4 | 68.5 |
| multilingual-e5-small (dense) | tydi_ar | 921 | 86.3 | 95.1 | 97.0 | 90.2 | 91.8 |
| multilingual-e5-small (dense) | tydi_en | 440 | 88.6 | 98.0 | 98.9 | 92.9 | 94.4 |
| multilingual-e5-small (dense) | xquad_ar2ar | 1190 | 69.2 | 86.9 | 91.1 | 76.9 | 80.4 |
| multilingual-e5-small (dense) | xquad_ar2en | 1190 | 31.7 | 54.0 | 62.9 | 41.4 | 46.5 |
| multilingual-e5-small (dense) | xquad_en2ar | 1190 | 51.4 | 75.5 | 82.7 | 61.7 | 66.8 |
| multilingual-e5-small (dense) | xquad_en2en | 1190 | 74.7 | 93.3 | 96.2 | 82.7 | 86.0 |
| multilingual-e5-small, Arabic-normalized input | arcd_ar | 702 | 48.9 | 80.1 | 84.8 | 62.5 | 68.0 |
| multilingual-e5-small, Arabic-normalized input | tydi_ar | 921 | 83.6 | 93.9 | 95.8 | 88.2 | 90.0 |
| multilingual-e5-small, Arabic-normalized input | tydi_en | 440 | 88.6 | 98.0 | 98.9 | 92.9 | 94.4 |
| multilingual-e5-small, Arabic-normalized input | xquad_ar2ar | 1190 | 63.4 | 82.9 | 87.2 | 71.9 | 75.6 |
| multilingual-e5-small, Arabic-normalized input | xquad_ar2en | 1190 | 26.4 | 46.6 | 53.8 | 34.8 | 39.3 |
| multilingual-e5-small, Arabic-normalized input | xquad_en2ar | 1190 | 50.9 | 76.1 | 82.5 | 61.7 | 66.8 |
| multilingual-e5-small, Arabic-normalized input | xquad_en2en | 1190 | 74.7 | 93.3 | 96.2 | 82.7 | 86.0 |
| bge-m3 (dense) | arcd_ar | 702 | 56.8 | 82.6 | 86.9 | 68.5 | 73.1 |
| bge-m3 (dense) | tydi_ar | 921 | 89.9 | 95.8 | 96.9 | 92.5 | 93.6 |
| bge-m3 (dense) | tydi_en | 440 | 91.6 | 99.1 | 99.5 | 95.0 | 96.2 |
| bge-m3 (dense) | xquad_ar2ar | 1190 | 72.6 | 88.1 | 91.9 | 79.4 | 82.5 |
| bge-m3 (dense) | xquad_ar2en | 1190 | 52.4 | 77.8 | 84.7 | 63.3 | 68.5 |
| bge-m3 (dense) | xquad_en2ar | 1190 | 59.2 | 79.1 | 84.4 | 67.8 | 71.8 |
| bge-m3 (dense) | xquad_en2en | 1190 | 72.7 | 91.1 | 94.5 | 80.6 | 84.0 |
| bge-m3, Arabic-normalized input | arcd_ar | 702 | 55.4 | 83.5 | 86.9 | 68.1 | 72.8 |
| bge-m3, Arabic-normalized input | tydi_ar | 921 | 89.5 | 95.3 | 96.3 | 92.1 | 93.1 |
| bge-m3, Arabic-normalized input | tydi_en | 440 | 91.6 | 99.1 | 99.5 | 95.0 | 96.2 |
| bge-m3, Arabic-normalized input | xquad_ar2ar | 1190 | 70.3 | 85.7 | 89.0 | 76.8 | 79.7 |
| bge-m3, Arabic-normalized input | xquad_ar2en | 1190 | 47.8 | 74.8 | 81.8 | 59.0 | 64.6 |
| bge-m3, Arabic-normalized input | xquad_en2ar | 1190 | 55.4 | 76.1 | 81.3 | 64.3 | 68.4 |
| bge-m3, Arabic-normalized input | xquad_en2en | 1190 | 72.6 | 91.1 | 94.5 | 80.6 | 84.0 |
| RRF e5-small + BM25(stem) | arcd_ar | 702 | 55.7 | 86.8 | 90.2 | 69.5 | 74.7 |
| RRF e5-small + BM25(stem) | tydi_ar | 921 | 80.5 | 92.1 | 94.6 | 85.7 | 87.9 |
| RRF e5-small + BM25(stem) | tydi_en | 440 | 80.5 | 92.5 | 95.5 | 85.9 | 88.3 |
| RRF e5-small + BM25(stem) | xquad_ar2ar | 1190 | 80.4 | 93.4 | 95.8 | 86.1 | 88.5 |
| RRF e5-small + BM25(stem) | xquad_ar2en | 1190 | 25.0 | 52.3 | 63.1 | 36.9 | 43.2 |
| RRF e5-small + BM25(stem) | xquad_en2ar | 1190 | 10.6 | 52.0 | 73.3 | 26.9 | 37.8 |
| RRF e5-small + BM25(stem) | xquad_en2en | 1190 | 80.6 | 95.2 | 97.1 | 86.9 | 89.5 |
| RRF e5-small + BM25(stem), language-gated | arcd_ar | 702 | 55.7 | 86.8 | 90.2 | 69.5 | 74.7 |
| RRF e5-small + BM25(stem), language-gated | tydi_ar | 921 | 80.5 | 92.1 | 94.6 | 85.7 | 87.9 |
| RRF e5-small + BM25(stem), language-gated | tydi_en | 440 | 80.5 | 92.5 | 95.5 | 85.9 | 88.3 |
| RRF e5-small + BM25(stem), language-gated | xquad_ar2ar | 1190 | 80.4 | 93.4 | 95.8 | 86.1 | 88.5 |
| RRF e5-small + BM25(stem), language-gated | xquad_ar2en | 1190 | 31.7 | 54.0 | 62.9 | 41.4 | 46.5 |
| RRF e5-small + BM25(stem), language-gated | xquad_en2ar | 1190 | 51.4 | 75.5 | 82.7 | 61.7 | 66.8 |
| RRF e5-small + BM25(stem), language-gated | xquad_en2en | 1190 | 80.6 | 95.2 | 97.1 | 86.9 | 89.5 |
| RRF bge-m3 + BM25(raw) | arcd_ar | 702 | 52.8 | 82.9 | 87.7 | 65.8 | 71.2 |
| RRF bge-m3 + BM25(raw) | tydi_ar | 921 | 76.0 | 89.1 | 92.6 | 81.8 | 84.4 |
| RRF bge-m3 + BM25(raw) | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(raw) | xquad_ar2ar | 1190 | 79.5 | 92.3 | 95.0 | 85.2 | 87.6 |
| RRF bge-m3 + BM25(raw) | xquad_ar2en | 1190 | 41.4 | 75.7 | 83.6 | 56.1 | 62.8 |
| RRF bge-m3 + BM25(raw) | xquad_en2ar | 1190 | 16.2 | 63.3 | 78.5 | 34.8 | 45.3 |
| RRF bge-m3 + BM25(raw) | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |
| RRF bge-m3 + BM25(norm) | arcd_ar | 702 | 58.4 | 85.9 | 89.9 | 70.6 | 75.3 |
| RRF bge-m3 + BM25(norm) | tydi_ar | 921 | 77.9 | 90.6 | 93.3 | 83.4 | 85.8 |
| RRF bge-m3 + BM25(norm) | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(norm) | xquad_ar2ar | 1190 | 79.6 | 92.2 | 95.1 | 85.2 | 87.6 |
| RRF bge-m3 + BM25(norm) | xquad_ar2en | 1190 | 41.1 | 75.7 | 83.6 | 55.8 | 62.6 |
| RRF bge-m3 + BM25(norm) | xquad_en2ar | 1190 | 16.2 | 63.3 | 78.5 | 34.8 | 45.3 |
| RRF bge-m3 + BM25(norm) | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |
| RRF bge-m3 + BM25(stem) | arcd_ar | 702 | 62.0 | 88.7 | 91.7 | 73.8 | 78.3 |
| RRF bge-m3 + BM25(stem) | tydi_ar | 921 | 81.8 | 92.7 | 95.3 | 86.7 | 88.8 |
| RRF bge-m3 + BM25(stem) | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(stem) | xquad_ar2ar | 1190 | 82.4 | 93.6 | 95.8 | 87.5 | 89.5 |
| RRF bge-m3 + BM25(stem) | xquad_ar2en | 1190 | 38.8 | 75.6 | 83.6 | 54.5 | 61.6 |
| RRF bge-m3 + BM25(stem) | xquad_en2ar | 1190 | 16.2 | 63.4 | 78.5 | 34.8 | 45.3 |
| RRF bge-m3 + BM25(stem) | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |
| RRF bge-m3 + BM25(stem), language-gated | arcd_ar | 702 | 62.0 | 88.7 | 91.7 | 73.8 | 78.3 |
| RRF bge-m3 + BM25(stem), language-gated | tydi_ar | 921 | 81.8 | 92.7 | 95.3 | 86.7 | 88.8 |
| RRF bge-m3 + BM25(stem), language-gated | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_ar2ar | 1190 | 82.4 | 93.6 | 95.8 | 87.5 | 89.5 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_ar2en | 1190 | 52.4 | 77.8 | 84.7 | 63.3 | 68.5 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_en2ar | 1190 | 59.2 | 79.1 | 84.4 | 67.8 | 71.8 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |

## Appendix D. Sensitivity of test results to near-duplicate leakage

Excluded test queries (gold passage near-duplicated in train): 14 of 6823.

| System | macro MRR@10, all queries | macro MRR@10, excluding flagged | difference |
|---|---:|---:|---:|
| e5-small baseline | 66.92 | 66.93 | +0.01 |
| bge-m3 baseline | 74.78 | 74.77 | -0.00 |
| e5s_inb_e1_s13 | 70.69 | 70.71 | +0.01 |
| e5s_hard1_e1_s13 | 72.86 | 72.87 | +0.01 |
| e5s_hard1_xl_e1_s13 | 75.74 | 75.76 | +0.02 |

## Appendix E. Raw audit / manifest

```json
{
  "gold_outside_pool": 0,
  "duplicate_pids_across_pools": 0,
  "questions_in_multiple_splits": 1,
  "question_split_pairs": {
    "dev+train": "1"
  },
  "title_overlap_all_passages(incl. distractors)": {
    "train&test": 10,
    "train&dev": 16,
    "dev&test": 0
  },
  "title_overlap_gold_passages_only": {
    "train&test": 0,
    "train&dev": 9,
    "dev&test": 0
  },
  "note_title_overlap": "Overlap with distractor-only articles comes from the per-source title hash (the same Wikipedia article can be a TyDi distractor in the test pool and an ARCD train article); identical paragraphs are removed by text key. Gold test articles are excluded from train/dev by construction (see build_data).",
  "near_duplicate_passages(6-gram containment)": {
    "test_vs_train": {
      "passages_checked": 9817,
      "containment>=0.5": 26,
      "containment>=0.8": 14,
      "gold_queries_affected": 14,
      "examples": [
        {
          "a": "ar-ae77a4b50516",
          "b": "ar-80de3bf9c6d3",
          "frac": 0.65
        },
        {
          "a": "ar-49ef52e929f3",
          "b": "ar-e9e040494650",
          "frac": 0.6
        },
        {
          "a": "en-5f1251d802d5",
          "b": "en-e4f96fe24c8f",
          "frac": 0.9
        }
      ]
    },
    "dev_vs_train": {
      "passages_checked": 4037,
      "containment>=0.5": 23,
      "containment>=0.8": 13,
      "gold_queries_affected": 50,
      "examples": [
        {
          "a": "ar-0adca5c65f83",
          "b": "ar-38cd2f0ce34f",
          "frac": 0.93
        },
        {
          "a": "ar-dc45f2a62f0a",
          "b": "ar-025454a9571d",
          "frac": 0.64
        },
        {
          "a": "ar-4b65214810c6",
          "b": "ar-9277e582e03e",
          "frac": 0.81
        }
      ]
    },
    "test_vs_dev": {
      "passages_checked": 9817,
      "containment>=0.5": 6,
      "containment>=0.8": 3,
      "gold_queries_affected": 6,
      "examples": [
        {
          "a": "ar-aca2aa84a564",
          "b": "ar-6db3ff04802e",
          "frac": 0.7
        },
        {
          "a": "ar-5267b6445f78",
          "b": "ar-6db3ff04802e",
          "frac": 0.71
        },
        {
          "a": "ar-98c25479c247",
          "b": "ar-3d280a0c7715",
          "frac": 0.55
        }
      ]
    }
  },
  "near_duplicate_questions": {
    "test": {
      "queries_checked": 6823,
      "near_dup_of_a_train_question(J>=0.8)": 8,
      "examples": [
        [
          "ما هو الحمض النووي الريبوزي الصغير النووي ؟",
          "ما هو الحمض النووي الريبوزي؟"
        ],
        [
          "متى تأسست حركة حماس؟",
          "متى تأسست حركة حماس العراق؟"
        ],
        [
          "Where was Zheng He born?",
          "Where was he born?"
        ]
      ]
    },
    "dev": {
      "queries_checked": 3163,
      "near_dup_of_a_train_question(J>=0.8)": 9,
      "examples": [
        [
          "من هو جون ستيوارت ؟",
          "من هو جون ستيوارت مل؟"
        ],
        [
          "ماهي أعلى رتبة عسكرية في سلم الرتب العسكرية الإيراني؟",
          "ماهي أعلى رتبة عسكرية في سلم الرتب العسكرية المصري؟"
        ],
        [
          "ما هو الحمض النووي المؤتلف؟",
          "ما هو الحمض النووي؟"
        ]
      ]
    }
  },
  "translated_queries_with_foreign_source_split": 0,
  "translated_queries": {
    "dev": 500,
    "train": 4499
  }
}
```

```json
{
 "seed": 13,
 "max_q_per_ctx": {
  "squad_train": 2
 },
 "licenses": {
  "tydi": "Apache-2.0 (google-research-datasets/tydiqa, secondary_task = GoldP)",
  "arcd": "MIT (hsseinmz/arcd)",
  "squad": "CC BY-SA 4.0 (rajpurkar/squad v1.1)",
  "xquad": "CC BY-SA 4.0 (google/xquad)"
 },
 "dropped": {
  "title_in_test_tydi_ar_train": 1094,
  "title_in_test_tydi_en_train": 6,
  "title_in_test_arcd_train": 306,
  "passage_in_other_pool_arcd_train": 12,
  "title_in_test_squad_train": 1610,
  "passage_in_other_pool_tydi_ar_train": 4,
  "dup_question_train_tydi_ar": 6,
  "dup_question_train_arcd_ar": 3,
  "dup_question_train_squad_en": 1
 },
 "passages_by_pool_lang": {
  "dev/ar": 1982,
  "dev/en": 2055,
  "test/ar": 4325,
  "test/en": 5492,
  "train/ar": 5425,
  "train/en": 5227
 },
 "queries_by_slice": {
  "dev/dev_arcd_ar": 111,
  "dev/dev_squad_en": 1661,
  "dev/dev_tydi_ar": 1037,
  "dev/dev_tydi_en": 354,
  "dev/dev_xl_ar2en": 300,
  "dev/dev_xl_en2ar": 200,
  "test/arcd_ar": 702,
  "test/tydi_ar": 921,
  "test/tydi_en": 440,
  "test/xquad_ar2ar": 1190,
  "test/xquad_ar2en": 1190,
  "test/xquad_en2ar": 1190,
  "test/xquad_en2en": 1190,
  "train/train_arcd_ar": 414,
  "train/train_squad_en": 6653,
  "train/train_tydi_ar": 7328,
  "train/train_tydi_en": 2024,
  "train/train_xl_ar2en": 3000,
  "train/train_xl_en2ar": 1499
 },
 "xling_candidates": {
  "dev/ar2en": 200,
  "dev/en2ar": 300,
  "train/ar2en": 1500,
  "train/en2ar": 3000
 },
 "translation_stats": {
  "candidates": 5000,
  "missing": 0,
  "invalid": 0,
  "dropped_duplicate_of_higher_priority_split": 1
 },
 "queries_by_split": {
  "dev": 3663,
  "test": 6823,
  "train": 20918
 },
 "passages_total": 24506,
 "passages_by_pool_kind": {
  "dev/background": 2054,
  "dev/gold": 1983,
  "test/background": 7834,
  "test/gold": 1983,
  "train/background": 1,
  "train/gold": 10651
 }
}
```
