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

{{T:HEADLINE - TEST}}

(Reranked rows use embedder seed 13 and a single reranker training run; every reranker scores the first stage's top-50.)

Same systems, Hit@1 (the fraction of queries whose gold passage is ranked first):

{{T:HEADLINE_H1}}

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

{{T:Baselines on TEST - MRR@10}}

{{T:Baselines on TEST - HIT@1}}

{{T:Baselines on TEST - HIT@10}}

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

{{T:Orthographic-noise robustness}}

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

{{T:EXP A_negatives - TEST MRR@10}}

{{T:EXP A_negatives - TEST Hit@1}}

**Takeaways.**
* In-batch negatives alone already help (+3.9 macro over the baseline, mostly on the Arabic and cross-lingual directions), but **English gets worse** (87.8 -> 86.4): the model forgets a little of what it knew.
* **One mined hard negative per query adds +2.2 macro** (70.8 +- 0.2 -> 73.0 +- 0.2, 3 seeds; paired bootstrap on seed 13: +1.8 / +1.1 / +3.3 / +2.4 points on the four directions, all CIs exclude 0) and repairs English (87.7).
* **Random negatives give nothing** (70.9 vs 70.8): the gain comes from hardness, not from the extra column.
* **The false-negative filter matters, and not uniformly.** Without a filter: 71.8 macro (-1.2). The effect is direction-dependent: en->ar *improves* without the filter (75.4 vs 74.1) while ar->en collapses (45.1 vs 49.4, CI of the difference excludes 0) - an unfiltered false negative on an Arabic-query/English-passage pair is the kind of label noise that hurts the very direction we care about.
  Stricter filters lose points too (margin 0.05: 71.5; dropping same-article candidates: 71.5; both pad many queries with random negatives and throw away informative near-misses). Skipping ranks 1-9 (candidates from ranks 10-30 only, margin 0.02) gave 71.6 (-1.3 vs the default): the hardest negatives are mostly true negatives, and discarding them costs more than the extra protection saves once the margin filter has removed the dangerous ones.
* **Who mines matters little:** mining with the stronger `bge-m3` instead of the baseline e5-small gave 73.1 vs 73.0 (single seed, within noise); 3 negatives per query gave 72.8 (also within noise, and 2x the cost: 15.9 GB peak memory at batch 64, 688 s vs 305 s).

### 4.2 (b) Data size

{{T:EXP B_datasize - TEST MRR@10}}

With mined negatives, 25% / 50% / 100% of the queries give macro 71.7 / 72.5 / 73.0 (baseline 66.9): about 80% of the gain from 4k queries and one minute of training, and a flattening curve. Without hard negatives the curve is flatter still (70.2 / 70.6 / 70.8) - negatives, not volume, carry the signal.
Training 25% of the data for 4 epochs (the same 256 optimizer steps) recovers little (hard: 71.8; in-batch: 70.5), so the missing gain is data diversity, not step count.

### 4.3 (c) Epochs, and overfitting on dev

{{T:EXP C_epochs - TEST MRR@10}}

{{T:Dev curves}}

Dev MRR@10 peaks after the **first** epoch in both recipes (hard-negative: 83.3 -> 82.8 -> 83.0; in-batch: 82.0 -> 81.0 -> 81.2; mono macro = ar->ar, en->en) and the final 3-epoch test numbers are no better than 1 epoch (72.4 vs 73.0; 70.8 vs 70.8).
The final logged training loss keeps falling (hard: ~0.20 after 1 epoch -> 0.10 after 3; in-batch: ~0.08 -> 0.03) while dev does not move: the model is memorizing its few thousand training passages. With data this small, early stopping on dev at one epoch is the right default.

### 4.4 (d) Hyper-parameters and batch size

{{T:EXP D_hyperparams - TEST MRR@10}}

The recipe is **insensitive** to the loss scale (20 vs 50), to lr between 3e-5 and 6e-5 (1e-5 is 1 point lower: not enough movement in 256 steps), and to batch size between 32 and 128 (72.8 / 72.9 / 72.3). Two sub-findings: (i) a 512-example batch via CachedMNRL
(the memory stays at 5.4 GB for batch 128; GradCache does what it promises) *under-trains* when the epoch count is fixed - only 32 optimizer steps: 70.3, i.e. 2.6 points worse - bigger batches need more epochs or a higher lr; (ii) at batch 64 with 3 negatives per query plain MNRL peaked at 15.9 GB, whereas the cached version needs 5.4 GB for the same gradient.

### 4.5 (e) Cross-lingual training data (the ar->en question)

{{T:EXP E_crosslingual - TEST MRR@10}}

{{T:EXP E_crosslingual - TEST Hit@10}}

* **Translated cross-lingual pairs are the biggest lever for the cross-lingual directions**: with mined negatives ar->en 49.4 -> 54.7 +- 0.6 and en->ar 74.1 -> 80.4 +- 0.4 (paired bootstrap on seed 13: +5.9 and +6.4 points, CIs exclude 0), at the price of -0.5 on ar->ar and -0.2 on en->en (not significant). For in-batch-only the same data gives +6.6 / +9.0.
* Hit@10 for ar->en rises from 62.9 (baseline) to 79.6; the Hit@50 ceiling for a downstream reranker rises from 83.1 to 91.3.
* Final recipe (hard negatives, margin 0.02, cross-lingual pairs, 1 epoch) vs the 568M `bge-m3` baseline: ar->ar +0.0 [-1.0, +1.1], en->en -0.5 [-1.6, +0.5] (tied), en->ar **+12.6** [+10.5, +14.7], ar->en **-8.2** [-10.4, -6.2]. The training set had twice as many ar-query/en-passage pairs (3,000) as en-query/ar-passage pairs (1,499; translation throughput limited the latter), yet the two directions gained similar amounts (+5.3 ar->en, +6.3 en->ar): ar->en remains the hard direction for a 118M model, while bge-m3, trained with large-scale cross-lingual objectives, is 8 points ahead on it.
* **Choosing the final recipe.** Alternatives tried on top of the same data: 3 cached negatives (dev macro over the four directions 77.4, test 75.6), 2 epochs (dev 78.1, test 75.8) and bge-m3-mined negatives (dev 77.9, test 75.7), against 77.5 on dev and 75.7 on test for the final recipe. By dev alone the 2-epoch variant would have been picked (+0.6, driven by the machine-translated ar->en / en->ar dev slices); we kept the pre-declared 1-epoch recipe because it is the only one with three seeds and the test difference is 0.1 points. Honest flag: the dev set is translated the same way as the training data, so it slightly over-rewards extra cross-lingual training.

### 4.6 Paired bootstrap summary (seed 13, test MRR@10, points [95% CI])

{{T:Paired bootstrap, TEST MRR@10}}

### 4.7 Training cost

Peak GPU memory (PyTorch allocator) and wall-clock on the GB10 (other jobs ran concurrently - times are noisy, memory is exact): in-batch 5.6 GB / ~5 min; 1 hard negative 9.0 GB / ~5-8 min; 3 hard negatives (plain MNRL) 15.9 GB / 11 min; cached losses 5.2-5.4 GB regardless of batch size; cross-lingual run (20.9k examples) ~10 min. Full table: appendix.

{{T:Training cost}}

## 5. Cross-encoder reranker (retrieve top-50 -> rerank -> top-10)

Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M parameters, multilingual MiniLM, trained on mMARCO incl. Arabic). Fine-tuning (`train_reranker.py`): pointwise binary cross-entropy, positives up-weighted x4,
1 gold + 4 negatives per query for the same 20,918 training queries (incl. the translated cross-lingual ones), 1 epoch (3,268 steps, batch 32, lr 2e-5, max 256 tokens), negatives mined by the **baseline** e5-small (rank 1-30, margin 0.02) because that is the distribution a reranker sees from a first stage.
Every reranker below rescored the first stage's **top-50** and the metrics are computed on the re-ordered list (candidates outside the top-50 keep their first-stage order and never reach the top-10 unless the reranker's list is short).

{{T:RERANK - TEST MRR@10}}

{{T:RERANK - TEST HIT@1}}

{{T:RERANK - TEST HIT@10}}

{{T:RERANK - candidate recall}}

{{T:RERANK - paired bootstrap}}

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

{{T:RERANK - strong 568M reference reranker}}

### 5.1 Latency (per query, one batch of N candidates, tokenization included)

{{T:Per-query latency}}

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

{{T:EXP A_negatives - DEV MRR@10}}

{{T:EXP E_crosslingual - DEV MRR@10}}

## Appendix B. Remaining test metrics for the embedders

{{T:EXP E_crosslingual - TEST Hit@1}}

{{T:EXP A_negatives - TEST Hit@10}}

{{T:Fine-tuned embedders - TEST nDCG@10}}

## Appendix C. Every slice, every metric, every baseline

{{T:Baselines on TEST - every slice}}

## Appendix D. Sensitivity of test results to near-duplicate leakage

{{T:Sensitivity to near-duplicate leakage}}

## Appendix E. Raw audit / manifest

{{T:Leakage audit}}

{{T:Data manifest}}
