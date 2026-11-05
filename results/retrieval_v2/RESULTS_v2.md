# v2 embedders - results (Arabic query -> English passage, and what else moved)

Measured on the DGX Spark (GB10, shared with llama-servers and another agent's MTEB runs) between 2026-11-04 01:30 and 2026-11-05 07:30 (Spark clock, UTC+3). Nothing was uploaded to Hugging Face and nothing was pushed to GitHub; model licences are **TBD** (MIT or CC BY-SA 4.0, the user decides).
All numbers are x100. Retrieval on the v1 benchmark is **MRR@10**; MTEB(ara) tables use the task main scores (nDCG@10, accuracy, AP, Spearman, V-measure, F1 ...) averaged over subsets and splits as the leaderboard does.
Three kinds of evidence are kept apart: **DEV / validation suite** (everything that selected a recipe), **TEST** (the v1 benchmark test split, touched only for final candidates), and a **confirmatory replication** (two training seeds that were never used in any selection step, trained after the recipe was frozen).
Code: `lab/retrieval/v2/` on the Spark = `src/arretrieval/retrieval/v2/` in this repository (new files only, v1 untouched). In this repository the Spark's `runs/retrieval/v2/results/` is `results/retrieval_v2/` (small files only: no stored rankings, no weights, no dataset text) and the queue scripts are in `scripts/retrieval_v2/`; the model cards are in `docs/retrieval/model_cards/`. Data and protocol are v1's (`RESULTS.md`) with the changes in section 2. The pre-registration trail with timestamps is section 12; the full append-only log is `PROGRESS.md`.

## 0. TL;DR

* **Arabic query -> English passage (TEST MRR@10, the v1 benchmark).** v2-m3 (bge-m3 fine-tuned, then weight-interpolated with bge-m3: 60% original + 40% fine-tuned, WiSE-FT) **71.2** vs bge-m3 63.3 (paired bootstrap +7.8 [+6.7, +8.9]) and v1 54.7 (3 seeds). Development seeds 13/14/15 average 70.9; the two **confirmatory** seeds (16, 17; recipe frozen before they were trained) 71.1 and 70.3. Macro over the four directions: 82.2 vs bge-m3 74.8.
* **The ≤300M models** (distilled from v2-m3, KD only, 3-seed soups): v2-base (278M) ar->en **66.6** (e5-base 49.9; also above bge-m3's 63.3), v2-small (118M) **59.7** (e5-small 41.4, v1 54.7; it does NOT reach bge-m3). Their macros: 80.2 and 76.9 (bge-m3 74.8).
* **MTEB(ara, v1), Mean (Task) over 19 tasks.** v2-m3 **72.29** vs bge-m3 71.98: **+0.31, task-bootstrap 95% CI [-0.34, +0.93], 10 wins / 8 losses / 1 tie, sign test p=0.81 - numerically first, statistically a tie.** The two confirmatory seeds reproduce it (72.28, 72.27). v2-base 68.91 vs e5-base 67.89 (+1.02, CI [-0.52, +2.37], 15/4/0, sign test p=0.019); v2-small 66.20 vs e5-small 65.63 (+0.57, CI [-1.24, +2.19]) and vs v1 65.00 (+1.19, CI [+0.43, +1.92] *).
* **What it cost (guard tasks, MTEB(ara) nDCG/AP x100, v2-m3 vs bge-m3):** MIRACL-HN 77.9 vs 78.9, NamaaMrTydi 76.0 vs 79.4, SemRel24STS 60.9 vs 59.1. The drift on the Arabic-only reranking task (NamaaMrTydi, -3.4) is **not** fixed; the students lose much more there (v2-base 68.2 vs e5-base 77.1, v2-small 68.9 vs e5-small 79.2).
* **With the fine-tuned cross-encoder on top (top-50 -> `rr_ft_bce_s13`)** the first-stage gap closes: ar->en 75.3 (v2-m3) vs 75.3 (bge-m3), macro 84.3 vs 84.5; for the small models the first stage still matters: v2-base 74.4 vs e5-base 71.3, v2-small 73.4 vs e5-small 68.9.

**Bottom line.** The known weakness is fixed: Arabic-query -> English-passage retrieval goes from 54.7 (v1) to 71.2 with v2-m3, above bge-m3's 63.3, with the other three directions up as well, and the result replicates on two untouched seeds. The same recipe distilled into the 278M and 118M models also lifts ar->en by +16.7 / +18.3 over their off-the-shelf bases. What is **not** shown: a statistically clear lead of v2-m3 over bge-m3 on MTEB(ara) (+0.31, CI includes 0), a win on the ar-en cross-lingual subsets over multilingual-e5-large, or the absence of forgetting on the Arabic-only reranking task (NamaaMrTydi -3.4 for v2-m3, about -9 to -10 for the students). Section 1 lists each bar item with its status.

## 1. Scorecard against the S-tier bar (`S_TIER.md`)

| bar item | measured | status |
|---|---|---|
| 1. ar->en >= bge-m3 with macro > bge-m3 (TEST) | v2-m3 71.2 vs 63.3 (+7.8 [+6.7, +8.9]); macro 82.2 vs 74.8 | **met** (v2-m3, v2-base); v2-small misses ar->en (59.7 < 63.3) but beats e5-small and v1 |
| 2a. no forgetting on MIRACL-HN / NamaaMrTydi / SemRel24 (within ~1 pt of the base) | MIRACL-HN -1.0, NamaaMrTydi -3.4, SemRel24 +1.8 | **partly**: MIRACL-HN at the edge of the 1-point tolerance, SemRel24 up, **NamaaMrTydi missed (-3.4)** |
| 2b. MTEB(ara) Mean (Task) > base | 72.29 vs 71.98 (+0.31, CI [-0.34, +0.93]) | numerically yes, **not statistically distinguishable**; reproduced by both confirmatory seeds |
| 3a. #1 on MTEB(ara) among <=300M models | v2-base 68.91 vs e5-base 67.89 (the best other <=300M model of the 10-model local board) | **met** for v2-base; v2-small is above e5-small and v1 but below v2-base and e5-base |
| 3b. #1 on the ar-en cross-lingual subsets | v2-m3 75.0, bge-m3 73.3, **e5-large 75.3**; v2-base 74.2 vs e5-base 69.1 | **met within <=300M** (v2-base); **NOT met overall**: multilingual-e5-large (560M) is 0.3 higher than v2-m3 |
| 4. contamination audit + licences documented | section 2 (audit table, licence table), section 8 (exact-match leak check) | **met with one disclosed exception**: 9 kept training rows equal MIRACL-HN dev queries exactly (section 8); the model licence is TBD (user decision: MIT or CC BY-SA 4.0) |
| 5. full-pipeline numbers with the fine-tuned reranker | section 9 | **reported**; with the reranker v2-m3 and bge-m3 are level |

## 2. Data, protocol and contamination (what changed vs v1)

* **More translated cross-lingual queries.** v1 had 4,499 Gemma-translated training queries. v2 translated 22,313 more questions with the same prompt (Gemma-4-26B-A4B on the shared llama-server, batches of 20, concurrency 2, thinking off): 20,999 new training queries (14,940 + 6,059) and 1,300 more DEV queries. Merged with v1's data this is 41,917 training queries; the contamination filter below drops 2,540, leaving **39,377 examples** (the final training set of every final model: TyDi-ar 6,354, TyDi-en 2,018, SQuAD-en 6,501, ARCD 373, v1 translations 4,245, new translations 19,886).
* **Extended DEV.** The translated dev slices grew from 300 to 1,000 (ar->en) and from 200 to 798 (en->ar) queries; v1's dev numbers were recomputed on it.
* **Contamination filter (applies to every v2 final run).** `SELECTED_TASKS.md` lists MIRACL, MLQA, XPQA, Belebele, Mintaka, MKQA and XQuAD as evaluation sets. An overlap audit (`audit_ood.py`) found that 408 of the 1,000 MIRACL-ar dev queries are near-duplicates (token Jaccard >= 0.8) of questions in v1's TyDi-QA-ar training pool (and 2,299 MIRACL corpus passages overlap with training passages). Against the *full* TyDi-QA-ar train set, 893 of the 1,000 are exact copies after normalisation (an exact-match check added on 2026-11-05). v2 therefore drops every training question that is a near-duplicate of a query of any of those sets (translated questions via their source question) and every example whose positive passage overlaps (>= 50% of its word 8-grams) a relevant passage of those sets: 2,540 queries (985 near-duplicate questions, 2,226 positives overlapping an evaluation positive; some meet both). By the near-duplicate detector the MIRACL count drops to 0 (table), but that detector has a blind spot (see below and section 8): **9 kept training rows still equal a MIRACL-HN dev query exactly**. The early small-model ablations in section 3.1 were trained BEFORE the audit on the unfiltered set; every v2-m3 / v2-base / v2-small run uses the filtered set.
* **Not used:** XQuAD, MLQA, XPQA, Belebele, Mintaka, MKQA, WebFAQ and the STS / bitext sets (evaluation); mMARCO / MS MARCO (non-commercial terms; mMARCO-ar is used only as an *evaluation proxy*); Natural Questions (its 10k MKQA subset is an MTEB(ara) task); Tatoeba / OPUS parallel sentences (MTEB(ara) bitext tasks). A third translation batch (23,418 SQuAD-train questions from 11,819 previously unused paragraphs, "data v3") was built and translated but **no model was trained on it**. **Residual leak, disclosed.** The near-duplicate index skips words with more than 300 postings, so short questions made of common words get no candidates and exact copies slip through. A read-only exact-match check after training found, among the kept training rows, **9 rows (8 distinct questions, all slice `train_tydi_ar`) that equal a MIRACL-HN.v2 ar dev query after normalisation** (0.8% of its 998 distinct normalised queries) and 0 for NamaaMrTydi; 7 translated English versions of them (slice `train_xl2_en2ar`) are in the kept training data as well (section 8.1). Nothing was retrained; the impact on MIRACL-HN is bounded by 8 of 1,000 queries and the v2 scores on that task are below bge-m3's anyway. MIRACL-ar *train* (1,515 queries after filtering) was used only in the replay experiment of section 3.2 (not in any final model) and as the 438-query held-out selection slice `devM`.
* Everything else is v1: article-level split, pools searched per language, bf16, 1 mined hard negative per query (baseline e5-small, ranks 1-30, 0.02 margin), max 256 tokens. v2-m3 changes only: base = bge-m3, lr 1e-5, scale 50, CachedMNRL (mini-batch 16, effective batch 64), 1 epoch.

| eval set (split) | corpus | passages >=50% 8-gram overlap with training passages (v1 data -> v2 filtered) | queries | queries near-duplicate of a training question (v1 data -> v2 filtered) |
|---|---:|---:|---:|---:|
| XPQARetrieval/ara-eng/test | 1,495 | 0 -> 0 | 742 | 0 -> 0 |
| XPQARetrieval/eng-ara/test | 1,533 | 0 -> 0 | 750 | 1 -> 0 |
| XPQARetrieval/ara-ara/test | 1,495 | 0 -> 0 | 750 | 1 -> 0 |
| MLQARetrieval/ara-eng/validation | 439 | 8 -> 2 | 517 | 1 -> 0 |
| MLQARetrieval/ara-eng/test | 4,646 | 56 -> 8 | 5,332 | 6 -> 0 |
| MLQARetrieval/eng-ara/validation | 438 | 1 -> 0 | 517 | 1 -> 0 |
| MLQARetrieval/eng-ara/test | 4,606 | 48 -> 3 | 5,333 | 5 -> 1 |
| MLQARetrieval/ara-ara/validation | 439 | 8 -> 2 | 517 | 1 -> 0 |
| MLQARetrieval/ara-ara/test | 4,646 | 56 -> 8 | 5,333 | 5 -> 1 |
| BelebeleRetrieval/arb_Arab-eng_Latn/test | 488 | 0 -> 0 | 900 | 0 -> 0 |
| BelebeleRetrieval/eng_Latn-arb_Arab/test | 488 | 0 -> 0 | 900 | 1 -> 0 |
| BelebeleRetrieval/arb_Arab-arb_Arab/test | 488 | 0 -> 0 | 900 | 1 -> 0 |
| MintakaRetrieval/ar/test | 1,491 | 0 -> 0 | 2,203 | 5 -> 1 |
| MKQARetrieval/ar/train | 6,658 | 0 -> 0 | 6,747 | 4 -> 0 |
| XQuADRetrieval/ar/validation | 240 | 0 -> 0 | 1,186 | 1 -> 0 |
| MIRACLRetrievalHardNegatives.v2/ar/dev | 192,103 | 2299 -> 1814 | 1,000 | 408 -> 0 |

### 2.1 Training data and licences

| source | what it is used for | licence | note |
|---|---|---|---|
| TyDi QA GoldP (ar, en) | train (questions + gold paragraphs) | Apache-2.0 | v1 data; train articles only |
| ARCD | train (373 queries after filtering) | MIT | train split only (validation = v1 test) |
| SQuAD v1.1 | train questions + paragraphs (`train` only; XQuAD comes from SQuAD *dev*) | CC BY-SA 4.0 | |
| Gemma-4-26B-A4B translations | Arabic / English versions of the questions above (20,999 new training queries + 1,300 dev, on top of v1's 4,499) | machine output of an open model (Gemma terms apply to the model; the questions inherit their source licences) | same prompt as v1; script-validated |
| MIRACL-ar train (`miracl/miracl`, `miracl/miracl-corpus`) | replay experiment (not in any final model) and the 438-query held-out selection slice `devM` | Apache-2.0 | bge-m3 itself was trained on MIRACL train |
| Validation-suite data (section 3.3) | evaluation only, never trained on: XNLI-ar *train* subsample (3,000 pairs), ArSarcasm train split (MIT), Arabic STS-B validation (apache-2.0 per its card) | see source | |
| base models | initialisation / teacher | multilingual-e5-small/base MIT; bge-m3 MIT; mMiniLM cross-encoder (reranker) Apache-2.0 | |
| NOT used | XQuAD, MLQA, XPQA, Belebele, Mintaka, MKQA, WebFAQ, STS/bitext sets; mMARCO / MS MARCO; Natural Questions; OPUS | | |

## 3. How the recipe was chosen (DEV and validation suite only)

### 3.1 The v1-recipe levers on e5-small do not move ar->en (DEV, seed 13)

| run | what changed vs the v1 recipe | ar->ar | en->en | ar->en | en->ar | macro | d macro | devM (MIRACL hold-out) | train s / GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base_e5-small` | multilingual-e5-small, off the shelf | 72.6 | 87.3 | 53.6 | 68.0 | 70.4 | | | |
| `base_bge-m3` | bge-m3, off the shelf (568M) | 75.3 | 87.5 | 73.5 | 76.6 | 78.2 | | | |
| `v1_s13/14/15` | published v1 recipe, 3 seeds (mean) | 78.1 | 87.1 | 65.1 | 77.9 | 77.1 | | | |
| `p1_base_r0` | e5-base, data v1, negs same:1 | 78.0 | 87.4 | 68.7 | 81.4 | 78.9 | +1.9 |  | 545 / 18.3 |
| `p1_bplang` | e5-small, data v1, negs same:1, batches by plang | 78.0 | 87.1 | 65.2 | 77.4 | 76.9 | -0.1 |  | 314 / 8.9 |
| `p1_kd1` | e5-small, data v1, negs same:1, KD w=1.0 T=20.0 | 76.2 | 88.0 | 66.2 | 77.5 | 77.0 | -0.0 |  | 380 / 8.9 |
| `p1_kd3` | e5-small, data v1, negs same:1, KD w=3.0 T=20.0 | 75.3 | 87.4 | 65.0 | 76.4 | 76.0 | -1.0 |  | 387 / 8.9 |
| `p1_kdonly` | e5-small, data v1, negs same:1, KD w=1.0 T=20.0 (KD only) | 74.0 | 86.4 | 63.7 | 74.5 | 74.6 | -2.4 |  | 364 / 8.9 |
| `p1_oth1` | e5-small, data v1, negs same:1,other:1 | 78.4 | 87.0 | 65.7 | 77.3 | 77.1 | +0.1 |  | 477 / 12.4 |
| `p1_oth1m` | e5-small, data v1, negs same:1,other:1 | 78.4 | 86.9 | 65.8 | 77.4 | 77.1 | +0.1 |  | 496 / 12.4 |
| `p1_r0` | e5-small, data v1, negs same:1 | 77.8 | 87.2 | 65.2 | 77.9 | 77.0 | +0.0 |  | 357 / 8.9 |
| `p1_r0m` | e5-small, data v1, negs same:1 | 77.8 | 87.2 | 65.1 | 77.9 | 77.0 | -0.0 |  | 368 / 8.9 |
| `p1_same2` | e5-small, data v1, negs same:2 | 77.7 | 87.2 | 65.8 | 77.8 | 77.1 | +0.1 |  | 482 / 12.4 |
| `p1_sym` | e5-small, data v1, negs same:1, symmetric InfoNCE | 77.2 | 86.7 | 64.5 | 78.0 | 76.6 | -0.4 |  | 329 / 8.9 |
| `p2_b_full` | e5-base, data v2, negs same:1 | 77.9 | 86.7 | 70.6 | 82.1 | 79.3 | +2.3 |  | 870 / 19.0 |
| `p2_s_e2` | e5-small, data v2, negs same:1, 2 ep | 76.6 | 86.8 | 68.4 | 80.8 | 78.1 | +1.1 |  | 956 / 9.3 |
| `p2_s_f25` | e5-small, data v2, xl2 x0.25, negs same:1 | 78.7 | 87.1 | 66.7 | 77.9 | 77.6 | +0.6 |  | 241 / 9.3 |
| `p2_s_f50` | e5-small, data v2, xl2 x0.5, negs same:1 | 77.6 | 86.7 | 66.7 | 79.5 | 77.6 | +0.6 |  | 391 / 9.3 |
| `p2_s_full` | e5-small, data v2, negs same:1 | 76.6 | 87.1 | 67.8 | 79.7 | 77.8 | +0.8 |  | 359 / 9.3 |
| `p2_s_kdxl` | e5-small, data v2, negs same:1, KD w=1.0 T=20.0, KD on xl only | 77.2 | 88.1 | 67.7 | 78.8 | 77.9 | +0.9 |  | 503 / 9.3 |
| `p2_xl2_a5k` | e5-small, data v2, negs same:1 | 78.4 | 86.8 | 66.9 | 79.1 | 77.8 | +0.8 |  | 347 / 9.0 |
| `p3_b_e1_s13` | e5-base, data v2, negs same:1, contamination-filtered | 77.4 | 87.1 | 70.3 | 81.6 | 79.1 | +2.1 |  | 797 / 19.0 |
| `p3_s_e1_s13` | e5-small, data v2, negs same:1, contamination-filtered | 78.3 | 86.8 | 67.4 | 79.2 | 77.9 | +0.9 |  | 265 / 9.3 |
| `p3_s_lr1e5_s13` | e5-small, data v2, negs same:1, lr 1e-05, contamination-filtered | 78.4 | 86.5 | 65.9 | 77.8 | 77.2 | +0.2 |  | 383 / 9.3 |
| `p3_s_lr2e5_s13` | e5-small, data v2, negs same:1, lr 2e-05, contamination-filtered | 79.2 | 86.7 | 67.0 | 78.6 | 77.9 | +0.9 |  | 438 / 9.3 |

`p1_*` = v1 data, `p2_*` = + new translations (unfiltered), `p3_*` = filtered. Single-run noise is about ±0.5 on one direction and ±0.3 on the macro. The DEV ar->en level (65-80) is above TEST (55-74) because the cross-lingual dev slices are machine-translated; section 3.5 shows that offset is the same for fine-tuned and off-the-shelf models.

1. **Mixed-language / more negatives did nothing.** v1 already mines a cross-lingual negative for an Arabic query. Adding an Arabic passage as an extra negative (`p1_oth1`): macro +0.1, ar->en +0.5 (noise); two same-language negatives (`p1_same2`) +0.1; language-homogeneous batches (`p1_bplang`) -0.1; symmetric InfoNCE (`p1_sym`) -0.4.
2. **Distillation from off-the-shelf bge-m3 did not help.** In-batch listwise KL towards bge-m3 with weight 1 / 3 / KD-only: macro 77.0 / 76.0 / 74.6 vs 77.0; the teacher *lowers* ar->ar (it is worse than the student there).
3. **More cross-lingual pairs is the lever that works for the small model, with diminishing returns:** DEV ar->en 65.2 (v1 data) -> 67.8 (+10.6k queries), en->ar 77.9 -> 79.7. On TEST the gain did **not** carry over to the small models (section 3.4).
4. **Two epochs** (`p2_s_e2`): macro +0.3 for twice the cost; lr 1e-5 / 2e-5 vs 3e-5: macro 77.2 / 77.9 / 77.9.
5. **Model size is the other lever:** e5-base on the same data reaches DEV ar->en 70.3-70.6 vs 67.4-67.8 for e5-small, and the fine-tuned bge-m3 is far above both (next table).

### 3.2 Fine-tuning bge-m3 itself, and the attempts to stop it forgetting (DEV, seed 13 unless named)

| run | what | ar->ar | en->en | ar->en | en->ar | macro | devM | mMARCO proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base_bge-m3` | bge-m3 off the shelf (reference) | 75.3 | 87.5 | 73.5 | 76.6 | 78.2 | 80.5 | 81.9 |
| `m3_a_s13` | bge-m3 fine-tuned itself, first run (11.9k translations) | 83.5 | 89.8 | 81.2 | 86.7 | 85.3 |  |  |
| `m3_b_s13` | **bge-m3 fine-tuned, full filtered v2 data (39,377 queries), lr 1e-5, scale 50, CachedMNRL(16), 1 epoch, seed 13 = alpha 1** | 83.0 | 90.3 | 81.0 | 87.1 | 85.3 | 79.1 | 80.8 |
| `m3_b_s14` | same, seed 14 | 83.5 | 89.8 | 80.8 | 87.1 | 85.3 |  |  |
| `m3_b_s15` | same, seed 15 | 82.5 | 89.7 | 80.6 | 86.5 | 84.8 | 78.9 |  |
| `m3_r_s13` | + MIRACL-ar train replay x3 (1,515 queries) | 82.1 | 89.6 | 80.8 | 87.3 | 84.9 | 78.7 |  |
| `m3_p_s13` | + preservation loss towards frozen bge-m3 (w 0.5) | 82.0 | 90.2 | 80.9 | 86.7 | 84.9 | 79.0 |  |
| `m3_fr_s13` | frozen embeddings + bottom 12 layers | 82.6 | 89.9 | 81.1 | 86.3 | 85.0 | 78.8 | 80.5 |
| `m3_lo_s13` | LoRA r=32 (attention + MLP), lr 1e-4 | 84.0 | 90.2 | 81.5 | 86.0 | 85.4 | 79.0 | 80.2 |
| `wm_a75` | WiSE 0.75 of m3_b_s13 towards bge-m3 | 81.8 | 90.0 | 80.5 | 87.3 | 84.9 | 79.8 | 81.4 |
| `wm_a50` | WiSE 0.5 | 81.1 | 89.6 | 79.8 | 86.4 | 84.2 | 79.8 | 81.8 |
| `w_m3b_a60` | WiSE 0.6 | 81.1 | 89.9 | 80.1 | 87.1 | 84.5 | 79.6 | 81.6 |
| `w_m3b_a40` | **WiSE 0.4 (= the pick, final_m3)** | 80.4 | 89.4 | 79.0 | 85.3 | 83.5 | 80.3 | 81.9 |
| `w_m3b_a35` | WiSE 0.35 | 79.7 | 89.3 | 78.7 | 84.6 | 83.0 | 80.0 | 81.9 |
| `w_m3b_a30` | WiSE 0.3 | 79.5 | 89.1 | 78.3 | 83.9 | 82.7 | 80.3 | 81.8 |
| `soup_m3` | uniform soup of seeds 13/14/15 (raw) | 82.8 | 90.1 | 81.1 | 87.4 | 85.4 | 79.4 | 80.6 |
| `soup_a75` | soup, WiSE 0.75 | 82.0 | 90.2 | 80.8 | 87.2 | 85.1 | 79.7 | 81.2 |
| `soup_a60` | soup, WiSE 0.6 | 81.6 | 89.9 | 80.4 | 87.1 | 84.7 | 79.9 | 81.6 |
| `soup_a50` | soup, WiSE 0.5 | 80.5 | 89.6 | 79.6 | 86.6 | 84.1 | 80.0 | 81.8 |
| `soup_a40` | soup, WiSE 0.4 | 79.4 | 89.4 | 79.0 | 85.3 | 83.3 | 80.3 | 82.1 |
| `w_m3_r_s13_a75` | replay run, WiSE 0.75 | 80.7 | 89.6 | 80.4 | 87.4 | 84.5 | 79.5 |  |
| `w_m3_p_s13_a75` | preservation run, WiSE 0.75 | 81.4 | 89.9 | 80.6 | 86.9 | 84.7 | 79.8 |  |
| `w_m3_fr_s13_a75` | freeze run, WiSE 0.75 | 82.1 | 89.8 | 80.5 | 86.7 | 84.8 | 79.7 | 81.3 |
| `w_m3_fr_s13_a50` | freeze run, WiSE 0.5 | 81.0 | 89.6 | 79.5 | 85.4 | 83.9 | 79.9 | 81.8 |
| `w_m3_lo_s13_a75` | LoRA run, WiSE 0.75 | 82.8 | 89.9 | 80.7 | 86.7 | 85.0 | 79.9 | 80.5 |
| `w_m3_lo_s13_a50` | LoRA run, WiSE 0.5 | 81.4 | 89.7 | 79.8 | 86.4 | 84.3 | 79.8 | 81.4 |

* **Fine-tuning bge-m3 itself is the step that works:** DEV ar->en 73.5 -> 81.0 and macro 78.2 -> 85.3 for the raw fine-tune (alpha 1), TEST ar->en 63.3 -> 73.9 (section 7).
* **It forgets Arabic-only skills.** The raw fine-tune loses 6.0 on MIRACL-HN, 10.1 on NamaaMrTydi (MTEB guards, section 7). **MIRACL-ar replay** (1,515 queries, x3), a **preservation loss** towards frozen bge-m3, **LoRA** (r=32) and **frozen bottom 12 layers** all fail to lift the held-out MIRACL slice `devM` above the raw fine-tune by more than noise (devM 78.7-79.0 vs 79.1 raw, bge-m3 80.5). The only lever that moves `devM` and the non-MTEB proxy is **weight interpolation back towards bge-m3 (WiSE-FT)**: alpha 0.4 gives devM 80.3 (bge-m3 80.5) and a proxy of 81.9 (bge-m3 81.9) at DEV ar->en 79.0.
* **Seeds and soups.** Seeds 13/14/15 of the raw recipe span 1.2 TEST ar->en points (73.9 / 73.2 / 72.7, std 0.6); the uniform soup of the three barely changes the picture (next section).
* The LoRA run was first lost to a bug in my merge code (see section 11) and rerun with the identical recipe after a gated merge self-test.

### 3.3 The pre-registered validation-suite selection of v2-m3

The earlier alpha (0.75) had been picked by a rule fixed before any guard was read, and scored a **tie** with bge-m3 on MTEB(ara) (72.04 vs 71.98). The coordinator asked for a principled selection that does not touch MTEB: a six-component validation suite built from non-test splits that mirror the MTEB(ara) task types, **written down before any candidate was scored on it** (section 12):
dev macro (retrieval), `devM` (MIRACL-train hold-out), the mMARCO-ar proxy, XNLI-ar average precision (a seeded 3,000-pair subsample of the XNLI-ar *train* split, because mteb's XNLI task scores both the test and the validation split), 8-shot logistic-regression probes for sarcasm and sentiment on the ArSarcasm train split (MIT; fitted and scored on disjoint train rows; the ar_sarcasm test split is what MTEB's TweetSarcasmClassification uses), and Spearman on the Arabic STS-B validation split.
Each component is min-max scaled over the pool of all models (1-point floor); the score is the unweighted mean; eligibility requires DEV ar->en >= bge-m3 + 5; a tie (strictly < 0.02) goes to fewer ingredients. The pool: WiSE alpha in {0.3, 0.35, 0.4, 0.5, 0.6, 0.75} of the m3_b seed-13 fine-tune, alpha in {0.4, 0.5, 0.6, 0.75} of the 3-seed soup (0.3 / 0.35 were registered for the winning family only), the LoRA and freeze runs raw and blended, and three reference rows.

| model | family | alpha | dev | ar->en | devM | proxy | XNLI(tr) | cls | (sarc/sent) | STS | **score** | eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| soup_a40 | soup | 0.40 | 83.3 | 79.0 | 80.3 | 82.1 | 79.2 | 56.9 | 60.0/53.8 | 81.9 | **0.7612** | yes |
| w_m3b_a40 | wise | 0.40 | 83.5 | 79.0 | 80.3 | 81.9 | 79.1 | 56.9 | 60.0/53.9 | 81.9 | **0.7486** | yes |
| w_m3b_a30 | wise | 0.30 | 82.7 | 78.3 | 80.3 | 81.8 | 79.3 | 56.9 | 60.1/53.8 | 81.8 | **0.7228** | no |
| w_m3b_a35 | wise | 0.35 | 83.0 | 78.7 | 80.0 | 81.9 | 79.2 | 57.0 | 60.1/53.8 | 81.9 | **0.7175** | yes |
| w_m3_fr_s13_a50 | freeze | 0.50 | 83.9 | 79.5 | 79.9 | 81.8 | 79.0 | 56.8 | 60.0/53.6 | 82.1 | **0.7150** | yes |
| wm_a50 | wise | 0.50 | 84.2 | 79.8 | 79.8 | 81.8 | 78.8 | 57.0 | 60.0/54.0 | 82.0 | **0.7100** | yes |
| soup_a50 | soup | 0.50 | 84.1 | 79.6 | 80.0 | 81.8 | 78.9 | 56.8 | 59.8/53.8 | 82.0 | **0.7081** | yes |
| soup_a60 | soup | 0.60 | 84.7 | 80.4 | 79.9 | 81.6 | 78.7 | 56.6 | 59.5/53.8 | 82.1 | **0.6722** | yes |
| w_m3b_a60 | wise | 0.60 | 84.5 | 80.1 | 79.6 | 81.6 | 78.5 | 56.9 | 60.0/53.9 | 82.0 | **0.6602** | yes |
| w_m3_fr_s13_a75 | freeze | 0.75 | 84.8 | 80.5 | 79.7 | 81.3 | 78.4 | 56.6 | 59.8/53.5 | 82.3 | **0.6486** | yes |
| wm_a75 | wise | 0.75 | 84.9 | 80.5 | 79.8 | 81.4 | 78.1 | 56.8 | 59.8/53.8 | 82.1 | **0.6429** | yes |
| w_m3_lo_s13_a50 | lora | 0.50 | 84.3 | 79.8 | 79.8 | 81.4 | 78.7 | 56.6 | 59.9/53.3 | 82.0 | **0.6250** | yes |
| bge-m3 | reference | 0.00 | 78.2 | 73.5 | 80.5 | 81.9 | 80.0 | 56.8 | 59.9/53.7 | 81.5 | **0.6193** | ref |
| soup_a75 | soup | 0.75 | 85.1 | 80.8 | 79.7 | 81.2 | 78.3 | 56.6 | 59.5/53.7 | 82.2 | **0.6004** | yes |
| w_m3_lo_s13_a75 | lora | 0.75 | 85.0 | 80.7 | 79.9 | 80.5 | 78.0 | 56.4 | 59.6/53.1 | 82.1 | **0.4990** | yes |
| m3_fr_s13 | freeze | 1.00 | 85.0 | 81.1 | 78.8 | 80.5 | 77.9 | 56.5 | 59.8/53.3 | 82.5 | **0.4762** | yes |
| m3_b_s13 | reference | 1.00 | 85.3 | 81.0 | 79.1 | 80.8 | 77.5 | 56.7 | 59.6/53.7 | 82.2 | **0.4741** | ref |
| soup_m3 | reference | 1.00 | 85.4 | 81.1 | 79.4 | 80.6 | 77.7 | 56.4 | 59.2/53.6 | 82.2 | **0.4636** | ref |
| m3_lo_s13 | lora | 1.00 | 85.4 | 81.5 | 79.0 | 80.2 | 77.3 | 56.0 | 59.3/52.7 | 82.1 | **0.2817** | yes |

* **Pick: `w_m3b_a40`** (single-seed WiSE, alpha 0.4), score 0.7486. The top raw score was `soup_a40` (0.7612); the gap of **0.0127 is inside the tie band, so the ingredients clause decided it** (single-seed before soup). I report it as a near-tie: the soup variant is equally defensible and was not run on TEST.
* The score keeps rising as alpha falls to 0.4. Alpha 0.30 (added by pre-registered amendment) fails eligibility (DEV ar->en 78.3 < 78.5); 0.35 is eligible but scores lower (0.7175), so the optimum among the tested points is 0.4.
* LoRA and freeze (and their blends) lose on the suite (best blends 0.625 and 0.715, see table).
* Across the pool the STS and probe components move by about one point and XNLI-train by under three; the ranking is driven mostly by the retrieval components (dev macro and `devM`) and the proxy. They are not independent evidence that MTEB's classification tasks are safe: one probe uses the same source as an MTEB task.

### 3.4 The students: distillation from the final teacher (DEV, seed 13 unless named)

| run | what | ar->ar | en->en | ar->en | en->ar | macro | devM | mMARCO proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base_e5-base` | e5-base off the shelf | 74.3 | 90.1 | 62.9 | 72.6 | 75.0 | 74.3 | 81.1 |
| `f4_b_e1_s13` | e5-base fine-tuned on the v2 data, no distillation | 76.1 | 86.1 | 71.0 | 81.9 | 78.8 |  |  |
| `f5_b_kd_s13` | + KD from the raw v2-m3 seed 13 (CE + KL w=1, 1 epoch) | 78.4 | 88.8 | 74.5 | 83.0 | 81.2 |  |  |
| `f5_b_kd3_s13` | + KD w=3 | 79.1 | 89.5 | 75.6 | 82.9 | 81.8 | 73.1 |  |
| `f5_b_kdonly_s13` | KD only (no CE), 1 epoch | 79.9 | 89.4 | 76.1 | 82.6 | 82.0 | 73.2 |  |
| `f5_b_kd_e2_s13` | CE + KD w=1, 2 epochs | 79.3 | 89.4 | 76.9 | 83.6 | 82.3 | 73.9 |  |
| `f5_b_kdonly_e2_s13` | KD only, 2 epochs  (**chosen recipe**) | 79.5 | 89.5 | 77.1 | 83.4 | 82.4 | 73.9 |  |
| `st_b_s13` | **FINAL recipe, teacher = final_m3**: e5-base KD-only 2 epochs, seed 13 | 77.3 | 88.7 | 75.7 | 83.2 | 81.2 | 75.2 |  |
| `st_b_s14` | seed 14 | 77.9 | 89.1 | 75.3 | 83.9 | 81.5 | 75.7 |  |
| `st_b_s15` | seed 15 | 77.6 | 88.8 | 75.0 | 83.3 | 81.2 | 74.7 |  |
| `st_b_soup` | **3-seed soup = v2-base** | 78.1 | 89.1 | 75.9 | 83.5 | 81.6 | 75.2 |  |
| `base_e5-small` | e5-small off the shelf | 72.6 | 87.3 | 53.6 | 68.0 | 70.4 | 74.4 | 79.5 |
| `f4_s_e1_s13` | e5-small fine-tuned on the v2 data, no distillation | 79.0 | 86.6 | 68.0 | 80.0 | 78.4 |  |  |
| `f5_s_kd_s13` | + KD from the raw v2-m3 seed 13 | 79.3 | 87.1 | 68.8 | 79.5 | 78.7 |  |  |
| `st_s_s13` | **FINAL recipe, teacher = final_m3**: e5-small KD-only 2 epochs, seed 13 | 76.7 | 87.7 | 69.4 | 80.4 | 78.6 | 73.8 |  |
| `st_s_s14` | seed 14 | 76.5 | 87.7 | 69.4 | 80.6 | 78.5 | 74.3 |  |
| `st_s_s15` | seed 15 | 76.2 | 87.6 | 69.3 | 80.3 | 78.4 | 74.4 |  |
| `st_s_soup` | **3-seed soup = v2-small** | 76.2 | 87.7 | 69.4 | 80.7 | 78.5 | 74.1 |  |

On the same data, plain fine-tuning moves the 118M model's TEST ar->en from 55.1 (v1) to 56.7-58.0 and the 278M model's to 59.9-60.7: capacity, not data, limits the small models. Distilling from the **fine-tuned** v2-m3 (listwise KL on in-batch scores, T = 20) is what moves them (e5-base TEST ar->en 60.7 -> 64.5 already with the raw seed-13 teacher); KD from *off-the-shelf* bge-m3 (section 3.1) did not. The recipe was picked on DEV among CE+KD w=1, w=3, KD-only and 2-epoch variants: **KD-only for 2 epochs** (lr 3e-5, scale 20, batch 64). The final students distil from `final_m3` (teacher embeddings tag `m3f`). v2-base and v2-small are the uniform soups of three seeds (rule fixed beforehand: soup unless its DEV macro is more than 0.5 below its seeds' mean; it was not).

### 3.5 Style diagnostic: is DEV optimistic because it is Gemma-translated?

The English XQuAD questions were translated with the same Gemma prompt; every model is scored on the human-translated and on the Gemma-translated version of the SAME 1,190 questions against the same English test pool.

| model | human-translated (XQuAD ar) | Gemma-translated (same questions) | gap |
|---|---:|---:|---:|
| v1_s13 | 55.1 | 60.9 | +5.8 |
| base_e5-small | 41.4 | 47.6 | +6.2 |
| bge-m3 | 63.3 | 68.4 | +5.1 |

Every model - including bge-m3, which never saw Gemma output - scores 5-6 points higher on Gemma-translated Arabic questions: human translations are noisier, and the offset is the same for fine-tuned and off-the-shelf models, so the fine-tuning gains do not come from learning the translator's style.

## 4. v2-m3, v2-base and v2-small on TEST

| system | params | ar->ar | en->en | ar->en | en->ar | macro | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge-m3 (off the shelf) | 568M | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 | 1 |
| multilingual-e5-base (off the shelf) | 278M | 78.0 | 89.8 | 49.9 | 71.9 | 72.4 | 1 |
| multilingual-e5-small (off the shelf) | 118M | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 | 1 |
| v1 e5-small, 3 seeds | 118M | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 | 3 |
| **v2-m3** = bge-m3 fine-tuned + WiSE 0.4 (pick), seed 13 | 568M | 85.4 | 89.8 | 71.2 | 82.6 | 82.2 | 1 |
| v2-m3, development seeds 13/14/15 | 568M | 85.4 ± 0.1 | 89.7 ± 0.1 | 70.9 ± 0.3 | 82.7 ± 0.1 | 82.2 ± 0.1 | 3 |
| v2-m3, **confirmatory** seeds 16/17 | 568M | 85.5 ± 0.0 | 89.7 ± 0.1 | 70.7 ± 0.5 | 82.6 ± 0.1 | 82.1 ± 0.1 | 2 |
| bge-m3 fine-tuned, RAW (alpha 1), seeds 13/14/15 | 568M | 86.9 ± 0.3 | 89.8 ± 0.2 | 73.3 ± 0.6 | 88.0 ± 0.4 | 84.5 ± 0.3 | 3 |
| bge-m3 fine-tuned, RAW (alpha 1), seeds 16/17 | 568M | 86.8 ± 0.2 | 89.8 ± 0.4 | 73.4 ± 0.3 | 88.0 ± 0.3 | 84.5 ± 0.3 | 2 |
| **v2-base** = e5-base distilled from v2-m3, 3-seed soup | 278M | 82.2 | 89.7 | 66.6 | 82.2 | 80.2 | 1 |
| v2-base, the 3 seeds | 278M | 82.1 ± 0.1 | 89.6 ± 0.1 | 65.9 ± 0.2 | 81.9 ± 0.3 | 79.9 ± 0.1 | 3 |
| **v2-small** = e5-small distilled from v2-m3, 3-seed soup | 118M | 80.3 | 88.0 | 59.7 | 79.7 | 76.9 | 1 |
| v2-small, the 3 seeds | 118M | 80.4 ± 0.0 | 87.9 ± 0.2 | 59.3 ± 0.3 | 79.7 ± 0.2 | 76.8 ± 0.0 | 3 |

`±` is the sample standard deviation over seeds. v2-m3 is the pick (WiSE 0.4); the RAW rows are the fine-tune without interpolation (the alpha = 1 end of the curve in section 7), which scores 2.4-2.7 ar->en points higher on TEST and forgets much more on the guard tasks. Of the ≤300M models only v2-base is above bge-m3's TEST ar->en; v2-small is below it but well above e5-small and v1.

Paired bootstrap (2,000 resamples, queries resampled within slice, point estimate [95% CI], * = CI excludes 0):

| comparison (system minus reference) | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| v2-m3 (seed 13) vs bge-m3 | +5.3 [+4.6, +5.9]* | +2.0 [+1.4, +2.5]* | +7.8 [+6.7, +8.9]* | +14.8 [+13.3, +16.4]* | +7.5 [+6.9, +8.0]* |
| v2-m3, mean of dev seeds 13/14/15 vs bge-m3 | +5.3 [+4.6, +5.9]* | +1.9 [+1.4, +2.4]* | +7.5 [+6.4, +8.6]* | +15.0 [+13.5, +16.5]* | +7.4 [+6.9, +7.9]* |
| v2-m3, mean of CONFIRMATORY seeds 16/17 vs bge-m3 | +5.3 [+4.7, +6.0]* | +1.9 [+1.3, +2.4]* | +7.3 [+6.2, +8.4]* | +14.9 [+13.3, +16.3]* | +7.3 [+6.8, +7.9]* |
| v2-m3 (seed 13) vs v1 (seed 13) | +5.3 [+4.3, +6.2]* | +2.5 [+1.6, +3.5]* | +16.0 [+14.0, +18.2]* | +2.2 [+0.5, +3.8]* | +6.5 [+5.7, +7.3]* |
| v2-base (soup) vs e5-base | +4.1 [+3.4, +4.9]* | -0.1 [-0.8, +0.6] | +16.7 [+14.8, +18.6]* | +10.3 [+8.8, +11.9]* | +7.8 [+7.1, +8.4]* |
| v2-base (soup) vs bge-m3 | +2.0 [+1.2, +3.0]* | +1.9 [+1.0, +2.7]* | +3.2 [+1.4, +5.1]* | +14.5 [+12.6, +16.2]* | +5.4 [+4.6, +6.1]* |
| v2-base (soup) vs v1 (seed 13) | +2.0 [+1.2, +2.8]* | +2.4 [+1.5, +3.3]* | +11.5 [+9.6, +13.4]* | +1.9 [+0.3, +3.4]* | +4.4 [+3.7, +5.1]* |
| v2-small (soup) vs e5-small | +3.5 [+2.7, +4.3]* | +0.2 [-0.5, +1.0] | +18.3 [+16.4, +20.3]* | +18.0 [+16.2, +19.9]* | +10.0 [+9.4, +10.7]* |
| v2-small (soup) vs v1 (seed 13) | +0.2 [-0.4, +0.8] | +0.7 [+0.1, +1.4]* | +4.6 [+3.3, +5.9]* | -0.7 [-1.8, +0.4] | +1.2 [+0.7, +1.7]* |
| v2-small (soup) vs bge-m3 | +0.2 [-0.8, +1.2] | +0.2 [-0.8, +1.2] | -3.6 [-5.6, -1.7]* | +11.9 [+9.9, +13.9]* | +2.2 [+1.3, +3.0]* |

### 4.1 Mixed Arabic+English index (language-bias diagnostic, TEST)

One index holding both languages of the test pool. `strict` counts only the passage in the *other* language than the query (the labelled gold; the parallel same-language paragraph becomes a distractor), `lenient` also accepts the parallel paragraph. A model with a same-language bias retrieves the Arabic twin for an Arabic query, which is a correct answer in practice but a miss in `strict`.

| model | slice | strict MRR@10 | lenient MRR@10 | top-1 in the query's own language |
|---|---|---:|---:|---:|
| bge-m3 | xquad_ar2en | 33.8 | 76.8 | 80% |
| bge-m3 | xquad_en2ar | 20.5 | 79.6 | 94% |
| e5-base | xquad_ar2en | 3.2 | 79.8 | 100% |
| e5-base | xquad_en2ar | 1.2 | 85.5 | 100% |
| e5-small | xquad_ar2en | 1.4 | 76.9 | 100% |
| e5-small | xquad_en2ar | 0.8 | 82.6 | 100% |
| v1 (seed 13) | xquad_ar2en | 27.5 | 78.1 | 83% |
| v1 (seed 13) | xquad_en2ar | 23.7 | 80.6 | 96% |
| v2-m3 (pick) | xquad_ar2en | 40.2 | 83.1 | 78% |
| v2-m3 (pick) | xquad_en2ar | 27.5 | 83.9 | 95% |
| bge-m3 fine-tuned RAW (seed 13) | xquad_ar2en | 44.6 | 84.2 | 71% |
| bge-m3 fine-tuned RAW (seed 13) | xquad_en2ar | 32.1 | 85.3 | 95% |
| v2-base (seed 13) | xquad_ar2en | 37.1 | 79.5 | 76% |
| v2-base (seed 13) | xquad_en2ar | 25.3 | 82.7 | 95% |
| v2-small (seed 13) | xquad_ar2en | 31.6 | 77.7 | 79% |
| v2-small (seed 13) | xquad_en2ar | 24.6 | 80.9 | 95% |

## 5. Confirmatory replication (seeds 16 and 17)

**Registered before training** (PRE-REGISTRATION 3, 22:10; recipe frozen 22:46:54 at the pick; the first confirmatory seed started 23:13): the recipe is the (family, alpha) of the final select, **unchanged**; two NEW seeds (16, 17) that appear in no selection step, trained with the identical arguments and the identical WiSE alpha towards bge-m3 (0.4); reported whatever they show.

| role | model (WiSE 0.4) | TEST ar->en | en->ar | ar->ar | en->en | macro | MTEB(ara) Mean (Task) | minus bge-m3 | same seed, RAW fine-tune: TEST ar->en / macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development seed 13 (= final_m3) | `final_m3` | 71.2 | 82.6 | 85.4 | 89.8 | 82.2 | 72.29 | +0.31 | 73.9 / 84.8 |
| development seed 14 | `w_m3b_a40_s14` | 70.9 | 82.7 | 85.5 | 89.8 | 82.2 | not scored |  | 73.2 / 84.3 |
| development seed 15 | `w_m3b_a40_s15` | 70.5 | 82.9 | 85.3 | 89.6 | 82.1 | not scored |  | 72.7 / 84.3 |
| **confirmatory seed 16** | `w_m3b_a40_s16` | 71.1 | 82.6 | 85.5 | 89.7 | 82.2 | 72.28 | +0.30 | 73.6 / 84.7 |
| **confirmatory seed 17** | `w_m3b_a40_s17` | 70.3 | 82.7 | 85.5 | 89.6 | 82.0 | 72.27 | +0.29 | 73.1 / 84.3 |

The confirmatory seeds reproduce the development seeds: TEST ar->en is 71.1 and 70.3 (development 71.2 / 70.9 / 70.5), TEST macro 82.2 / 82.0 (development 82.2 / 82.2 / 82.1), and MTEB(ara) Mean (Task) is 72.28 and 72.27 against 72.29 for seed 13. The paired bootstrap of each confirmatory seed against bge-m3 is in section 6.2: each is +0.29 to +0.30 with a CI that includes 0, the same conclusion as for seed 13 (a numerical lead that is not statistically distinguishable). The seed-to-seed spread of the MTEB mean (0.02) is far below the lead's CI half-width (about 0.65), so the lead is stable but small.

## 6. MTEB(ara, v1)

All models are scored by the same pipeline (mteb 2.26.33, fp16, batch 32, max length 512; the bge-m3 family without e5 prompts, ModelMeta copied from the registered base model). The board below adds the v2 models to the MTEB agent's 10-model local leaderboard (`results/mteb_board.json`, same definitions as `mteb-arabic/scripts/make_leaderboard.py`; the baselines are the MTEB agent's own runs).

| model | params | Mean (Task) | Mean (TaskType) | Retr. | Rerank. | STS | PairClf | Classif. | Clust. | Bitext | ar-only subsets | **ar-en cross-lingual subsets** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v2-m3 (pick)** | 568M | 72.29 | 70.93 | 74.4 | 76.0 | 67.3 | 82.8 | 64.5 | 37.1 | 94.5 | 70.5 | 75.0 |
| v2-m3 confirmatory seed 16 | 568M | 72.28 | 70.86 | 74.4 | 75.7 | 67.4 | 82.6 | 64.5 | 37.1 | 94.4 | 70.5 | 75.0 |
| v2-m3 confirmatory seed 17 | 568M | 72.27 | 70.83 | 74.3 | 75.5 | 67.4 | 82.8 | 64.5 | 36.7 | 94.5 | 70.4 | 75.0 |
| BAAI/bge-m3 | 568M | 71.98 | 70.89 | 73.5 | 79.4 | 66.3 | 83.5 | 64.5 | 34.8 | 94.2 | 70.4 | 73.3 |
| intfloat/multilingual-e5-large | 560M | 71.44 | 70.32 | 74.0 | 79.6 | 68.1 | 79.5 | 61.8 | 35.1 | 94.1 | 69.5 | 75.3 |
| **v2-base** | 278M | 68.91 | 67.62 | 71.1 | 68.2 | 65.5 | 74.5 | 60.4 | 40.0 | 93.7 | 66.3 | 74.2 |
| intfloat/multilingual-e5-base | 278M | 67.89 | 67.28 | 68.8 | 77.1 | 64.9 | 74.0 | 59.6 | 34.2 | 92.4 | 66.5 | 69.1 |
| **v2-small** | 118M | 66.20 | 65.42 | 67.9 | 68.9 | 59.1 | 73.6 | 59.4 | 38.8 | 90.2 | 64.1 | 69.7 |
| intfloat/multilingual-e5-small | 118M | 65.63 | 65.70 | 66.0 | 79.2 | 60.1 | 74.1 | 58.3 | 34.7 | 87.4 | 65.0 | 63.9 |
| v1 (mzahran1/arabic-english-e5-small-retrieval) | 118M | 65.00 | 64.45 | 66.3 | 65.9 | 57.4 | 71.8 | 59.0 | 41.8 | 88.9 | 63.1 | 68.1 |

### 6.1 Per task

| task | bge-m3 | v2-m3 | s16 | s17 | v2-m3 - bge-m3 | shares a source with our training data? |
|---|---:|---:|---:|---:|---:|---|
| MIRACLRetrievalHardNegatives.v2 | 78.9 | 77.9 | 78.0 | 78.1 | -1.0 | TyDi/MIRACL questions (near-duplicates filtered; 9 exact copies remain in the training rows, section 8.1) |
| XQuADRetrieval | 94.4 | 95.3 | 95.3 | 95.1 | +0.9 | SQuAD-dev translations = v1 TEST source (excluded) |
| MLQARetrieval | 60.8 | 63.8 | 63.8 | 63.7 | +3.0 | SQuAD-style Wikipedia (no training use) |
| BelebeleRetrieval | 88.7 | 89.9 | 89.9 | 89.8 | +1.2 |  |
| XPQARetrieval | 38.0 | 39.2 | 39.1 | 39.0 | +1.2 |  |
| WebFAQRetrieval | 80.3 | 80.1 | 80.2 | 80.1 | -0.2 |  |
| NamaaMrTydiReranking | 79.4 | 76.0 | 75.7 | 75.5 | -3.4 | Mr.TyDi test (TyDi source; v2 trains on TyDi-ar train) |
| STS17 | 75.1 | 76.8 | 77.0 | 76.9 | +1.7 |  |
| STS22.v2 | 64.7 | 64.1 | 64.0 | 64.3 | -0.6 |  |
| SemRel24STS | 59.1 | 60.9 | 61.2 | 61.1 | +1.8 |  |
| XNLI | 77.9 | 76.4 | 76.2 | 76.5 | -1.5 |  |
| ArEntail | 89.1 | 89.1 | 88.9 | 89.1 | +0.0 |  |
| AJGT.v2 | 84.1 | 83.6 | 83.8 | 83.7 | -0.5 |  |
| TweetEmotionClassification.v2 | 57.8 | 58.3 | 58.2 | 58.2 | +0.5 |  |
| TweetSarcasmClassification.v2 | 61.7 | 61.6 | 61.3 | 61.8 | -0.1 |  |
| HotelReviewSentimentClassification.v2 | 54.4 | 54.4 | 54.8 | 54.4 | -0.1 |  |
| SIB200ClusteringS2S.v2 | 34.8 | 37.1 | 37.1 | 36.7 | +2.3 |  |
| FloresBitextMining | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |  |
| Tatoeba | 88.3 | 89.1 | 88.8 | 89.1 | +0.7 |  |
| **Mean (Task)** | 71.98 | 72.29 | 72.28 | 72.27 | +0.31 | |

| task | e5-base | v2-base | e5-small | v1 | v2-small | v2-base - e5-base | v2-small - e5-small | v2-small - v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MIRACLRetrievalHardNegatives.v2 | 73.0 | 70.7 | 72.4 | 66.3 | 68.3 | -2.3 | -4.1 | +2.0 |
| XQuADRetrieval | 94.6 | 93.6 | 93.5 | 91.4 | 92.4 | -1.1 | -1.1 | +1.0 |
| MLQARetrieval | 55.1 | 60.5 | 50.7 | 54.4 | 57.4 | +5.5 | +6.6 | +2.9 |
| BelebeleRetrieval | 84.0 | 88.4 | 80.1 | 82.6 | 84.4 | +4.4 | +4.3 | +1.8 |
| XPQARetrieval | 30.6 | 36.4 | 29.9 | 31.7 | 33.0 | +5.8 | +3.2 | +1.4 |
| WebFAQRetrieval | 75.7 | 76.8 | 69.6 | 71.4 | 72.2 | +1.1 | +2.6 | +0.8 |
| NamaaMrTydiReranking | 77.1 | 68.2 | 79.2 | 65.9 | 68.9 | -8.9 | -10.4 | +3.0 |
| STS17 | 74.0 | 75.2 | 68.0 | 71.1 | 70.6 | +1.2 | +2.6 | -0.5 |
| STS22.v2 | 61.3 | 62.4 | 61.4 | 59.8 | 60.7 | +1.1 | -0.8 | +0.9 |
| SemRel24STS | 59.2 | 59.0 | 51.0 | 41.2 | 46.1 | -0.2 | -4.8 | +4.9 |
| XNLI | 67.3 | 67.8 | 68.9 | 66.2 | 68.4 | +0.5 | -0.5 | +2.2 |
| ArEntail | 80.6 | 81.3 | 79.4 | 77.5 | 78.9 | +0.6 | -0.5 | +1.4 |
| AJGT.v2 | 78.4 | 80.1 | 75.4 | 78.4 | 77.9 | +1.7 | +2.6 | -0.5 |
| TweetEmotionClassification.v2 | 52.4 | 53.0 | 50.4 | 50.0 | 50.7 | +0.7 | +0.3 | +0.7 |
| TweetSarcasmClassification.v2 | 57.5 | 58.3 | 58.6 | 61.2 | 60.3 | +0.8 | +1.7 | -0.9 |
| HotelReviewSentimentClassification.v2 | 50.1 | 50.2 | 48.9 | 46.4 | 48.5 | +0.1 | -0.4 | +2.0 |
| SIB200ClusteringS2S.v2 | 34.2 | 40.0 | 34.7 | 41.8 | 38.8 | +5.8 | +4.2 | -3.0 |
| FloresBitextMining | 99.2 | 99.9 | 98.0 | 98.7 | 99.3 | +0.7 | +1.3 | +0.7 |
| Tatoeba | 85.6 | 87.5 | 76.8 | 79.1 | 81.0 | +1.9 | +4.2 | +1.9 |
| **Mean (Task)** | 67.89 | 68.91 | 65.63 | 65.00 | 66.20 | +1.02 | +0.57 | +1.19 |

### 6.2 Is the difference real? (CPU, from the result caches)

For each pair: per-task wins/losses/ties (tie = |difference| < 0.005), a percentile-bootstrap 95% CI of the Mean (Task) difference obtained by **resampling the 19 tasks** (10,000 resamples, seed 13), and an exact two-sided sign test over the non-tied tasks. `*` marks a CI that excludes 0. Resampling tasks asks "would a different set of tasks reverse the ranking?"; it ignores the within-task query noise, so it is a coarse but conservative-in-spirit test, with only 19 tasks.

| new | reference | Mean(Task) new / ref | difference | 95% CI (resampling the 19 tasks, 10k) | wins / losses / ties | exact sign test p |
|---|---|---:|---:|---|---:|---:|
| `final_m3_wa75_s13` | `BAAI__bge-m3` | 72.04 / 71.98 | +0.06 | [-1.14, +1.16] | 9 / 9 / 1 | 1.000 |
| `final_m3` | `BAAI__bge-m3` | 72.29 / 71.98 | +0.31 | [-0.34, +0.93] | 10 / 8 / 1 | 0.815 |
| `w_m3b_a40_s16` | `BAAI__bge-m3` | 72.28 / 71.98 | +0.30 | [-0.40, +0.95] | 10 / 8 / 1 | 0.815 |
| `w_m3b_a40_s17` | `BAAI__bge-m3` | 72.27 / 71.98 | +0.29 | [-0.39, +0.91] | 10 / 7 / 2 | 0.629 |
| `final_base` | `intfloat__multilingual-e5-base` | 68.91 / 67.89 | +1.02 | [-0.52, +2.37] | 15 / 4 / 0 | 0.019 |
| `final_small` | `intfloat__multilingual-e5-small` | 66.20 / 65.63 | +0.57 | [-1.24, +2.19] | 11 / 8 / 0 | 0.648 |
| `final_small` | `mzahran1__arabic-english-e5-small-retrieval` | 66.20 / 65.00 | +1.19 | [+0.43, +1.92] * | 15 / 4 / 0 | 0.019 |

* **v2-m3 vs bge-m3:** +0.31, 10 wins / 8 losses / 1 tie, CI includes 0, sign test p = 0.815. The honest reading is "numerically ahead, not distinguishable", not "better". Largest per-task gains: MLQA +3.0, SIB200 clustering +2.3, SemRel24 +1.8, STS17 +1.7, Belebele +1.2, XPQA +1.2, XQuAD +0.9; largest losses: NamaaMrTydi -3.4, XNLI -1.5, MIRACL-HN -1.0, AJGT -0.5, STS22 -0.6. The three confirmatory-seed columns agree with seed 13 to within 0.3 on every task.
* **v2-base vs e5-base:** +1.02 with 15 wins / 4 losses (sign test p = 0.019), but the task-bootstrap CI [-0.52, +2.37] includes 0: the gain is broad across tasks but very uneven in size (MLQA +5.5, XPQA +5.8, Belebele +4.4, SIB200 +5.8, against NamaaMrTydi -8.9, MIRACL-HN -2.3 and XQuAD -1.1).
* **v2-small vs e5-small:** +0.57, CI includes 0. **v2-small vs v1:** +1.19, CI [+0.43, +1.92] excludes 0, 15 wins / 4 losses. v2-small loses 4.1 on MIRACL-HN, 10.4 on NamaaMrTydi and 4.8 on SemRel24 against e5-small and gains 6.6 on MLQA, 4.3 on Belebele, 4.2 on Tatoeba.

## 7. Blend curve (POST-HOC figure, not used for selection)

The m3_b seed-13 fine-tune interpolated with bge-m3: `W = (1 - alpha) W_bge-m3 + alpha W_ft` (alpha = the fine-tuned share; 0 = bge-m3, 1 = the raw fine-tune). TEST and the guards were read for the blends AFTER the pick was frozen; the curve is descriptive and `results/alpha_curve.json` carries `"posthoc": true`.

| alpha (fine-tuned share) | TEST ar->en | TEST macro | devM | mMARCO proxy | MIRACL-HN ar | NamaaMrTydi | SemRel24STS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 63.3 | 74.8 | 80.5 | 81.9 | 78.9 | 79.4 | 59.1 |
| 0.30 | 69.7 | 81.0 | 80.3 | 81.8 | 78.5 | 77.0 | 60.5 |
| 0.40 **(pick)** | 71.2 | 82.2 | 80.3 | 81.9 | 77.9 | 76.0 | 60.9 |
| 0.50 | 72.3 | 83.1 | 79.8 | 81.8 | 77.5 | 74.7 | 61.2 |
| 0.60 | 73.1 | 83.8 | 79.6 | 81.6 | 76.8 | 73.7 | 61.5 |
| 0.75 | 73.5 | 84.4 | 79.8 | 81.4 | 75.7 | 72.1 | 61.8 |
| 1.00 | 73.9 | 84.8 | 79.1 | 80.8 | 72.9 | 69.3 | 62.1 |

Reading: ar->en and macro rise steadily with alpha (63.3 -> 73.9 and 74.8 -> 84.8) while MIRACL-HN and NamaaMrTydi fall steadily (78.9 -> 72.9 and 79.4 -> 69.3) and SemRel24 rises (59.1 -> 62.1); `devM` and the proxy decline slowly (devM 80.5 -> 79.1, proxy 81.9 -> 80.8). There is no knee that gets the full ar->en gain for free: the pick (0.4) keeps about 75% of the TEST ar->en gain (+7.9 of +10.6) and of the macro gain (+7.4 of +10.0) at a MIRACL-HN cost of 1.0 and a Namaa cost of 3.4.

## 8. Guard tasks and the contamination split

The MTEB(ara) guards (MIRACL-HN.v2 ar, NamaaMrTydiReranking, SemRel24STS) are in section 6.1 and the blend curve. bge-m3 was trained on TyDi-QA / Mr.TyDi train and MIRACL train; the MIRACL-ar dev queries contain many TyDi-ar train questions, and v2's training data removed everything that overlaps MIRACL dev. The diagnostic below (no training) splits the queries of the two guard tasks into CONTAMINATED (an exact normalised copy, or a near-duplicate with token Jaccard >= 0.8, of a question of the FULL TyDi-QA-ar train set or of the MIRACL-ar train set) and CLEAN and recomputes the metric per subset **with passage titles included** (as mteb does; metric computed by our own code: nDCG@10 over the full corpus for MIRACL, average precision over the candidate lists for Namaa). Two corrections were needed here (section 11): five models had stale rows from a first run without titles (recomputed; bge-m3 now reproduces mteb: 79.0 vs 78.9), and the first version of the labelling used only the Jaccard detector, which misses exact copies of short common-word questions (it flagged 816 of 1,000 MIRACL queries; 893 are exact TyDi-train copies). The labels below include exact copies. The per-query scores are cached, so only the report was recomputed.


**MIRACL-HN.v2 ar (nDCG@10 over the full corpus)** - 1000 queries: both 20, clean 105, tydi 875

| model | all | CLEAN | CONTAMINATED (tydi / miracl / both) | clean - contaminated |
|---|---:|---:|---:|---:|
| bge-m3 | 79.0 | 71.9 (n=105) | 79.9 (n=895) | -8.0 |
| e5-base | 73.0 | 67.4 (n=105) | 73.6 (n=895) | -6.2 |
| e5-small | 72.4 | 64.7 (n=105) | 73.3 (n=895) | -8.5 |
| final_base | 70.7 | 61.6 (n=105) | 71.8 (n=895) | -10.2 |
| final_m3 | 77.9 | 70.5 (n=105) | 78.8 (n=895) | -8.3 |
| final_small | 68.3 | 57.1 (n=105) | 69.6 (n=895) | -12.5 |
| m3_b_s13 | 72.9 | 63.3 (n=105) | 74.0 (n=895) | -10.7 |
| v1 | 65.8 | 54.0 (n=105) | 67.2 (n=895) | -13.2 |
| wm_a75 | 75.7 | 67.7 (n=105) | 76.7 (n=895) | -8.9 |

**NamaaMrTydi (average precision over the candidate lists)** - 918 queries: both 6, clean 891, miracl 1, tydi 20

| model | all | CLEAN | CONTAMINATED (tydi / miracl / both) | clean - contaminated |
|---|---:|---:|---:|---:|
| bge-m3 | 79.4 | 79.1 (n=891) | 88.5 (n=27) | -9.4 |
| e5-base | 77.4 | 77.2 (n=891) | 82.7 (n=27) | -5.5 |
| e5-small | 79.3 | 79.2 (n=891) | 84.1 (n=27) | -4.9 |
| final_base | 68.1 | 68.2 (n=891) | 67.4 (n=27) | +0.8 |
| final_m3 | 76.0 | 75.9 (n=891) | 80.7 (n=27) | -4.8 |
| final_small | 68.9 | 68.7 (n=891) | 74.2 (n=27) | -5.5 |
| m3_b_s13 | 69.4 | 69.4 (n=891) | 69.6 (n=27) | -0.2 |
| v1 | 65.8 | 65.6 (n=891) | 70.7 (n=27) | -5.1 |
| wm_a75 | 72.1 | 72.0 (n=891) | 74.5 (n=27) | -2.5 |

Paired per-query differences on each subset (2,000 bootstrap resamples):

| task | subset | n | v2-m3 minus bge-m3 [95% CI] | v2-base minus e5-base | v2-small minus e5-small |
|---|---|---:|---|---|---|
| MIRACL-HN.v2 ar | clean | 105 | -1.4 [-3.3, +0.5] | -5.7 [-9.0, -2.6] | -7.6 [-11.5, -4.1] |
| MIRACL-HN.v2 ar | contaminated | 895 | -1.1 [-1.6, -0.5] | -1.8 [-3.0, -0.8] | -3.7 [-5.0, -2.4] |
| NamaaMrTydi | clean | 891 | -3.2 [-4.2, -2.2] | -9.0 [-10.4, -7.6] | -10.5 [-12.2, -9.0] |
| NamaaMrTydi | contaminated | 27 | -7.8 [-15.9, -1.5] | -15.3 [-24.4, -6.9] | -9.9 [-22.2, +2.8] |

Reading, without taking a side: 895 of the 1,000 MIRACL-HN queries are exact or near copies of TyDi / MIRACL training questions, so only 105 are clean; for NamaaMrTydi 27 of 918 are flagged. **MIRACL-HN:** bge-m3 scores 79.9 on the flagged and 71.9 on the clean queries; v2-m3 78.8 and 70.5, a deficit of 1.1 and 1.4 points, similar in both subsets and with intervals that overlap (the clean subset is small). Every model in the table scores lower on the clean subset (by 6.2 to 13.2 points), including the off-the-shelf e5 models, so the clean queries are harder or differently distributed; the gap is therefore not specific to bge-m3. The models that were fine-tuned on TyDi training questions show larger clean-vs-flagged gaps than bge-m3 (-8.0): v1 -13.2, the raw fine-tune -10.7, v2-small -12.5, v2-base -10.2; v2-m3 (-8.3) is close to bge-m3. Hence there is no sign that the MIRACL-HN deficit of v2-m3 is produced by bge-m3 having memorised the flagged queries (the deficit is of the same size on the clean queries, with a wide interval), and the 8 exactly leaked queries (next section) cannot account for it either. The students' MIRACL-HN deficits are larger on the clean queries (v2-base -5.7, v2-small -7.6) than on the flagged ones (-1.8, -3.7). **NamaaMrTydi:** the deficit of v2-m3 is -3.2 on the 891 clean queries and larger on the 27 flagged ones (-7.8), so the loss on that task is not a contamination effect either. These subset comparisons use a detector with known blind spots (it misses near-duplicates of short common-word questions beyond the exact copies now flagged); the labels are best read as a lower bound on contamination.

### 8.1 Exact copies of evaluation queries among the kept training rows (post-hoc check)

| evaluation set | queries (distinct after normalisation) | KEPT training rows that equal one of its queries | distinct questions | training slice | translated copies of those rows (via `src_qid`) in the kept training data |
|---|---:|---:|---:|---|---:|
| MIRACL-HN.v2 ar dev | 1000 (998) | 9 | 8 | train_tydi_ar 9 | 7 |
| NamaaMrTydi test | 918 (916) | 0 | 0 | - | 0 |

MIRACL-HN.v2 ar (our nDCG@10 over the full corpus, with titles) with and without the 8 evaluation queries that have an exact copy in the kept training data:

| model | all queries | without the leaked queries | difference |
|---|---:|---:|---:|
| bge-m3 | 79.0 | 79.2 | +0.2 |
| e5-base | 73.0 | 73.2 | +0.2 |
| e5-small | 72.4 | 72.6 | +0.2 |
| final_base | 70.7 | 71.0 | +0.2 |
| final_m3 | 77.9 | 78.1 | +0.2 |
| final_small | 68.3 | 68.5 | +0.3 |
| m3_b_s13 | 72.9 | 73.1 | +0.2 |
| v1 | 65.8 | 66.0 | +0.2 |
| wm_a75 | 75.7 | 75.9 | +0.2 |

The leaked rows are 8 distinct TyDi-ar questions; the table also shows how many translated copies of them (the Arabic or English version produced by the translation step, linked by `src_qid`) are in the kept training data and what the MIRACL-HN score would be without the leaked evaluation queries. Removing the leaked evaluation queries raises every model's MIRACL-HN score by 0.2-0.3 points, including models that never trained on them (bge-m3, e5-base, e5-small), so the queries are simply harder than average and the leak does not change the comparison between models; it is disclosed because the contamination filter was supposed to remove such rows.

## 9. Reranker pairing (TEST, top-50 -> `rr_ft_bce_s13`, the v1 fine-tuned cross-encoder)

| first stage | first stage ar->en | + reranker ar->en | first stage macro | + reranker macro | + reranker en->ar | + reranker ar->ar |
|---|---:|---:|---:|---:|---:|---:|
| bge-m3 | 63.3 | 75.3 | 74.8 | 84.5 | 84.4 | 87.6 |
| e5-base | 49.9 | 71.3 | 72.4 | 83.5 | 84.5 | 87.6 |
| e5-small | 41.4 | 68.9 | 66.9 | 82.7 | 83.8 | 87.2 |
| v2-m3 (pick) | 71.2 | 75.3 | 82.2 | 84.3 | 83.6 | 87.4 |
| v2-base (soup) | 66.6 | 74.4 | 80.2 | 84.0 | 83.8 | 87.1 |
| v2-small (soup) | 59.7 | 73.4 | 76.9 | 83.6 | 83.4 | 87.0 |

The reranker was fine-tuned for v1 and is unchanged. Reading: with this reranker the first-stage differences between v2-m3 and bge-m3 vanish (equal ar->en, macro within 0.2), because the cross-encoder already repairs most ranking errors of a strong first stage; the v2-m3 first-stage gain does not become a full-pipeline accuracy gain at top-50. Whether it allows a shorter candidate list (lower latency) was not measured. For the small models the first stage still matters after reranking (+3.1 and +4.5 ar->en for v2-base and v2-small over their bases).

## 10. What did not work

* **Negatives / batching / symmetric loss / hardness weighting on the v1 recipe:** no gain beyond noise (section 3.1).
* **KD from off-the-shelf bge-m3 into e5-small:** no gain (section 3.1). KD from the *fine-tuned* teacher works.
* **More translated data for the small models** (22k vs 12k questions): no TEST gain; only the large model profits.
* **MIRACL-ar replay, preservation loss, LoRA, frozen layers:** none lifts the held-out MIRACL slice above the raw fine-tune; WiSE interpolation is the only working lever, and it trades ar->en for the Arabic-only tasks along the curve of section 7.
* **Soup of three seeds at the pick's alpha:** scores marginally higher on the suite than the single seed (0.7612 vs 0.7486) but within the tie band; not adopted.
* **Preventing the loss on the Arabic reranking task:** NamaaMrTydi is still 3.4 below bge-m3 for v2-m3 and 9-10 below the base for both students.

## 11. Caveats, corrections and what is not verified

* **The MTEB(ara) lead of v2-m3 is a tie.** +0.31 with a task-bootstrap CI that includes 0 (section 6.2); two confirmatory seeds reproduce the point estimate, not a significance. Resampling 19 tasks is coarse.
* **#1 on the ar-en cross-lingual subsets is not achieved overall:** multilingual-e5-large (560M) scores 75.3 vs 75.0 for v2-m3. v2-base is first among the ≤300M models.
* **Students lose heavily on NamaaMrTydi** (about -9 to -10), partly offset by gains elsewhere; their Mean (TaskType) is 67.62 (v2-base, vs 67.28) and 65.42 (v2-small, **below** e5-small's 65.70).
* **The pick is a near-tie** (gap 0.0127 < 0.02 between the single-seed and soup variants); the rule decided, I would not claim more.
* **Selection inputs that are not independent of MTEB:** the ArSarcasm probes use the same source as the MTEB TweetSarcasmClassification task (disjoint splits); the suite was designed to mirror MTEB task types by request. No MTEB number was read by the selection, but the earlier alpha (0.75) had been scored on MTEB before the suite existed.
* **Machine-translated DEV and Gemma-translated training questions**: the style diagnostic (3.5) shows a constant offset, not a model-specific one; TEST for the cross-lingual slices uses human-translated XQuAD questions.
* **Corrections made during the work:** (a) the first LoRA run trained 615/615 steps and then crashed at the adapter merge (a bug in my code: the module handle was looked up through the sentence-transformers wrapper, which returned the already-injected encoder); nothing was saved, the dependent evaluations died on the missing path, and the driver's exit codes were masked by `run_docker.sh`, so it looked like "rc=0". The fix keeps the PeftModel handle, saves the adapter before the merge, checks that no LoRA parameter remains, and passed a gated self-test; the run was repeated with the identical recipe and entered the pool as registered. (b) The pre-registered tie band was strict (`< 0.02`) but the first selection code was inclusive; corrected, scores now print to 4 decimals. (c) The `guard_split` step of the last queue skipped five models whose cached rows came from the first (untitled) run; moved to `stale_untitled/`, recomputed with titles (bge-m3 sanity check against mteb passed), see PROGRESS.md.
* **Log footnote.** The queue logs print "MTEB result files for sc_...: 0" after each MTEB scoring. That count is wrong (a shell variable was overwritten by the step function); every scored model has all 19 task files and the board in section 6 is built from them (it requires all 19).
* **Residual training leak (disclosed, not fixed).** The contamination filter's near-duplicate detector skips words with more than 300 postings, so exact copies of short common-word questions can survive: 9 kept training rows (8 distinct questions, slice `train_tydi_ar`) equal a MIRACL-HN.v2 ar dev query exactly, none equal a NamaaMrTydi query (section 8). The effect on the reported MIRACL-HN numbers is at most 8 of 1,000 queries and v2 is below bge-m3 on that task anyway; the audit claim "0 near-duplicates left" is true for the detector used, not for exact copies. The `exclude_qids.json` list was not rebuilt and no model was retrained.
* **The first contamination-split labels under-counted.** Corrected labels (exact copies included) are used in section 8; the earlier figures (816 contaminated / 184 clean) are not used anywhere. The earlier statements "42% verbatim" in `PROGRESS.md` and two docstrings referred to the 408 near-duplicates against v1's training pool; they were corrected in the code, and `PROGRESS.md` (append-only) carries a dated correction.
* **Licences of the released weights are TBD**; training-data licences are in section 2.1. `NamaaMrTydiReranking` shares its source (Mr.TyDi) with the TyDi training questions; MIRACL near-duplicates were filtered, Mr.TyDi test items were not audited individually.
* **Not verified:** external replication; any effect on retrieval domains other than Wikipedia-style QA; behaviour at sequence lengths above 256 beyond what MTEB(ara) covers (max length 512 at evaluation); the soup variant on TEST and the MTEB(ara); multiple-comparison correction for the number of tables.

## 12. Pre-registration trail (timestamps from `PROGRESS.md`)

| when (Spark clock, +03) | what was written down | where |
|---|---|---|
| 2026-11-04 03:59 | Translation set FROZEN at 11,875 translations (ablation data); later extended to the full 22,313 and filtered (05:23 contamination filter built before the final runs) | PROGRESS.md 03:59 / 05:23 |
| 2026-11-04 05:40 | Guards (MIRACL-HN.v2 ar, NamaaMrTydi, SemRel24) reported on final candidates only, never used for selection; mitigations judged on dev + a non-MTEB(ara) proxy (mMARCO-ar subsample) | PROGRESS.md 05:40 |
| 2026-11-04 ~11:05 | WiSE rule fixed BEFORE any guard eval: dev macro within 0.5 of raw AND proxy within 1.0 of bge-m3 -> alpha 0.75 (the first v2-m3 candidate, final_m3_wa75_s13) | PROGRESS.md 11:05 |
| 2026-11-04 ~14:00 | PRE-REGISTERED rule for the replay / preservation levers: qualifies if devM >= bge-m3 devM - 1.0, dev ar->en >= 79.0, dev macro >= m3_b - 1.5; highest dev macro among the qualifying | PROGRESS.md 14:00 (applied 17:39: qualifiers wm_a75, wm_a50, w_m3_p_a75/a50 -> wm_a75) |
| 2026-11-04 17:46 | Coordinator re-prioritisation: score final_m3_wa75_s13 on MTEB(ara) first (Mean(Task) 72.04 vs 71.98 = a tie), then parameter-limited runs | PROGRESS.md 17:46 |
| 2026-11-04 ~19:05 | PRE-REGISTRATION of the 6-component validation suite, normalisation, pool, eligibility, tie-break (XNLI train-subsample deviation flagged: mteb's XNLI scores validation too) | PROGRESS.md 19:05 |
| 2026-11-04 ~19:20 | CORRECTION (before any suite number existed): Arabic STS-B validation split from Omartificial-Intelligence-Space/Arabic-stsb (stsb_multi_mt has no Arabic) | PROGRESS.md 19:20 |
| 2026-11-04 ~21:35 | AMENDMENT 2 (after the LoRA crash): rerun LoRA with the fixed merge; strict tie band; alpha 0.30 (+0.35 only if 0.30 fails eligibility) for the winning family only; full-pool final select | PROGRESS.md 21:35 |
| 2026-11-04 ~22:10 | PRE-REGISTRATION 3: confirmatory seeds 16/17 with the frozen recipe, student rule (soup unless its dev macro is >0.5 below its seeds' mean), post-hoc blend curve, MTEB paired statistics | PROGRESS.md 22:10 |
| 2026-11-04 22:46:54 | RECIPE FROZEN: family wise, alpha 0.4, pick w_m3b_a40 (score 0.7486; runner-up soup_a40 0.7612; gap 0.0127 < 0.02 -> ingredients clause) | PROGRESS.md 22:46 |
| 2026-11-04 23:13 | Confirmatory seeds 16 and 17 start (after the freeze line) | PROGRESS.md 23:13 |
| 2026-11-05 01:13 / 02:56 / 03:04 | Teacher re-embedded from final_m3; student picks by the registered rule: final_base = st_b_soup, final_small = st_s_soup | PROGRESS.md 01:13, 02:56, 03:04 |
| 2026-11-05 06:38 / 07:04 | CORRECTION of the drive16 guard-split step (coordinator-approved): 5 stale untitled cache rows recomputed with titles; sanity check bge-m3 MIRACL-HN 79.0 vs mteb 78.9 passed | PROGRESS.md 06:38, 07:04 |
| 2026-11-05 07:2x | CORRECTION (coordinator's exact-match checks): the contamination labels missed exact copies (893/1000 exact TyDi-train copies vs 814 flagged); 9 kept training rows equal MIRACL-HN dev queries; labels recomputed, leak disclosed, not retrained | PROGRESS.md 07:2x |

## 13. Cost and files

* **GPU time (wall-clock on a shared GB10, so an upper bound on exclusive time):** 46 training runs = 12.2 h of training; the 10 MTEB(ara) scorings of v2 and alpha-curve models = 1.9 h of summed task evaluation_time; the gated job queue (drive scripts 2-17: training, merges, DEV/TEST/devM/mixed/proxy/suite evaluations, MTEB scorings, teacher embeddings, guard split, reranker) adds up to about 19 h of step wall-clock (it includes short gate waits), plus about 3 h of ungated ablation jobs before the gate existed (estimate) - **roughly 22 h of GPU wall-clock in total**. The Gemma translation (about 2 h of llama-server time at concurrency 2) is not in that figure. Calendar span: 2026-11-04 01:30 to 2026-11-05 about 07:30 (30 h), because jobs were serialised behind other agents' work (one GPU job machine-wide, MemAvailable >= 25 GB).
* Everything is under `~/projects/qubit-ai-lab/runs/retrieval/v2/` (`results/`, `mteb/`, `data/`, `logs/`, `models/`); code in `lab/retrieval/v2/`; the final models are `models/final_m3`, `models/final_base`, `models/final_small` (clean sentence-transformers folders). The post-hoc analysis files are `results/alpha_curve.json`, `results/mteb_paired.json`, `results/mteb_board.json`, `results/confirmatory.json`, `results/suite_pick.json`.
