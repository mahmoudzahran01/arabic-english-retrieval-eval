# Data card

Everything in this repository is built from **public datasets**. The raw datasets are not committed; the build scripts download
them from their public sources (pinned revisions / checksums) and rebuild every table. What *is* committed are small derived
artefacts (per-system metrics, id-only indexes, judge verdicts) described at the end.

## Sources and licences

Licences were checked on 2026-11-03 against the Hugging Face dataset cards and the upstream repositories.

| Dataset | What it is | Used for | Licence | Source (pinned revision) |
|---|---|---|---|---|
| **TyDi QA, GoldP** (Arabic, English) | native questions with a gold passage, Wikipedia | train + dev pools; **test = official dev** (921 ar, 440 en queries) | Apache-2.0 | `google-research-datasets/tydiqa`, `secondary_task`, rev `da78f23f9119` ([upstream](https://github.com/google-research-datasets/tydiqa)); Clark et al., 2020 |
| **ARCD** | Arabic reading-comprehension questions on Wikipedia | train + dev pools; **test = validation** (702 queries) | MIT | `hsseinmz/arcd`, rev `cc6906b6eda5` ([upstream](https://github.com/husseinmozannar/SOQAL)); Mozannar et al., 2019 |
| **SQuAD v1.1** | English reading comprehension on Wikipedia | train + dev pools (at most 2 questions per paragraph); distractor passages for the test pool | CC BY-SA 4.0 | `rajpurkar/squad`, rev `7b6d24c440a3`; Rajpurkar et al., 2016 |
| **XQuAD** (Arabic, English) | 1,190 SQuAD-dev questions on 240 paragraphs, professionally translated; the two languages are *parallel* | **test only** (the clean ar->ar, en->en, ar->en, en->ar slices, 4 x 1,190 queries); also the whole corpus of the RAG-evaluation part | CC BY-SA 4.0 | `google/xquad`, rev `51adfef1c128`; GitHub `google-deepmind/xquad` (its README: CC BY-SA 4.0); Artetxe et al., 2020 |

Models used (off the shelf or as fine-tuning starting points) keep their own licences: `intfloat/multilingual-e5-small` (MIT),
`BAAI/bge-m3` (MIT), `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0), `BAAI/bge-reranker-v2-m3` (Apache-2.0),
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0). No weights are distributed here.

## What the retrieval benchmark contains

| split | passages (ar / en) | queries | contents |
|---|---|---:|---|
| train pool | 5,425 / 5,227 | 20,918 (16,419 monolingual + 4,499 LLM-translated cross-lingual) | TyDi-ar/en train, ARCD train, SQuAD train |
| dev pool | 1,982 / 2,055 | 3,663 (incl. 500 translated) | held-out articles of the same sources; model selection only |
| test pool | 4,325 / 5,492 | 6,823 | TyDi-ar dev (921), TyDi-en dev (440), ARCD validation (702), XQuAD x 4 directions (4 x 1,190) |

Totals: 24,506 passages and 31,404 queries (`results/retrieval/data_card/manifest.json`). A query is only ever searched against the
passages of its own pool, in the passage language of its slice.

**Cross-lingual queries.** *Test* queries in the cross-lingual directions are XQuAD's human translations (parallel questions over parallel
paragraphs, so the gold passage is the English or Arabic twin of the original). *Train/dev* cross-lingual queries are machine
translations (Gemma-4-26B-A4B, `retrieval/translate.py`) of real questions, 5,000 in total, paired with the original gold passage. Every
translation is script-checked (target language, not copied, numbering intact); they are clean Modern Standard Arabic and clean English.

## Split and leakage procedure (`retrieval/build_data.py`, audited by `retrieval/audit_leakage.py`)

1. **Whole articles, not paragraphs.** Each source is split by a hash (seed 13) of the article title into train / dev / dev-background /
   test-background pools, so paragraphs of one article never straddle pools.
2. **Test sources win.** Every article that appears in a test source is removed from train and dev in every source (3,016 rows dropped).
3. **Duplicates resolve by priority** test > dev > train: a duplicate passage or question that lives in a higher-priority pool is dropped
   from the lower one.
4. **Independent audit** over the final tables (it does not reuse the build logic): gold passage inside its own pool (0 violations);
   shared passage ids across pools (0); test-gold article titles that occur in train (0); question texts shared across splits (1: a
   translated duplicate between dev and train); near-duplicate passages by 6-gram containment >= 50% (26 of 9,817 test passages, 14 gold
   queries); near-duplicate questions at Jaccard >= 0.8 (8 test questions, generic ones like "Where was he born?").
5. **Sensitivity.** Removing the 14 affected test queries changes every system's macro MRR@10 by +0.01 to +0.02 points (RESULTS.md, Appendix D).

The split is a pure function of the pinned public datasets and the seed; the only non-deterministic ingredient is the LLM translation of
train/dev cross-lingual queries (see below).

## The RAG-evaluation corpus (`rageval/build_data.py`)

XQuAD only (seed 42): 240 paragraphs -> a **220-passage corpus** (each paragraph kept in exactly one language at random, 110 ar / 110 en),
**180 evaluation questions** (160 answerable, 40 each of ar->ar, ar->en, en->en, en->ar; 20 unanswerable, whose paragraph is removed
from the corpus), and a **932-question pool** used only to build the planted-error set for judge calibration, so judge tuning never
sees the questions the RAG system is scored on. 186 planted-error items were built from verified-correct answers; 6 were excluded for
label hygiene, leaving 180 labelled items. The files are downloaded and checksum-verified by `rageval/download_data.py`.

## What is committed, and under which licence

| Path | Contents | Contains dataset text? |
|---|---|---|
| `results/retrieval/*.json`, `*.npz` | per-system, per-slice metrics and per-query metric arrays | no |
| `results/retrieval/data_card/` | manifest, leakage audit, near-duplicate passage ids, id-only index of the test queries | no (ids and counts; the audit quotes six short question pairs) |
| `results/rageval/answers_*.jsonl` | generated answers, retrieved passage ids / ranks / scores | questions and reference answers (short XQuAD strings); **passage text removed** |
| `results/rageval/data/meta_items.jsonl` | the planted-error set: 180 labelled items | questions, reference answers, generated / edited answers; **passages and reference sentences removed** |
| `results/rageval/judge_*/` | per-item judge verdicts and raw judge replies | short quotations in some replies |

The few strings above that originate from XQuAD / SQuAD / TyDi QA (questions, short answer spans) stay under the licence of their source
(CC BY-SA 4.0 for XQuAD and SQuAD, Apache-2.0 for TyDi QA) with attribution to the papers cited in the table; the code is MIT.

**Not redistributed:** the raw datasets, the passage and query tables built from them, and the 5,000 LLM-translated training/dev queries
(`xling_translations.jsonl`). The latter are derivatives of CC BY-SA 4.0 and Apache-2.0 questions; regenerating them takes ~2 h with
any capable OpenAI-compatible LLM (`scripts/retrieval/01_data.sh`) and will give slightly different translations, hence slightly
different fine-tuning numbers than the published ones (the committed result files are the published ones).

## Citations

* Clark et al. (2020). *TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages.* TACL.
* Mozannar, Hajal, Maamary, Hajj (2019). *Neural Arabic Question Answering.* ACL ArabicNLP workshop (ARCD).
* Rajpurkar, Zhang, Lopyrev, Liang (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP.
* Artetxe, Ruder, Yogatama (2020). *On the Cross-lingual Transferability of Monolingual Representations.* ACL (XQuAD).
