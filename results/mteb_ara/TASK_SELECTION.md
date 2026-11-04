# MTEB(ara, v1): task selection

mteb 2.26.33. Every task with an Arabic language code was reviewed: 103 in total. That is the 72 returned by
`get_tasks(languages=["ara"])`, 12 older versions of tasks that were superseded, and 19 tasks whose Arabic is coded `arb`, `arq`, `ary` or another dialect code
(FLORES, NTREX, Belebele, XQuAD, SemRel24, SIB-200, AfriSenti, ...), which the `ara` filter silently misses. 19 tasks are in the benchmark (27 scored task/subset cells).

## What was asked in #1875 and #2135, and what this does

| Reviewer point | How it is handled |
|---|---|
| Fewer than 25 tasks (#1875) | 19 tasks, 27 scored (task, subset) cells. Recent language benchmarks are of similar size (Thai 15, Japanese 15). If 25 is still wanted, the closest candidates are listed under "Near misses". |
| Justify the selection; translated or LM-generated data? (#2135) | Table below, one row per task. The benchmark has no machine-translated and no LM-generated task. The remaining translated tasks (XQuAD, XNLI, FLORES, NTREX, SIB-200) are professional or expert human translations. |
| Coverage of task types, clustering missing (#1875, #2135) | All seven text types are covered, including clustering. The only existing clustering task with Modern Standard Arabic is a 30-sample HUME subset, so a `SIB200ClusteringS2S.v2` variant is added (see below). |
| Too many sentiment / hate-speech tasks (#2135) | 4 classification tasks, each a different label space (sentiment, emotion, sarcasm, review rating). Hate speech, language-ID and the star-rating review duplicates are out. |
| MassiveIntent / MassiveScenario are poor translations | Excluded. |
| Both HardNegatives and the full variant | Only `MIRACLRetrievalHardNegatives.v2`. |
| Bitext mining pulled in 21 languages | Bitext subsets are listed explicitly and are Arabic <-> English only. The benchmark language set is {ara, arb, eng}. |
| Licences (#2135) | Column below plus the "Licences" section. Non-commercial or no-derivative tasks are excluded. |
| A few models must be run so the leaderboard is not empty (#2135) | 10 models evaluated, see `LEADERBOARD.md` and the results-repo files. |
| Stalled with no follow-up | The branch is complete (benchmark, tests, results), so a reviewer can act without waiting for the author. |

## Selection criteria

1. Human-written or human-translated data. No machine translation, no LM-generated queries.
2. Not tiny (more than a few hundred test items) and not noisy (spot-checked, see below).
3. No near-duplicates (same source, same sentences, or the same task in two sizes).
4. Licence stated and not non-commercial / no-derivatives, or flagged.
5. Affordable: prefer the HardNegatives / downsampled variants. All 20 tasks run in minutes on one GPU for a small model.
6. Arabic only, plus Arabic <-> English cross-lingual subsets. Subsets are always listed explicitly (a plain `languages=["ara"]` filter would, for example, pull in MLQA `ara-deu`).

In subset names the first language is the corpus and the second the query (`MLQARetrieval ara-eng` = Arabic documents, English queries).

## Data checks done

Run on the exact subsets and splits used (scripts in `scripts/check_*.py`): ids in qrels exist in the corpus, no queries without qrels, empty or duplicated texts, script of
queries and documents (Arabic vs Latin as expected for each direction), score ranges, label balance, and 3 random examples per subset read by hand.

* MLQARetrieval `ara-deu` is broken (German queries paired with unrelated Arabic passages, as reported in #5585). `ara-ara`, `ara-eng` and `eng-ara` read correctly. Only those three are used.
* MIRACLRetrievalHardNegatives.v2 (ar): all 1,982 positive judgments are in the corpus (1,173 judged-negative pairs are not, which is harmless); 3,366 duplicate passage texts out of 192k.
* SadeemQuestionRetrieval: 4,805 of 22,979 documents are exact duplicates under different ids. Excluded.
* MKQARetrieval and MintakaRetrieval: the "documents" are answer strings of about 12 characters (a name or number). Excluded.
* NamaaMrTydiReranking: 2 empty and 1 duplicated document out of 4,586. Left as is.
* MIRACL Arabic dev vs TyDi-QA: 408 of the 1,000 queries in MIRACLRetrievalHardNegatives.v2 (ar) are verbatim TyDi-QA Arabic train questions (found while auditing a TyDi-trained model). Models trained on TyDi-QA or Mr. TyDi train have seen them; see the contamination comment in the drafts.
* XPQARetrieval is noisy by design (short answers, some documents are product-attribute JSON in `eng-ara`). Kept as the only product-domain task, flagged here.
* Belebele and XQuAD: one positive per query, all ids resolve.

## What 10 models say about the tasks

Ten models were run on every task (`LEADERBOARD.md`, `task_diagnostics.csv`). Ranges are in the Included table. What it shows:

* NTREXBitextMining was dropped after the first full run: it ranks the 10 models almost exactly like FLORES (Spearman 0.96) and both are near the ceiling for multilingual models (FLORES median 98, NTREX 95). FLORES is kept because it is the standard test; it separates Arabic-only models (about 27-43) from multilingual ones (87-100) and says little beyond that.
* MLQA, Belebele and XPQA rank the models almost identically (Spearman 0.96-0.98). Their cross-lingual subsets split multilingual models from Arabic-only ones, so they carry related information. They are kept because they cover different domains (Wikipedia QA, FLORES passages, product questions).
* STS22, SemRel24, XNLI, ArEntail, AJGT, TweetSarcasm, SIB200 clustering and Tatoeba agree weakly with the rest of the benchmark (rho below 0.3 against the mean of the other tasks). The Arabic-only models do well on them while the large multilingual retrievers do not. They measure something different, so they stay for task-type coverage; their rankings are noisy with only 193-1,949 test items.
* XQuAD is the easiest retrieval task (90.6 median, 240 paragraphs) and most models have seen SQuAD-style data; it still spreads models over 77-96.
* With 10 models the correlations are rough guides, not estimates with error bars.


## Included (19)

| Task | Type | Licence (mteb) | Sample creation | Score range over 10 models | Why |
|---|---|---|---|---:|---|
| FloresBitextMining | BitextMining | cc-by-sa-4.0 | created | 27-100 | Human-translated FLORES-200 devtest, arb<->eng, both directions. Near-saturated for multilingual models; kept because it is the standard bitext test and splits Arabic-only models from multilingual ones. |
| Tatoeba | BitextMining | cc-by-2.0 | found | 14-89 | Short crowd-sourced sentence pairs (LASER 1k test), ara->eng. |
| AJGT.v2 | Classification | afl-3.0 | found | 70-88 | Human-annotated tweet sentiment (MSA + Jordanian dialect); one sentiment-style task, v2 fixes label errors. |
| HotelReviewSentimentClassification.v2 | Classification | not specified | found | 44-54 | Longer review text (HARD); labels derived from star ratings. The only reviews task kept. |
| TweetEmotionClassification.v2 | Classification | not specified | found | 40-58 | Human-annotated 8-way emotion in tweets; different label space from the sentiment task. |
| TweetSarcasmClassification.v2 | Classification | mit | found | 54-66 | Human-annotated sarcasm (mixed dialects), MIT licence; imbalanced, harder pragmatic task. |
| SIB200ClusteringS2S.v2 | Clustering | cc-by-4.0 | human-translated and localized | 25-42 | Only text clustering task for MSA: SIB-200 (FLORES, human translated) topics, 1,434 sentences, 14 topics. New .v2 added because SIB200ClusteringS2S has no arb_Arab. |
| ArEntail | PairClassification | not specified | found | 73-92 | Manually curated Arabic entailment from news headlines, native Arabic (not translated); 1,000 test pairs. |
| XNLI | PairClassification | not specified | created | 65-85 | Professional translation of MNLI; Arabic test, 1,365 pairs. Licence is unspecified in mteb (upstream may be non-commercial, see Licences below). |
| NamaaMrTydiReranking | Reranking | cc-by-sa-3.0 | found | 52-80 | Human-annotated TyDi-QA Arabic questions as reranking; 918 queries. Replaces MrTidyRetrieval (same source, 2.1M passages). |
| BelebeleRetrieval | Retrieval | cc-by-sa-4.0 | created | 45-91 | Expert-written questions over FLORES passages: MSA monolingual plus Arabic<->English; 900 queries, 488 passages. Dialect and romanised variants left out. |
| MIRACLRetrievalHardNegatives.v2 | Retrieval | cc-by-sa-4.0 | created | 12-79 | Expert-annotated Arabic Wikipedia search; 1,000 queries over 192k passages; the HardNegatives downsample keeps the run cheap. |
| MLQARetrieval | Retrieval | cc-by-sa-3.0 | found | 26-63 | Human-annotated Wikipedia QA. Only ara-ara, ara-eng, eng-ara used; ara-deu (and the other non-English pairs) left out, ara-deu qrels are wrong (issue #5585, checked). |
| WebFAQRetrieval | Retrieval | cc-by-4.0 | found | 41-80 | Web FAQ question-answer pairs, 10,000 queries over 142k answers; adds the FAQ/web domain. Answers are FAQ text found on the web (not hand-judged). |
| XPQARetrieval | Retrieval | cdla-sharing-1.0 | found | 20-38 | Only product-domain task (customer questions vs answers); noisy by nature, kept for domain coverage. ara-ara, ara-eng, eng-ara. |
| XQuADRetrieval | Retrieval | cc-by-sa-4.0 | created | 77-96 | Professional translation of SQuAD v1.1 dev; 1,186 queries. Small corpus (240 paragraphs), so it is the easiest retrieval task here (see results). |
| STS17 | STS | not specified | created | 56-80 | SemEval-2017 human-scored pairs: ar-ar and en-ar (cross-lingual), 250 pairs each. |
| STS22.v2 | STS | not specified | found | 49-65 | SemEval-2022 news article pairs, Arabic subset; 193 pairs, long documents. |
| SemRel24STS | STS | not specified | created | 41-68 | Human-annotated relatedness (SemRel 2024), MSA subset `arb`; 595 pairs. The `ary`/`arq` dialect subsets are left for a dialect extension. |

## Excluded (84)

| Task | Type | Licence (mteb) | Sample creation |  Why |
|---|---|---|---|---|
| OmnilingualASRA2TRetrieval | Any2AnyMultilingualRetrieval | cc-by-4.0 | created | image/audio modality, not a text-embedding task |
| OmnilingualASRT2ARetrieval | Any2AnyMultilingualRetrieval | cc-by-4.0 | created | image/audio modality, not a text-embedding task |
| WITI2TRetrieval | Any2AnyMultilingualRetrieval | cc-by-sa-4.0 | found | image/audio modality, not a text-embedding task |
| WITT2IRetrieval | Any2AnyMultilingualRetrieval | cc-by-sa-4.0 | found | image/audio modality, not a text-embedding task |
| XM3600I2TRetrieval | Any2AnyMultilingualRetrieval | cc-by-sa-4.0 | found | image/audio modality, not a text-embedding task |
| XM3600T2IRetrieval | Any2AnyMultilingualRetrieval | cc-by-sa-4.0 | found | image/audio modality, not a text-embedding task |
| CommonVoiceMini17A2TRetrieval | Any2AnyRetrieval | cc0-1.0 | found | image/audio modality, not a text-embedding task |
| CommonVoiceMini17T2ARetrieval | Any2AnyRetrieval | cc0-1.0 | found | image/audio modality, not a text-embedding task |
| CommonVoiceMini21A2TRetrieval | Any2AnyRetrieval | cc0-1.0 | found | image/audio modality, not a text-embedding task |
| CommonVoiceMini21T2ARetrieval | Any2AnyRetrieval | cc0-1.0 | found | image/audio modality, not a text-embedding task |
| FleursA2TRetrieval | Any2AnyRetrieval | apache-2.0 | found | image/audio modality, not a text-embedding task |
| FleursA2TRetrieval.v2 | Any2AnyRetrieval | apache-2.0 | found | image/audio modality, not a text-embedding task |
| FleursT2ARetrieval | Any2AnyRetrieval | apache-2.0 | found | image/audio modality, not a text-embedding task |
| FleursT2ARetrieval.v2 | Any2AnyRetrieval | apache-2.0 | found | image/audio modality, not a text-embedding task |
| GoogleSVQA2TRetrieval | Any2AnyRetrieval | cc-by-4.0 | found | image/audio modality, not a text-embedding task |
| GoogleSVQT2ARetrieval | Any2AnyRetrieval | cc-by-4.0 | found | image/audio modality, not a text-embedding task |
| SIBFLEURS | AudioClassification | cc-by-4.0 | found | image/audio modality, not a text-embedding task |
| BibleNLPBitextMining | BitextMining | cc-by-sa-4.0 | created | Single religious-text domain; FLORES, NTREX and Tatoeba cover Arabic-English mining. |
| IWSLT2017BitextMining | BitextMining | cc-by-nc-nd-4.0 | found | cc-by-nc-nd-4.0 (non-commercial, no derivatives). |
| NTREXBitextMining | BitextMining | cc-by-sa-4.0 | human-translated and localized | Near-duplicate of FLORES for this purpose: human-translated ar<->en sentence pairs, both saturated (medians 95.5 and 98.4), and the model ranking is almost identical (Spearman 0.96 over the 10 models run). FLORES kept. |
| WebFAQBitextMiningQAs | BitextMining | cc-by-4.0 | human-translated | Pairs are aligned with LaBSE similarity (labse_similarity column), which favours LaBSE; Arabic pairs are mostly with non-English languages. |
| WebFAQBitextMiningQuestions | BitextMining | cc-by-4.0 | human-translated | As WebFAQBitextMiningQAs. |
| AJGT | Classification | afl-3.0 | found | Superseded by AJGT.v2. |
| AfriHateClassification | Classification | cc-by-4.0 | found | As AfriSenti (dialect subsets only, hate speech). |
| AfriSentiClassification | Classification | cc-by-4.0 | found | Algerian/Moroccan dialect subsets of an African-language collection; belongs in a dialect-focused benchmark. |
| AfriSentiLangClassification | Classification | cc-by-nc-sa-4.0 | found | Language identification and dialect-only. |
| HUMEMultilingualSentimentClassification | Classification | not specified | found | HUME subset of MultilingualSentimentClassification. |
| HotelReviewSentimentClassification | Classification | not specified | found | Superseded by HotelReviewSentimentClassification.v2. |
| LanguageClassification | Classification | not specified | found | Language identification is not an embedding-quality task (reviewer comment on #2135). |
| MassiveIntentClassification | Classification | apache-2.0 | human-translated and localized | Human-translated/localised from English; reviewers flagged Arabic translation quality (#1875, #2135). |
| MassiveScenarioClassification | Classification | apache-2.0 | human-translated and localized | As MassiveIntentClassification. |
| MultiHateClassification | Classification | cc-by-4.0 | created | Template-based functional tests (Constructed domain), 1,000 examples; hate/sentiment tasks were asked to be cut to a few (#2135). |
| MultilingualSentimentClassification | Classification | not specified | found | Mixed-source data with derived labels and no stated licence; 'ara' also covers dialect codes. |
| OnlineStoreReviewSentimentClassification | Classification | not specified | found | Superseded by OnlineStoreReviewSentimentClassification.v2. |
| OnlineStoreReviewSentimentClassification.v2 | Classification | not specified | found | Review sentiment with star-derived labels, same genre as HotelReview (one reviews task kept). |
| RestaurantReviewSentimentClassification | Classification | not specified | found | Superseded by RestaurantReviewSentimentClassification.v2. |
| RestaurantReviewSentimentClassification.v2 | Classification | not specified | found | As OnlineStoreReview: same genre as the kept reviews task. |
| SIB200Classification | Classification | cc-by-sa-4.0 | human-translated and localized | Superseded by SIB200Classification.v2. |
| SIB200Classification.v2 | Classification | cc-by-4.0 | human-translated and localized | Superseded by SIB200Classification.v3. |
| SIB200Classification.v3 | Classification | cc-by-4.0 | human-translated and localized | Same sentences and labels as the clustering task kept; avoids a near-duplicate. |
| TweetEmotionClassification | Classification | not specified | found | Superseded by TweetEmotionClassification.v2. |
| TweetSarcasmClassification | Classification | mit | found | Superseded by TweetSarcasmClassification.v2. |
| TweetSentimentClassification | Classification | cc-by-3.0 | found | Only 256 test tweets for Arabic; kept tweet tasks are larger. |
| HUMESIB200ClusteringS2S | Clustering | cc-by-4.0 | found | 30 samples (human-evaluation subset), too small. |
| SIB200ClusteringS2S | Clustering | cc-by-sa-4.0 | human-translated and localized | No arb_Arab subset (only romanised arb_Latn and dialects); superseded here by SIB200ClusteringS2S.v2. |
| JinaVDRAirbnbSyntheticRetrieval | DocumentUnderstanding | cc0-1.0 | found | image/audio modality, not a text-embedding task |
| JinaVDRArabicChartQARetrieval | DocumentUnderstanding | mit | found | image/audio modality, not a text-embedding task |
| JinaVDRArabicInfographicsVQARetrieval | DocumentUnderstanding | mit | found | image/audio modality, not a text-embedding task |
| JinaVDRGitHubReadmeRetrieval | DocumentUnderstanding | multiple | found | image/audio modality, not a text-embedding task |
| JinaVDRTweetStockSyntheticsRetrieval | DocumentUnderstanding | not specified | found | image/audio modality, not a text-embedding task |
| JinaVDRWikimediaCommonsDocumentsRetrieval | DocumentUnderstanding | multiple | found | image/audio modality, not a text-embedding task |
| MIRACLVisionRetrieval | DocumentUnderstanding | cc-by-sa-4.0 | created | image/audio modality, not a text-embedding task |
| EmotionAnalysisPlus | MultilabelClassification | cc-by-4.0 | created | Superseded by EmotionAnalysisPlus.v2. |
| EmotionAnalysisPlus.v2 | MultilabelClassification | cc-by-4.0 | created | Algerian/Moroccan dialect subsets only (multilabel). |
| MIRACLReranking | Reranking | cc-by-sa-4.0 | created | Same queries and documents as MIRACLRetrieval, 290k docs; near-duplicate of the kept MIRACL task. |
| MVLSIBSent2Img | Reranking | cc-by-sa-4.0 | multiple | image/audio modality, not a text-embedding task |
| MultiLongDocReranking | Reranking | mit | found | As MultiLongDocRetrieval (LM-generated queries, long documents). |
| MIRACLRetrieval | Retrieval | cc-by-sa-4.0 | created | Full corpus (millions of passages) and the same queries as the HardNegatives.v2 task kept. |
| MIRACLRetrievalHardNegatives | Retrieval | cc-by-sa-4.0 | created | Older HardNegatives version; .v2 is the maintained one (reviewer: do not include both). |
| MKQARetrieval | Retrieval | cc-by-3.0 | found | Documents are bare answer strings (avg 12 chars, e.g. a person name); measures memorised facts, not retrieval. |
| MMarcoRetrievalMultilingual | Retrieval | apache-2.0 | machine-translated | Machine-translated MS MARCO. |
| MintakaRetrieval | Retrieval | cc-by-4.0 | human-translated | Same problem as MKQA: documents are bare answer entities (avg 13 chars). |
| MrTidyRetrieval | Retrieval | cc-by-sa-3.0 | found | 2.1M-passage Arabic corpus; same source as NamaaMrTydiReranking which is kept. |
| MultiLongDocRetrieval | Retrieval | mit | found | LM-generated queries and very long documents (beyond the 512-token limit of most Arabic models). |
| MultilingualNanoArguAnaRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoClimateFeverRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoDBPediaRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoFEVERRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoFiQA2018Retrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoHotpotQARetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoMSMARCORetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR. |
| MultilingualNanoNFCorpusRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoNQRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoQuoraRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoSCIDOCSRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoSciFactRetrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| MultilingualNanoTouche2020Retrieval | Retrieval | cc-by-4.0 | machine-translated | Machine-translated NanoBEIR (translation quality, and not native Arabic). |
| PublicHealthQA | Retrieval | cc-by-nc-sa-3.0 | found | cc-by-nc-sa-3.0 and only 172 Arabic samples (queries plus documents). |
| SadeemQuestionRetrieval | Retrieval | not specified | found | Licence not specified, no dataset card, 21% of documents are exact duplicates under different ids (4,805 of 22,979), which makes the qrels ambiguous. |
| HUMESTS22 | STS | not specified | found | HUME subset of STS22. |
| STS22 | STS | not specified | found | Superseded by STS22.v2. |
| VisualSTS17Eng | VisualSTS(eng) | not specified | rendered | image/audio modality, not a text-embedding task |
| STS17MultilingualVisualSTS | VisualSTS(multi) | not specified | rendered | image/audio modality, not a text-embedding task |
| VisualSTS17Multilingual | VisualSTS(multi) | not specified | rendered | image/audio modality, not a text-embedding task |

## Added code

* `SIB200ClusteringS2S.v2`: the existing SIB200ClusteringS2S has no `arb_Arab` (only romanised `arb_Latn` and dialects) because it predates the 14-topic `mteb/sib200.v3` data
  that `SIB200Classification.v3` already uses. The v2 task is the same task on that data (all 205 language pairs), 14 topics, 1,434 sentences for `arb_Arab`.
  It is a separate commit so it can be reviewed or dropped independently. If it is dropped, the fallback is the existing task restricted to the eight Arabic-dialect subsets (acm, acq, aeb, ajp, apc, ars, ary, arz).
* `NamaaMrTydiReranking.adapted_from = ["MrTidyRetrieval"]`: it is the Arabic Mr. TyDi test split, but mteb did not link them, so models trained on Mr. TyDi (e5, bge-m3, ...) were shown as zero-shot on it.
* `ModelMeta` for `mzahran1/arabic-english-e5-small-retrieval` (see `mzahran_models.py`) and for `omarelshehy/Arabic-Retrieval-v1.0`.

## Licences

Column "Licence (mteb)" is what mteb's metadata says. Stated and permissive: MIRACL, XQuAD, MLQA, Belebele, FLORES, NTREX, Tatoeba, WebFAQ, SIB-200 (CC BY / CC BY-SA), XPQA (CDLA-Sharing), AJGT (AFL-3.0), TweetSarcasm (MIT).
Not stated in mteb (and I did not find a licence on the Hugging Face card): STS17, STS22.v2, SemRel24STS, XNLI, ArEntail, TweetEmotion, HotelReview. For XNLI the upstream project
may restrict commercial use; I did not verify this. XNLI is already used by MTEB(Multilingual) and MTEB(tha). If reviewers prefer, XNLI can be dropped without changing anything else.

## Known limitations

* One clustering task. There is no larger Arabic clustering dataset in mteb; ORCA (mentioned in #2135) has not been added.
* Modern Standard Arabic dominates. AJGT, TweetEmotion, TweetSarcasm and STS22 contain dialect, but there is no dialect benchmark section. A natural v2 is to add the dialect subsets of SemRel24 (`ary`, `arq`), AfriSenti and Belebele (`arz`, `apc`, ...).
* XQuAD is saturated for most models (see results) and is SQuAD-derived, so many models have seen SQuAD train.
* WebFAQ and XPQA labels are "found" (FAQ pairs, product Q&A), not hand-judged.
* Scores of models that trained on MIRACL, Mr. TyDi or SQuAD (e5, bge-m3, the fine-tuned model) are flagged non-zero-shot by `ModelMeta.training_datasets` on MIRACL, NamaaMrTydi and XQuAD.
