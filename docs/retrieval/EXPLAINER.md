# Retrieval fine-tuning explained - a hands-on guide built on the experiments in this repository

Audience: someone comfortable with Python and basic ML who has used embeddings and RAG but has not trained a retriever.
Every number below was measured in this lab (see [`RESULTS.md`](RESULTS.md) for the full tables, [`REPRODUCING.md`](REPRODUCING.md) for the exact
commands). When a number appears, it is a **TEST** number on queries/passages that never appeared in training, unless it says "dev".
Points are percentage points of the metric x 100 (e.g. MRR@10 = 0.768 is written 76.8).

---

## 1. The pipeline we are studying

```
user question
   |
   v
[stage 1: candidate generation]   BM25 (lexical)  +  dense bi-encoder (embeddings)   -> merge with RRF -> ~50 candidates
   |
   v
[stage 2: rerank]                 cross-encoder scores (question, passage) pairs        -> top ~5-10
   |
   v
[stage 3: generation]             LLM reads the top passages and answers
```

Stage 1 must be **cheap and high-recall** (touch millions of passages in milliseconds). Stage 2 can be **expensive and high-precision**
(it only sees ~50 passages). Stage 3 can only use what stages 1-2 delivered, so a retrieval miss is an unrecoverable answer miss:
the LLM cannot cite a passage it never saw. That is why retrieval quality is measured *separately* from answer quality.

A typical production stack maps onto it one-to-one: embedding model + vector database = dense stage-1 channel, BM25 = lexical channel, RRF = merge,
reranker = stage 2. This lab rebuilds that stack at small scale with open models so every knob can be turned and measured.

## 2. Bi-encoder vs cross-encoder

**Bi-encoder** (a.k.a. dual encoder, "embedding model"): two independent encodings, then a dot product.

```
q_vec = f(question)          p_vec = f(passage)          score = cos(q_vec, p_vec)
```

* Passage vectors are computed **once, offline**, and stored in a vector index (Qdrant, FAISS...).
* At query time you encode only the question (one forward pass) and run approximate nearest-neighbour search: sub-linear in corpus size.
* The price: the model must squeeze everything a passage might answer into **one fixed-size vector** before it knows the question.

**Cross-encoder** (reranker): one network sees both texts at once.

```
score = h([CLS] question [SEP] passage)       # full self-attention between question tokens and passage tokens
```

* Question words can attend to passage words directly, so it can check "does this sentence actually answer *this* question?".
* Nothing can be precomputed: you pay one forward pass **per candidate, per query**. Scoring 1M passages per query is impossible;
  scoring 50 is fine. Hence the two-stage design.

|  | bi-encoder | cross-encoder |
|---|---|---|
| passage side | precomputed once | recomputed for every query |
| per-query cost | 1 encoder pass + ANN lookup | N encoder passes (N = candidates) |
| quality | good, limited by single-vector bottleneck | better at fine distinctions |
| used for | recall over the whole corpus | precision over a short list |

Measured on this machine (see RESULTS.md "Latency"): scoring 50 candidates with a 118M-parameter cross-encoder takes a median **94 ms** on the (shared) GPU, 7.2 s on one CPU thread and ~3.2 s on 4-8 threads, versus **16 ms** to embed the query with e5-small (and a stronger 568M reranker, bge-reranker-v2-m3, takes 560 ms on the same GPU).

## 3. Contrastive training and in-batch negatives (the math you should be able to do on a whiteboard)

Training data: pairs `(q_i, p_i)` where `p_i` answers `q_i`. Take a **batch** of `B` pairs. Encode all questions and all passages and build
the `B x B` score matrix

```
S_ij = scale * cos(q_i, p_j)            scale = 1 / temperature   (we use 20, i.e. temperature 0.05)
```

Row `i` is "question `i` against every passage in the batch". The right answer is the diagonal entry `j = i`; every other passage in the
batch is treated as a **negative for free** - these are the *in-batch negatives*. The loss is a softmax cross-entropy per row
(this is InfoNCE; sentence-transformers calls it `MultipleNegativesRankingLoss`):

```
L_i = - log( exp(S_ii) / sum_j exp(S_ij) ) = - S_ii + log sum_j exp(S_ij)
```

Intuition: "make the right passage's probability, among all passages in the batch, go to 1".

**Worked example.** Three questions, three passages, cosine similarities (scale 20). Passage 3 is a near-miss for question 1
(same topic, different fact):

```
            p1    p2    p3
q1        0.80  0.30  0.72        softmax(20*row) = [0.832, 0.000, 0.168]   L_1 = -log 0.832 = 0.184
q2        0.25  0.85  0.20                          [0.000, 1.000, 0.000]   L_2 = 0.000
q3        0.40  0.35  0.78                          [0.001, 0.000, 0.999]   L_3 = 0.001
```

The gradient of `L_i` with respect to the logit `S_ij` is simply `softmax_ij - [j == i]`. So for q1: `-0.168` on the correct passage
(pull together) and `+0.168` on p3 (push apart), but **~0.000 on p2**. The easy negative (similarity 0.30) contributes essentially nothing;
the near-miss gets *all* of the learning signal. This single fact is the whole case for hard negatives (section 4).

Practical consequences that we tested or relied on:

* **Batch size = number of negatives.** Larger `B` gives each question more negatives, hence a harder task and (usually) better embeddings.
  Memory grows with `B`, which is what **CachedMNRL / GradCache** fixes: first encode the whole batch without gradients to get the loss and
  its gradient w.r.t. the embeddings, then re-encode in small mini-batches and back-propagate the stored embedding-gradients. Same gradient,
  memory of a small batch (cost: ~2 forward passes). We tested it: a batch of 128 (CachedMNRL) trained in 5.4 GB and scored 72.3 vs 72.9 at batch 64; a batch of 512 also fit in 5.4 GB but got only 32 optimizer steps per epoch and under-trained (70.3) - more negatives per step do not help if you do fewer steps.
* **Temperature / scale.** Low temperature (large scale) sharpens the softmax so the loss concentrates on the hardest negatives - good when labels
  are clean, dangerous when false negatives exist (they get the largest push). In our runs scale 50 (temperature 0.02) vs 20 made no measurable difference (macro 72.6 vs 72.9, single seed), so we kept 20.
* **Duplicate passages in a batch are false negatives.** Datasets such as SQuAD/TyDi have several questions per paragraph. If two of them land in
  the same batch, passage `p_j` is the correct answer for `q_i` *and* is scored as its negative. We use sentence-transformers'
  `NoDuplicatesBatchSampler`, which refuses to put two copies of the same text in one batch.
* The loss only sees what it is given: language, length and style cues in the batch are shortcuts the model can use. Mixed-language batches
  make "wrong language" an easy, uninformative negative.

## 4. Hard negatives and the false-negative trap

A **hard negative** is a passage that looks relevant to the model (high similarity) but is not the answer. Because the gradient on a negative is its
softmax probability (section 3), hard negatives carry nearly all of the useful signal once the model is past the easy stage.

**Mining recipe** (what `negatives.py` does):

1. Embed every training passage with a baseline model; for each training question retrieve the top-100 passages from the training pool.
2. Remove the gold passage.
3. Remove passages we cannot trust to be negatives (below).
4. Keep the top-`n` survivors as the question's hard negatives and feed them as an extra column `(anchor, positive, negative_1...)`.

**The false-negative trap.** Datasets label *one* gold passage per question; they almost never label "all other passages that also answer
this". The passages the baseline ranks right below (or even above!) the gold are, by construction, the ones most likely to be **unlabelled positives**:
a near-duplicate paragraph, the same fact stated in the article's lead and in its infobox, an older revision of the same policy.
If you train the model to push them away, you are teaching it that relevant == irrelevant, and the damage is largest exactly where the
loss pays most attention (hardest negatives, low temperature). It is the most common way mined-negative fine-tuning backfires.

**Mitigations we implemented, and why each helps** (`NegPolicy` in `negatives.py`):

| mitigation | idea | cost |
|---|---|---|
| drop gold + exact/near duplicates of gold (token-Jaccard >= 0.8) | same text under two ids is not a negative | none |
| **score margin**: drop candidates scoring above `gold_score - margin` | "positive-aware" mining (cf. NV-Retriever): a candidate that scores as well as the gold is more likely an unlabelled positive than a true negative | removes some genuinely hard negatives |
| **rank window**: skip the top ranks (e.g. start at rank 10) | the very top of the list is where unlabelled positives concentrate | negatives get easier (measured: -1.3 macro) |
| **same-article exclusion** | sibling paragraphs of the gold's article often restate the fact | removes the hardest topical negatives too (measured: -1.5 macro) |
| use a stronger/different model to mine (here bge-m3) | a better ranker places true positives above the cut | extra compute, may be biased to its own errors |
| LLM judge on a sample (here Gemma) | *measure* the false-negative rate instead of guessing | slow; used for audit, not for mining at scale |
| in-batch: `NoDuplicates` sampler | duplicate passages in a batch | none |

What we measured: an LLM audit (Gemma, 638 judgments) found that **8.8%** of the negatives at ranks 1-3 contain the answer (1.9% at ranks 4-10, 0.6% at ranks 11-30) and **19%** (6 of 31) of the candidates that score *above* the gold. With a 0.02-cosine margin filter the mined-negative run scored +1.2 macro over the unfiltered run (73.0 vs 71.8) - and the effect was direction-dependent (ar->en +4.1, en->ar -1.3) - while a stricter margin (0.05) or dropping same-article candidates lost ~1.5 points because they throw away informative negatives. Random negatives gave nothing (70.9 vs 70.8 for in-batch only).

## 5. Metrics: Hit@k, MRR, nDCG

All our queries have exactly one labelled gold passage (so "relevant set" = 1 item), and we evaluate by ranking the whole pool for that split.
Let `rank` be the position (1 = best) of the gold passage.

* **Hit@k** (a.k.a. Success@k, Recall@k when there is one gold): `1` if `rank <= k` else `0`, averaged over queries.
  *Use when:* the downstream consumer takes the top-k as a set (RAG that stuffs the top 5 chunks into the prompt -> Hit@5 is "did the LLM get a chance?").
  Hit@1 matters for single-answer UIs. Hit@50 is the **ceiling for any reranker** that reorders the top 50.
* **MRR@k** (Mean Reciprocal Rank): `1/rank` if `rank <= k` else `0`. Rank 1 -> 1.0, rank 2 -> 0.5, rank 10 -> 0.1. *Use when:* position matters smoothly
  (you read down a list and stop at the first good hit). It is steep at the top: moving gold from rank 2 to 1 is worth +0.5, from 10 to 9 only +0.011.
* **nDCG@k**: `DCG = sum_{i<=k} rel_i / log2(i+1)` and `IDCG` = DCG of the ideal ordering; `nDCG = DCG/IDCG`. With one binary gold, `nDCG@10 = 1/log2(rank+1)`
  (rank 1 -> 1.0, rank 2 -> 0.631, rank 3 -> 0.5, rank 10 -> 0.289). *Use when:* there are several relevant passages and/or graded relevance, since it rewards
  putting *all* relevant items high. It decays more gently than MRR.

**Worked example** (three queries, gold ranks 1, 3 and "not in top 10"): Hit@1 = 1/3 = 0.333; Hit@5 = 2/3; MRR@10 = (1 + 0.333 + 0)/3 = 0.444;
nDCG@10 = (1.0 + 0.5 + 0)/3 = 0.5. The unit tests in `tests/retrieval/test_metrics.py` check these formulas on hand-computed cases (including a two-relevant case:
relevant at ranks 1 and 3 -> DCG 1.5, IDCG 1.631, nDCG 0.9197).

**How big a difference is real?** With n = 1190 queries and a hit-rate near 0.7, the standard error is sqrt(0.7*0.3/1190) = 1.3 points, so two systems
within ~2-3 points on one slice may be indistinguishable. We therefore (i) use *paired* comparisons (same queries for both systems; the bootstrap resamples queries
and keeps pairing) and (ii) run 3 training seeds for the headline comparisons and report mean +- std.

## 6. Why Arabic makes lexical search hard - and what normalization does

BM25 matches **tokens**. If the query and the passage spell the same word differently, BM25 sees two unrelated tokens. Arabic produces such variants constantly:

| phenomenon | example | what goes wrong |
|---|---|---|
| **hamza on alef** | أحمد / احمد / إحمد | writers (and phone keyboards) drop or swap the hamza; أ إ آ ا are four different code points |
| **ya vs alef maqsura** | مستشفى / مستشفي | ى and ي are interchangeable in many orthographies (Egyptian, Gulf typing) |
| **ta marbuta vs ha** | مدرسة / مدرسه | final ة often typed as ه |
| **tatweel (kashida)** | كتــاب | decorative elongation ـ inside a word; meaningless but changes the string |
| **diacritics (tashkeel)** | كَتَبَ / كتب | optional vowel marks; present in religious/educational text, absent elsewhere |
| **digits** | ٢٠١٥ / 2015 | Arabic-Indic vs Western digits |
| **clitics (morphology)** | والكتاب = و + ال + كتاب | "and the book" is one orthographic word; the article ال, conjunction و, prepositions ب/ك/ل/ف glue onto the stem. A query for كتاب misses a passage that only has والكتاب |
| **inflection** | معلم / معلمون / معلمين / معلمات | plural/dual/gender suffixes change the surface form |

`textnorm.py` implements three tokenizations, used as three BM25 systems:

* `raw`  - lowercase + split on non-letters, *keeping* diacritics/tatweel/variants (what a naive tokenizer does)
* `norm` - NFKC, digits -> 0-9, strip diacritics and tatweel, أإآ -> ا, ى -> ي, ة -> ه
* `stem` - `norm` + a **light stemmer** (strip the clitic prefixes وال بال كال فال لل ال and a few suffixes ها ان ات ون ين ية ه ي when the stem stays >= 3 letters), in the spirit of the classic light10 stemmer

**Important details that bite in production**

1. Normalization must be applied **identically at index time and query time**, otherwise you create the mismatch you were trying to remove.
2. Normalization is for the **lexical** channel. A neural embedder was trained on raw web text; feeding it normalized text moves it off-distribution (ة -> ه creates strings
   that are not real words). We measured this rather than assume it: feeding Arabic-normalized text to e5-small lowered ar->ar from 76.8 to 74.2 and ar->en from 41.4 to 34.8 (bge-m3: -1.2 and -4.3). BM25 gained: raw 67.4, normalized 71.3, normalized + light stemmer 76.1 on ar->ar.
3. Light stemming trades precision for recall (it conflates distinct words); it helps recall-oriented first stages but can hurt exact-match tasks. Heavier morphological
   analyzers (Farasa, CAMeL) exist; we deliberately kept the stemmer small and transparent.
4. Test data in this lab is clean Wikipedia text. To see the typing-noise effect we also corrupted the *queries* (random hamza drops, ة->ه, ى->ي) and re-evaluated:
   63% of the queries changed; raw BM25 lost 5.7 points, normalized or stemmed BM25 lost 0 (they undo this corruption by construction), e5-small lost 1.1, our fine-tuned e5-small 1.0 and bge-m3 0.5 - dense models are fairly robust to these variants, the raw lexical channel is not.

## 7. Why hybrid BM25 + dense (and why RRF)

* BM25 is excellent for **rare tokens**: names, IDs, numbers, product codes, exact phrases - things an embedding model smooths over.
* Dense is excellent for **paraphrase and meaning**: "when was he born" vs "date of birth", cross-lingual matches, typos.
* Their errors are partly independent, so fusing helps. **Reciprocal Rank Fusion** merges *ranks*, not scores:

```
RRF(d) = sum over rankers r of  1 / (k + rank_r(d))          k = 60 (a damping constant from the original paper)
```

  Because it uses ranks it needs no score calibration between a BM25 score (unbounded, corpus-dependent) and a cosine (bounded) - that is its big practical appeal.
  The price: it throws away *how much* better rank 1 is than rank 2, and it weights each ranker equally unless you add weights.

What happened here: RRF(e5-small, BM25-stem) raised ar->ar from 76.8 to 80.4 and RRF(bge-m3, BM25-stem) from 80.2 to 82.7; it lowered en->en (87.8 -> 86.4) and destroyed en->ar (61.7 -> 26.9). The language-gated version keeps the Arabic gain and leaves the cross-lingual numbers at the dense values. After fine-tuning the dense model the Arabic hybrid gain shrank from +3.6 to +1.5.

**Two failure modes we observed** (important for any multilingual corpus): (1) when one ranker is simply much better (English: dense >> BM25), equal-weight RRF pulls the
fused list *down* toward the weaker ranker; (2) in cross-lingual search BM25 has no shared vocabulary, so its "top hit" is noise (it matches digits or Latin-script names), but RRF
cannot tell that rank 1 of BM25 is meaningless and happily promotes it. The fix we evaluated is a **language gate**: fuse only when query and passage languages match, and use the dense
list alone for cross-lingual queries. Weighted RRF or a BM25 score floor are alternatives.

## 8. Cross-lingual retrieval and the ar -> en asymmetry

A common production complaint is that an **Arabic query against English documents** retrieves worse than the reverse. We can measure that cleanly because XQuAD is *parallel*:
the same 1190 questions and the same 240 paragraphs exist in Arabic and in English, so the only thing that changes between slices is the language of the query and/or the passage.

Four slices over the same facts (e5-small baseline, MRR@10 x100):

| query -> passage | what it tests | corpus searched | e5-small | bge-m3 |
|---|---|---|---:|---:|
| en -> en | monolingual English | 5.5k English passages | 82.7 | 80.6 |
| ar -> ar | monolingual Arabic | 4.3k Arabic passages | 77.0 | 79.4 |
| en -> ar | English question, Arabic passage | 4.3k Arabic passages | 61.7 | 67.8 |
| **ar -> en** | **Arabic question, English passage** | 5.5k English passages | **41.4** | **63.3** |

How to read it without fooling yourself:

* Compare slices that share the **same corpus** (so corpus difficulty cancels): ar->en vs en->en (both search English) and en->ar vs ar->ar (both search Arabic).
  ar->en loses 41 points relative to en->en for e5-small, while en->ar loses only 15 relative to ar->ar. So the cross-lingual gap is **asymmetric**: an Arabic question facing English
  passages is far worse off than an English question facing Arabic passages, even though the model "should" be able to match meaning across languages.
* The two corpora differ in size and in distractor style, so the raw ar->en vs en->ar gap (20 points for e5-small, 4.4 for bge-m3) over-states the pure direction effect; the same-corpus
  comparison above is the cleaner evidence.
* A bigger/better multilingual model (bge-m3, 568M parameters, trained with explicit cross-lingual objectives) shrinks the gap a lot (ar->en 63 vs en->ar 68) - the asymmetry is a property
  of the *model's alignment*, not of the task.

Likely causes (hypotheses, not proven by this lab): (1) the model saw far more English than Arabic query-passage pairs in training, so Arabic queries map to a region of the space that is
better aligned with Arabic passages than with English ones (a "language cluster" effect); (2) Arabic morphology fragments words into more sentencepiece pieces, giving noisier query embeddings;
(3) short queries carry little context to disambiguate across languages.

**What fixed it here.** Adding machine-translated cross-lingual training pairs (an English question translated to Arabic, paired with the original English passage) moved ar->en from
49.4 to 54.7 MRR@10 (and en->ar from 74.1 to 80.4) with the *same* training recipe - about twice what mining hard negatives added for that direction (+2.7). See RESULTS.md for the in-batch-only control (+6.6 / +9.0).
The practical lesson: if a failure mode is a *distribution gap* (a language pair your training data never exercised), the cheapest fix is data that exercises it, and hard negatives are secondary.

## 9. Reranking: what the cross-encoder buys, and what limits it

* A reranker can only re-order what the first stage returned, so **candidate recall (Hit@50) is its ceiling**. In this lab the baseline first stage has Hit@50 of 83.1% for ar->en, 94.1% for en->ar, 95.3% for ar->ar and 99.4% for en->en (ar->en rises to 91.3% after fine-tuning the embedder).
* Training the reranker: each training query contributes (query, gold) with label 1 and (query, mined negative) pairs with label 0 (binary cross-entropy, positives up-weighted by the
  negative:positive ratio). The negatives must come from the **first-stage distribution** - the reranker will only ever see what stage 1 returns - so we mine them with the same baseline
  e5-small retriever, with the same margin filter against false negatives. We compare against random negatives to measure what hard-negative mining is worth for a cross-encoder.
* Results and latency are in RESULTS.md. A cross-encoder re-ordering the baseline's top-50 lifted macro MRR@10 from 66.9 to 80.5 off the shelf and to 82.7 after fine-tuning; ar->en went from 41.4 to 68.9, the biggest single lever we measured. Mined negatives matter even more for rerankers than for embedders: random negatives gave 80.1 and wrecked English (en->en 85.9 vs 91.0).
* Latency math you should be able to do: reranking cost is `N_candidates x cost(one pair forward)` (batched, so sub-linear in practice on a GPU); going from top-20 to top-50 to top-100 candidates
  trades recall for latency. Our measured depth curve: reranking the top-10 / 20 / 50 gives macro MRR@10 78.4 / 81.3 / 82.7 (ar->en 58.0 / 64.7 / 68.9). The curve tracks first-stage recall (ar->en Hit@10 = 62.9, Hit@50 = 83.1), so depth buys back exactly the candidates the first stage ranks low; 20 -> 50 costs 2.4x the GPU time (39 -> 94 ms) for +1.4 macro.

## 10. How this maps to a production RAG stack (embeddings + vector database + BM25 + RRF + reranker)

| Production component | What this lab says about it |
|---|---|
| **Off-the-shelf or hosted embeddings (dense channel)** | Treat it like our `bge-m3`/`e5` baselines: strong, but check it **per direction** - an off-the-shelf model can be much worse on Arabic-query -> English-doc than on the reverse (e5-small: 41 vs 62; bge-m3: 63 vs 68). Measure it on your own queries before tuning anything. |
| **Vector database** | Stores the passage vectors; nothing here changes its role. Two implications: (1) fine-tuning the embedder means **re-embedding the whole corpus** (a model change is a data migration; version your collections), (2) score thresholds tuned on one model do not transfer to another. |
| **BM25 channel** | Index and query with the **same Arabic normalization** (and light stemming). That alone moved Arabic BM25 from 67.4 to 76.1 MRR@10 here, and it protects against typing noise. Do not feed normalized text to the dense model without testing. |
| **RRF merge** | Robust and calibration-free, but equal-weight RRF can hurt where one channel dominates and **fusing a lexical channel into a cross-lingual query adds noise**. Gate BM25 by language match (or weight it). |
| **Reranker** | The biggest single lever we measured for ar->en and en->ar, because a joint model compares the actual words. Budget the latency (RESULTS.md) and rerank the top ~50. |

What to do next in production (in order of value per effort):

1. **Build a labelled eval set from real traffic** (even 200-300 real queries with an accepted source passage) and report Hit@k / MRR **per language direction**. Everything above is only a template for this.
2. **Fix the lexical channel first** (normalization, stemming, language-gated fusion): cheap, no GPU, no re-embedding.
3. **Add or tune a reranker** on top-50 before touching the embedder - it fixes ranking without a re-index.
4. **Only then fine-tune the embedder**, using *your* documents: LLM-generated synthetic queries per chunk (+ a few real ones), mined hard negatives with a false-negative filter, and a held-out split **by document** so you do not leak. In an enterprise corpus false negatives are worse than on Wikipedia (versioned policies, near-duplicate decks) - de-duplicate by document id/version before mining.
5. Watch cost: mining + training an e5-small-size model took minutes here; the expensive parts are the evaluation set and the re-embedding.

## 11. Ten interview questions, with model answers grounded in our numbers

(Numbers are test-set MRR@10 x100 unless stated; "macro" = mean over the four directions ar->ar, en->en, ar->en, en->ar; "+-" is the std over 3 training seeds.)

**Q1. When would you use a bi-encoder, and when a cross-encoder?**
Bi-encoder for stage 1: passages are embedded once, so a query costs one encoder pass plus an ANN lookup, and it scales to millions of passages. Cross-encoder for stage 2: it reads
query and passage together, so it is more precise but needs one forward pass per candidate, which is only affordable on a short list. In our runs the baseline e5-small bi-encoder has
Hit@1 of 58.3 (macro) and the same candidates reranked by a cross-encoder reach 77.6 (macro Hit@1; 75.2 with the off-the-shelf reranker); the price is a median 94 ms per query on the GPU for 50 candidates (vs 16 ms to embed the query), or several seconds on CPU. I would never run the cross-encoder over the full corpus, and I would
not skip it when precision at the top matters.

**Q2. Explain the contrastive loss and what in-batch negatives are.**
For a batch of B (query, positive) pairs I build the BxB cosine matrix, multiply by a scale (1/temperature, 20 here) and apply softmax cross-entropy per row with the diagonal as the label.
Every other passage in the batch is a free negative. The gradient on logit (i,j) is `softmax_ij - 1[i=j]`, so easy negatives (near-zero probability) contribute almost nothing and near-misses
dominate. Bigger batch = more negatives = a harder task; CachedMNRL/GradCache gets the large batch without the memory by recomputing activations per mini-batch. We checked it: a batch of 512 via CachedMNRL ran in 5.4 GB (plain MNRL at batch 64 with 3 negatives peaked at 15.9 GB) but scored 70.3 vs 72.9 for batch 64, because with a fixed epoch count it only gets 32 optimizer steps - in our setting the number of updates, not the number of negatives, was the binding constraint (batch 32 / 64 / 128 all scored 72.3-72.9). I also mention that duplicates in a batch are false negatives, which is why we used a no-duplicates sampler.

**Q3. Why do hard negatives help, and what is the false-negative trap?**
Because the loss gradient lives on the near-misses. With in-batch negatives alone we got a macro of 70.8 +- 0.2 (baseline 66.9); adding one mined hard negative per query gave 73.0 +- 0.2, i.e. +2.2 on
macro (+1.1 to +3.0 points depending on the direction, all significant under a paired bootstrap). Random negatives gave 70.9 - indistinguishable from in-batch only - so it is the *hardness*, not the extra column, that pays.
The trap: the passages ranked right under the gold are the most likely unlabelled positives. An LLM audit of 638 judged (query, passage) pairs found that 8.8% of the negatives at ranks 1-3 actually
contained the answer, versus 1.9% at ranks 4-10 and 0.6% at ranks 11-30, and 19% (6 of 31) when the candidate scored *above* the gold. Training against those would teach the model that relevant = irrelevant.

**Q4. How did you mitigate false negatives, and did it matter?**
Drop the gold and near-duplicates, then apply a score margin: discard candidates scoring within 0.02 cosine of the gold (cosine range for e5 is compressed, 0.7-0.9, so 0.02 is a meaningful gap).
The LLM audit says that margin removes the 11-19% false-negative zone. Result: margin 0.02 gave 73.0 +- 0.2 vs 71.8 with no filter (single seed; a 1.2-point gap is ~5 seed-stds). A stricter margin (0.05) or
dropping same-article candidates *lost* points (71.5 each), because they also delete genuinely informative hard negatives and force padding with random negatives (23% of queries ran out of mined negatives at margin 0.05). So the
filter has a sweet spot; I would tune it on a dev set and verify with an LLM audit rather than guess. A stronger miner (bge-m3) was no better: 73.1 vs 72.9 for the same seed (within noise).

**Q5. Hit@k, MRR, nDCG - which do you report for a RAG system?**
Hit@k (success@k) for "is the gold in the context window?", with k = how many chunks you actually stuff in the prompt; MRR when the position of the first good hit matters (single-answer UX); nDCG when there
are several relevant passages or graded relevance. For a reranker I also report Hit@50 of the first stage as its ceiling: ours rises from 83.1 to 91.3 for ar->en after fine-tuning, which is what lets a reranker work at all.
I always report per language direction, because the macro hides that ar->en and en->ar behave very differently.

**Q6. Fine-tuning gave +X. How do you know it is real and not noise or leakage?**
(1) Splits are by *article* and by pool: test passages and test articles never appear in training, test queries are the official TyDi dev, ARCD validation and XQuAD (derived from SQuAD-dev articles, none of
which are in train); (2) an independent audit script checks duplicate questions, titles and 6-gram passage overlap across pools - 14 of 6,823 test queries have a gold passage that overlaps a training passage and
removing them changes macro MRR by 0.01 points; (3) 3 training seeds for the headline comparisons (std 0.1-0.3 points) and a paired bootstrap over queries (e.g. final recipe vs baseline: +3.3 [2.5, 4.2] on ar->ar,
+13.7 [11.8, 15.7] on ar->en); (4) the dev set was used for model selection only (epoch count / early stopping, a check of the final recipe); defaults such as the 0.02 margin were fixed from train-pool statistics before any training run and are *reported as ablations*, not selected on test; the test set was only read for reporting. Remaining caveat: Wikipedia QA is not an enterprise-document domain.

**Q7. How much data and how many epochs do you need?**
Returns diminish fast: with hard negatives, 25% / 50% / 100% of the 16.4k queries gave macro 71.7 / 72.5 / 73.0, i.e. ~80% of the gain from a quarter of the data (4k queries, ~1 minute of training). Three
epochs did not beat one (72.4 vs 73.0; dev peaked at epoch 1 and fell 0.3-1.0 points after), so with a small, clean-ish set I would stop early and spend the effort on data quality and hard negatives instead.
Training the 25% subset for 4 epochs (same number of steps as 100% x 1 epoch) did not close the gap either (71.8), so it is data diversity, not step count.

**Q8. Why does Arabic need normalization for BM25 but not (necessarily) for the dense model?**
BM25 matches tokens, and Arabic has many spellings of one word: أ/إ/آ/ا, ى/ي, ة/ه, tatweel, diacritics, clitics (و+ال+كتاب). Normalizing both index and query collapses them: raw BM25 67.4 -> normalized 71.3 -> normalized + light
stemmer 76.1 on ar->ar, and with typing noise in the queries (hamza dropped etc.) raw BM25 fell 5.7 points while the normalized variants did not move. A dense model was trained on raw text and has learned many of these invariances; feeding it
normalized text moved it *off* distribution: e5-small dropped from 76.8 to 74.2 on ar->ar and from 41.4 to 34.8 on ar->en. Rule of thumb: normalize for the lexical channel, test before normalizing for the dense one.

**Q9. Hybrid search: when does it help and when does it hurt?**
It helped where lexical and semantic signals complement each other: ar->ar went from 76.8 (dense) to 80.4 with RRF(e5-small, BM25-stem), and for bge-m3 from 80.2 to 82.7. It hurt English->English (87.8 -> 86.4) because equal-weight RRF
drags the better ranker towards the weaker one, and it badly hurt cross-lingual queries (ar->en 41.4 -> 36.9, en->ar 61.7 -> 26.9) because BM25 has no shared vocabulary there and its rank-1 is noise that RRF cannot detect.
Fix: gate the lexical channel on language match (our gated hybrid restores 41.4 / 61.7 and keeps the Arabic gain). After fine-tuning, the Arabic hybrid gain shrank (+3.6 -> +1.5), so I would re-tune fusion weights after every embedder change.

**Q10. A product team sees an ar->en retrieval asymmetry. What would you do?**
First measure it on real queries per direction (our XQuAD slices reproduce it: e5-small ar->en 41.4 vs en->en 82.7 over the same English corpus). Then in order of cost: (1) a reranker over the top-50 - the off-the-shelf mMiniLM cross-encoder
lifted ar->en from 41.4 to 65.7 and the fine-tuned one to 68.9 (needs Hit@50 high enough: 83%); (2) a stronger embedder (bge-m3: 63.3); (3) fine-tune the embedder with cross-lingual pairs - adding Gemma-translated queries paired with the
original English passages lifted ar->en from 49.4 to 54.7 +- 0.6 and en->ar from 74.1 to 80.4 on top of hard negatives, about twice what mining hard negatives added for that direction (+2.7); (4) query translation / expansion. Caveats: machine-translated training queries are cleaner than real code-mixed Arabic, and
Wikipedia is not your corpus - validate on production queries.

---

## 12. What this lab does NOT show

* **Domain transfer.** All data is Wikipedia-style factoid QA. An enterprise corpus (analytics docs, definitions, policies, tables) is a different distribution; absolute numbers will differ and so may the ranking of methods. Nothing here tells you the size of the fine-tuning gain on your data.
* **Open-domain scale.** Test corpora have 4.3k Arabic and 5.5k English passages; production has orders of magnitude more distractors, where first-stage recall and ANN approximation matter. Absolute MRR will be lower at scale.
* **Single gold passage per query.** Real relevance is multi-positive and graded; with one label per query, correct-but-unlabelled passages count as errors, which understates every system and biases against recall-oriented models. (This is exactly the false-negative problem we studied, applied to evaluation.) MIRACL-style human-judged multi-positive sets were not used.
* **Synthetic cross-lingual training data.** The ar->en/en->ar *training* queries are Gemma translations of real questions (clean, formal MSA); the *test* cross-lingual queries are human translations (XQuAD). Real users code-mix, use dialect and transliterate; we only tested orthographic noise, not dialect.
* **XQuAD cross-lingual measures parallel content.** The English and Arabic passages say the same thing, which is the easy case; in production the English and Arabic documents are different documents.
* **One small model family.** Fine-tuning gains for e5-small do not transfer automatically to a larger or already domain-tuned model (the baseline bge-m3, 568M, beats our fine-tuned e5-small on ar->en: 63.3 vs 54.7). We did not fine-tune bge-m3, a commercial embedding API, or a hosted vector-database pipeline end-to-end.
* **English got slightly worse.** The best recipe is flat-to-slightly-negative on English XQuAD (82.7 -> 80.8): fine-tuning on a mixed corpus trades a little English quality for Arabic and cross-lingual gains; with a monolingual-English requirement you would want a mix that protects it.
* **One reranker recipe, one negative source.** The reranker was fine-tuned once (single seed) with negatives from the baseline e5-small and evaluated at depths 10/20/50 only; a different first stage changes what the reranker sees at inference (it transferred well to the bge-m3 hybrid here, but that is one observation).
* **Statistics.** 3 seeds only for headline comparisons, single seed for the other ablations (read differences < ~1 point as noise); the test set was used for reporting only, but 20+ configurations were compared on it, so the best-looking one is mildly optimistic.
* **Timing.** The GPU and CPU were shared with other jobs throughout; latencies are indicative, ratios are more trustworthy than absolutes.
* **Judge noise.** The LLM false-negative audit is itself a model with ~91% agreement with the dataset's labelled positives; its rates are estimates (probably lower bounds), from 638 judgments.
