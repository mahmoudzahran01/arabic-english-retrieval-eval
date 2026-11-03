# Evaluating RAG and LLM judges: an explainer for people who have to defend the numbers

*Audience: you know what an embedding and an LLM call are, but you have not yet had to prove that an evaluation is trustworthy.
Everything below is tied to numbers measured in this lab (see [RESULTS.md](RESULTS.md)); section 9 lists what those numbers do **not** show.*

**The one-sentence version.** A RAG evaluation is a *measuring instrument* built out of another LLM; before you believe what it
says about your system you must measure the instrument itself, by feeding it mistakes whose answer you already know.

---

## 1. What can go wrong in a RAG system (and so what you must measure)

```
question --> [retriever] --> top-k passages --> [generator LLM + prompt] --> answer (+ citations)
                |                                       |
        did the right evidence                 is the answer (a) grounded in what it was given,
        reach the generator?                   (b) correct, (c) relevant, (d) in the right language,
        (retrieval metrics)                    (e) honest when the evidence is missing?
                                               (generation metrics)
```

Failures at the two stages look identical from outside ("the answer is wrong") but have different fixes. In this lab's baseline
the 34 refusals of the 160 answerable questions broke down as **28 because retrieval never found the gold passage and only 6
because the generator refused despite having it** - so tuning the prompt would have attacked the minority cause.
Always attribute a failure to a stage *before* fixing it.

**Retrieval metrics (need a labelled gold passage; no LLM involved, so cheap and exact):**
* **hit@k** - fraction of questions whose gold passage is in the top-k. Answers "did the evidence reach the generator at all?"
* **MRR** - mean of 1/rank of the gold passage; rewards putting it *first*. Reranking improves MRR even when hit@k barely moves.
* **Context recall / precision (the Ragas versions)** - same idea, but *judged by an LLM* against a reference answer instead of
  matched against a gold id. They exist because most real corpora have no gold ids. Cost: they inherit the judge's errors
  (RESULTS.md section 3.2 shows Ragas context recall next to the exact retrieval outcome).

**Generation metrics (need an LLM or a human, so they must be validated - section 5):**
* **Faithfulness / groundedness** - are the answer's claims supported by the *retrieved context*? It does **not** ask whether the
  claims are true in the world. An answer can be faithful and wrong (the context was wrong/irrelevant), or correct and unfaithful
  (the model used memory).
* **Correctness** - does the answer agree with a *reference answer*? Needs a reference; ignores the context.
* **Answer relevance** - does the answer address the question? (Not: is it true.)
* **Citation accuracy** - does the passage the answer *cites* actually contain the claim? This is a separate property from
  faithfulness: "right answer, wrong footnote" is faithful to the context as a whole but breaks user trust.
* **Language / format / abstention behaviour** - deterministic properties that LLM metrics tend to ignore. In this lab 27% of the
  baseline's real answers were written in the wrong language and **no Ragas metric noticed**; a 5-line script did.

## 2. What each Ragas metric actually computes (0.4.x, `ragas.metrics.collections`)

Ragas is not magic: every LLM metric is "ask the LLM to decompose, then ask it to verify, then count". Knowing the recipe tells
you where it breaks.

| metric | recipe | inputs | main failure modes (seen here unless marked) |
|---|---|---|---|
| **Faithfulness** | (1) LLM rewrites the answer into atomic statements; (2) LLM labels each statement supported / not supported by the retrieved context (NLI-style); score = supported / total | question, answer, contexts | *The decomposer is an LLM too*: it can drop or merge statements, so an appended hallucination can disappear before it is ever checked (measured with our own re-implementation: RESULTS.md section 4.5). A refusal ("not in the documents") has no statements -> NaN or a meaningless score. |
| **Response / Answer relevancy** | LLM generates N questions the answer would answer (and flags "noncommittal"); score = mean cosine(embedding(original question), embedding(generated question)); forced to 0 if noncommittal | question, answer, an **embedding model** | Needs an embedder that handles your language; a *correct refusal* is scored 0 by design; answers in the wrong language still score high if the embedder is multilingual (it measures topic, not language). Score is bounded by the embedder's cosine geometry (MiniLM scores differ from bge-m3 on the same generated questions). |
| **Context precision** (with reference) | for each retrieved chunk the LLM says "useful for the reference answer?"; score = average precision over ranks | question, reference, contexts (ordered!) | Order-sensitive (a good chunk at rank 3 scores lower than at rank 1 - correct, but surprising); depends on the LLM's opinion of "useful". |
| **Context recall** | LLM splits the *reference* into sentences, asks which are attributable to the retrieved context; score = attributed / total | question, reference, contexts | Only as good as the reference's granularity: a bare-span reference ("308") has one "sentence". |
| **Factual correctness** | LLM decomposes answer and reference into claims, NLI both ways; precision, recall or F1 of claims | answer, reference | **Collapses on short references**: in the toy test a correct answer vs the reference "308" scored 0.0, and 0.10 vs 0.53 on real answers with a sentence reference (RESULTS.md section 3). Penalises verbose-but-correct answers through recall. |
| **Answer correctness** | weighted mix (default 0.75/0.25) of a factuality score and embedding similarity of answer and reference | answer, reference, embedder | Inherits both families of failure; the weights are arbitrary. |

Rule of thumb: *a Ragas score is a claim-counting statistic over an LLM's opinions*. It is useful as a **relative** signal (did
config B improve over A on the same questions?) and dangerous as an **absolute** one ("our faithfulness is 0.87").

## 3. LLM-as-judge: the biases you must test for, not assume away

An LLM judge is a classifier whose labels you did not collect. Known biases, each with the test we ran in `bias_checks.py`:

* **Position bias** - in pairwise "which answer is better" prompts the judge prefers the first (or second) slot regardless of
  content. *Test:* judge every pair in both orders; a consistent judge picks the same answer after the swap. For two equally
  good answers the probability of picking slot A should be 50%.
* **Verbosity bias** - longer answers win. *Test:* pad a wrong answer with content-free fluff and see whether the padded wrong
  answer now beats the short correct one; pad a correct answer and see whether it beats an identical short one.
* **Self-preference** - a model rates its own style higher. *Test:* pairs of verified-correct answers from two different
  generators, judged by both models; compare each judge's win-rate for its own model's answer (confounded by style/length).
* **Language bias** - judges are better in English; Arabic items, mixed-script numbers (٣ vs 3), and cross-lingual
  evidence/answer pairs fail more. *Test:* report every metric per language and per (question language, evidence language).
* **Arabic-specific traps** (all of which bit somewhere in this lab): digit scripts and separators (`٣٠٨` vs `308`, `٬` vs `,`, `711.988` vs `711,988`);
  orthographic variants (alef forms, taa marbuta, diacritics, tatweel) that defeat exact string matching; names that are *transliterated* rather
  than translated (`بانثرز` vs `Panthers`), so a cross-lingual reference-contains-answer check fails on correct answers; mixed-direction text; and subword
  tokenizers that spend more tokens per Arabic word, so a 128-word-piece embedder truncates Arabic paragraphs sooner. Normalise before comparing, and test
  these variants explicitly - they are cheap to plant.
* **Leniency / "yes" bias** - small models say YES to everything unless forced to quote evidence (the naive v0 prompts here).
* **Sycophancy to the answer's own claims** - a judge shown the answer first may *adopt* it. Show the reference/context first,
  ask for evidence quotes, and keep the verdict last.

## 4. Why a judge must be meta-evaluated against planted errors

> The rule used throughout this repository: **"every grader is meta-evaluated against planted errors before it may score a model."**

The reasoning is the same as for any test instrument. If you ask a new judge "is this answer faithful?" on real outputs, you get
a number but no way to know whether the number is signal or noise: you have no ground truth. So you build items where *you*
created the defect:

1. take an answer you have verified is correct (a string match against the reference plus a human-checkable citation);
2. edit exactly one thing - change a number, swap an entity, append an invented sentence, drop half of a two-part answer,
   translate it, move the citation to the wrong passage, replace it with a refusal;
3. the label of the edited item is now **true by construction**, with no annotator disagreement;
4. run the judge on the planted items and compute, per defect type, how often it flags (recall) and how often it wrongly flags
   the untouched clean twin (false-positive rate);
5. only judges that clear a pre-agreed bar are allowed to score models.

What planted errors give you that real data cannot: *exact labels, controlled difficulty, balanced classes*. What they cannot
give you: realism - the defects are synthetic and usually easier than real ones (section 9).

The same trick also catches judges that look good for the wrong reason: a "correctness" judge that scores 0.94 F1 might just be a
string-matcher (it was, almost - `string_match` is in the results table as the baseline you must beat).

## 5. Agreement statistics: why raw agreement misleads and what kappa fixes

Suppose 95 of 100 answers are good and a lazy judge says "good" to everything. Raw agreement with the truth is **95%**. Yet it
found 0 of 5 defects. **Cohen's kappa** subtracts the agreement you would get by chance given each rater's label frequencies:

    kappa = (p_o - p_e) / (1 - p_e),   p_o = observed agreement, p_e = sum over labels of (rater-1 freq x rater-2 freq)

For the lazy judge: p_o = 0.95, p_e = 0.95*1.00 + 0.05*0.00 = 0.95, so kappa = 0 (this exact example is a unit test in
`tests/rageval/test_stats.py`). Hand example from the same file: truth 25 good / 25 bad, judge says "good" to 20 of the good and 10 of the bad:
p_o = 35/50 = 0.70, p_e = 0.5*0.6 + 0.5*0.4 = 0.50, kappa = 0.40.

Reading kappa: < 0 worse than chance, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 near-perfect (Landis & Koch
bands; arbitrary but widely used). Two caveats: kappa depends on prevalence (the same judge has a different kappa on a 50/50 set
than on a 95/5 set - so always report precision/recall of the *defect* class too), and with n=60 it has a wide interval.

**Precision/recall for defects.** Treat "defect" as the positive class. *Recall* = of the real defects, what fraction did the
judge flag (the dangerous miss rate is 1-recall). *Precision* = of the items flagged, what fraction really had a defect (cost of
false alarms = wasted human review). Report both **per defect type**, because an overall F1 can hide a judge that is perfect on
numbers and blind to citations.

## 6. Bootstrap confidence intervals and the small-n traps

You evaluated 180 questions and system B scores 4 points higher than A. Is that real? The **bootstrap** answers it without
distributional assumptions: resample the questions with replacement, recompute the metric (or the *paired difference* B-A on the
same questions), repeat 10,000 times, and read the 2.5th/97.5th percentiles.

Traps this lab demonstrates or guards against:
* **Pair your comparisons.** A and B answer the same questions; most of the variance is question difficulty, which cancels in the
  paired difference. An unpaired comparison throws that power away. (We use a paired bootstrap and an exact McNemar test on the
  discordant pairs.)
* **Intervals shrink like 1/sqrt(n).** Going from 30 to 600 questions narrows the interval ~4.5x (unit test). With n=20
  unanswerable questions a perfect 20/20 still has a 95% lower bound of ~84%: "100% abstention" is not a claim you can make.
* **Cluster dependence.** Items that share a base answer or a passage are not independent; resample *clusters* (we cluster the
  planted items by base question) or your interval is too narrow.
* **Multiple comparisons / forking paths.** Run five configs and report the winner and the p-value is no longer 0.05. We list
  every config that was run and say which comparison was fixed in advance.
* **Minimum detectable effect.** With n=180, p~0.7 and paired correlation 0.5 you cannot reliably detect differences smaller than
  ~10 points. A "no significant difference" at that n is *absence of evidence*, not evidence of no effect.
* **Tuning on the test set.** Judge prompts were tuned on the dev split only; the numbers we headline are the held-out test split.

## 7. Online vs offline evaluation

*Offline* (this lab): fixed questions, fixed references, run before shipping; cheap, repeatable, comparable across versions, but
the question distribution is whatever you imagined. *Online*: real traffic after shipping - thumbs up/down, edits, follow-up
questions, escalation to a human, citation click-through, latency/cost, and **sampled human review of a slice of production
answers** (judged by the calibrated judge, then audited by humans). Offline tells you "B beats A on the golden set"; online tells
you whether the golden set was the right set. Mature teams loop: production failures become new golden items; the judge is
re-calibrated on the new items; drift in judge-vs-human agreement is itself a monitored metric.

The practical rule: **gate releases on offline metrics, monitor on online metrics, and refresh the offline set from online
failures every cycle.**

---

## 8. Ten interview questions, with model answers grounded in this lab's numbers

**1. "How would you evaluate a RAG system you've just inherited?"**
Split it in two and measure each half separately. Retrieval first, because it is exact and cheap: build ~150 questions with gold
passage ids, report hit@k and MRR per language and per question type (here: hit@3 = 81%, MRR 0.73; 28 of the baseline's 34
refusals were retrieval misses, not generator failures). Then generation: faithfulness to the retrieved context, correctness
against a reference, citation accuracy, language/format compliance, and abstention on questions the corpus cannot answer. Use
deterministic checks wherever the property is mechanical (language script, abstain-iff-unanswerable, citation ids exist) and an
LLM judge only for the semantic ones - and before trusting the judge, calibrate it (question 3). Finish with paired comparisons
and confidence intervals, never a single leaderboard number.

**2. "Ragas says faithfulness 0.9 - ship it?"**
No. A Ragas score is a count of an LLM's opinions about LLM-extracted claims. (a) We showed the claim extractor can drop a
hallucinated sentence: a Ragas-style extract-then-verify judge flagged only 33% of appended hallucinations, while the same
verifier on a deterministic sentence split flagged 100%. (b) With a Qwen3-4B evaluator on the planted set, Ragas faithfulness scored *correct refusals* as unfaithful 100% of
the time and flagged 17-38% of correct answers that merely wrote a number in Arabic-Indic digits (with Gemma on real answers, correct refusals averaged 0.75 and relevancy forced them to 0). (c) It is blind to citation
errors and to the answer's language: 27% of the baseline's real answers were in the wrong language and no Ragas metric noticed.
Faithfulness is a useful *relative* signal on the same questions; its absolute value is a property of the evaluator model.

**3. "How do you know your LLM judge is any good?"**
Meta-evaluate it on planted errors - this is the rule applied throughout this repository. Take verified-correct answers, edit exactly one thing (wrong
number, wrong entity, invented sentence, dropped fact, translated answer, wrong citation, false refusal), so the label is true by
construction; run the judge; report per-defect-type recall, false-positive rate on the untouched twins, and Cohen's kappa with a
bootstrap interval. Our numbers: Qwen3-4B's naive end-to-end judge had kappa 0.42 (it caught 0% of omitted facts and wrong-language
answers, 11% (2/18) of invented sentences); a rubric + decomposition + deterministic-checks pipeline reached kappa 0.97 on the held-out
test split. Then check it against humans on a sample of *real* outputs, because planted errors are easier than real ones.

**4. "Explain Cohen's kappa. Why not accuracy?"**
Accuracy rewards agreeing by luck when classes are imbalanced. If 95% of answers are good, a judge that always says "good" has 95%
accuracy and has found zero defects; kappa = (p_o - p_e)/(1 - p_e) = 0 because chance agreement p_e is already 0.95 (unit test in
`tests/rageval/test_stats.py`). Kappa still depends on prevalence, so I also report precision/recall for the defect class and per-defect-type
recall. On our test set, `string_match` had accuracy 0.98 and `overall_v0` (Qwen) had accuracy 0.68 with precision 1.00 - the
latter looks "precise" and is nearly useless (recall 0.55) - which is why recall must be reported next to precision.

**5. "Your judge has recall 1.0 on swapped numbers but 0.3 on invented sentences. What do you do?"**
Find out *why* before rewriting prompts: we traced it to the claim-extraction step (the extractor omitted the invented sentence),
not the verifier. Fix the step that failed - replace LLM claim extraction with a deterministic sentence split (recall -> 1.0) -
rather than adding more instructions. The general rule: every LLM step in a judge pipeline is its own instrument and needs its
own planted-error test; use code for steps that code can do.

**6. "What biases should you test an LLM judge for?"**
Position, verbosity, self-preference, language, leniency. We tested them with pairs judged in both orders. Qwen3-4B, given two
*equally correct* answers (one with a trailing content-free sentence), chose the first slot 92% of the time and was consistent
across the swap only 17% of the time - a pure position effect; given a correct and a wrong-number answer it picked the correct
one 98% of the time, and still 92% when the wrong one was padded with fluff. Gemma, by contrast, showed no position effect (slot A 50%, 100% order-consistent in every suite, 8 pairs each) but a *content* preference: in forced choice it chose the concise answer over the padded one 100% of the time (the opposite of verbosity bias, induced by our "prefer the clearer one" instruction) and the ASCII-digit version over the identical Arabic-Indic-digit version 100% of the time. Neither judge showed self-preference for its own style (both preferred the Qwen-written answers, 61% and 65%) - a style confound, with only 19-40 pairs. Mitigations: judge both orders and
count only consistent verdicts (or average), prefer pointwise rubric grading with a reference over pairwise preference, use a
judge from a different model family than the generator, and always have a deterministic length/format guard.

**7. "Why did a tiny 4B model become a good judge, and is that a trap?"**
Because we changed the *task*, not the model: reference-guided, rubric-explicit, one narrow question per call (is the number the
same? does the cited passage contain the claim?), evidence quotes before the verdict. The trap: the planted-error set is easy
enough to saturate (Gemma's improved judges hit kappa 1.00 on 91 items), so it cannot tell a good judge from a great one, and
an improved judge tuned against it may be tuned to its quirks. Mitigate with a held-out split (we tuned on dev, reported on test),
harder planted cases, and the human-labelled sample of real outputs.

**8. "The reranker version scores 4 points higher. Is that real?"**
Check four things. (1) *Paired* difference with a bootstrap over questions - same 180 questions in both arms. In this lab the
end-to-end "acceptable answer" rate went 59.4% -> 85.0% (+25.6 points, 95% CI [+18.3, +32.8], McNemar 51 fixed vs 5 broken),
but the pure-content correctness gain was only +6.7 points [+1.1, +12.2] (p = 0.02, which would not survive a Bonferroni correction over the
7 metrics we looked at) - borderline, and with n = 180 the smallest difference we can reliably detect is about 10 points. (2) *Which* change
caused it: reranking moved gold-hit@3 by +10.6 points and cut over-refusal from 21% to 11%; the language instruction took right-language
answers from 73% to 100% - the dominant part of the "acceptable" gain, and a one-line prompt fix, not a model-quality miracle. (3) *In-sample vs fresh*: the
fixes were found by looking at these very questions, so we re-ran A and I on 48 fresh questions (acceptable 58.3% -> 85.4%, +27.1 points [+14.6, +39.6], 13 questions fixed and none broken; content-only +6.2 [0.0, +14.6], too few questions to resolve). (4) *Multiple
comparisons*: we ran several configs offline (dense/rerank/k/chunks); the one reported was the best, so its p-value is optimistic.

**9. "How do you evaluate abstention ('I don't know') behaviour?"**
Hold out the gold passage for some questions so the corpus truly cannot answer them, and measure abstention on those *and*
over-refusal on answerable ones - one metric alone is gameable (always refuse = 100% abstention). Our baseline abstained on 20/20
unanswerable questions but also on 21% of answerable ones; 28 of those 34 were retrieval misses. The 95% lower bound for 20/20 is
~84% - a perfect score on n = 20 is not a claim of perfection. And note that generic metrics punish correct refusals: Ragas
answer-relevancy returns 0 for a noncommittal answer by design, so a correct "not in the documents" scores as a failure.

**10. "Offline vs online evaluation, and how would you set this up for an Arabic/English RAG product?"**
Offline: a versioned golden set (stratified by language, question type, answerable/unanswerable, cross-lingual), exact retrieval
metrics, deterministic checks, and a *calibrated* judge whose kappa/recall per defect type is published with the set and re-measured
whenever the judge model or prompt changes; releases gate on paired CIs, not point estimates. Online: sample production answers
daily, score with the calibrated judge, send a stratified slice (plus every judge/user disagreement) to human review, track
judge-vs-human kappa as a drift alarm, and feed confirmed failures back into the golden set and the planted-error set. Report
everything per language: here the weak judges were up to ~30 points worse in Arabic on some axes while the strong ones were not -
an aggregate would have hidden it.

---

## 9. What this does NOT show

* **That the judges are good on real errors.** All kappas are against *planted* errors, which are cleaner than real ones (one
  edit, unambiguous, string-checkable). Strong judges saturate (kappa 1.00 on 91 items). Real answers have partial support,
  paraphrased claims, ambiguous references, multi-sentence inferences. The human sheet (`human_labeling.csv`) is how you measure
  that gap - and it is yours to fill; we did not fill it.
* **That Gemma is a better judge than Qwen in general.** They differ in size, prompt-following, and decoding speed; we ran two
  models on one task family. Part of the gap is that Gemma also did *not* need the pipeline (one-shot kappa 0.95).
* **Anything about other Arabic dialects or enterprise domains.** XQuAD is translated Wikipedia in Modern Standard Arabic. Financial
  tables, scanned documents, dialect, code-switching, long documents, multi-hop questions and conversational follow-ups are not
  covered. Corpus size is 220 passages - retrieval difficulty at 100k passages is different.
* **That the improvement transfers.** Reranking and an explicit language instruction helped on this corpus and this generator;
  the size of the language effect is specific to Gemma's habit of following the passage language.
* **Statistical power for rare defect types.** Test-split per-type counts are 4-15 items; a "1.00" can hide a 30% miss rate.
  Differences between two good judges are not resolvable at this n.
* **Ragas at full scale.** Ragas on the RAG answers was run on 20 stratified questions plus 6 targeted retrieval misses (Gemma throughput), and on the planted
  set with a 4B evaluator (Qwen3-4B, 96 items; Gemma was not used there); per-language Ragas numbers are therefore wide-interval
  descriptions, not benchmarks.
* **Safety, cost, latency, and user-perceived quality** - none were measured.
* **That nothing leaked.** Judge prompts were written/inspected on the dev split, but dev and test come from the same 932-question
  pool, the same generator and the same corpus; a new corpus or generator may shift every number.
