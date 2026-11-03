### Ragas on `A_baseline` (Gemma evaluator), mean [95% bootstrap CI]

| slice | faithfulness | answer relevancy bge | answer relevancy minilm | context precision | context recall | factual correctness | factual correctness span ref | semantic similarity |
|---|---|---|---|---|---|---|---|---|
| all answerable (q=16) | 0.95 [0.88,1.00] n=16 | 0.76 [0.70,0.83] n=16 | 0.74 [0.66,0.82] n=16 | 0.91 [0.81,1.00] n=16 | 1.00 [1.00,1.00] n=16 | 0.53 [0.32,0.74] n=16 | 0.10 [0.00,0.25] n=16 | 0.78 [0.70,0.85] n=16 |
| en questions (q=8) | 0.91 [0.78,1.00] n=8 | 0.76 [0.69,0.84] n=8 | 0.72 [0.62,0.83] n=8 | 0.94 [0.81,1.00] n=8 | 1.00 [1.00,1.00] n=8 | 0.53 [0.18,0.88] n=8 | 0.12 [0.00,0.38] n=8 | 0.75 [0.63,0.86] n=8 |
| ar questions (q=8) | 1.00 [1.00,1.00] n=8 | 0.77 [0.66,0.87] n=8 | 0.76 [0.65,0.88] n=8 | 0.87 [0.75,1.00] n=8 | 1.00 [1.00,1.00] n=8 | 0.54 [0.26,0.79] n=8 | 0.08 [0.00,0.25] n=8 | 0.82 [0.72,0.90] n=8 |
| en mono (q=4) | 1.00 [1.00,1.00] n=4 | 0.77 [0.70,0.86] n=4 | 0.68 [0.53,0.84] n=4 | 1.00 [1.00,1.00] n=4 | 1.00 [1.00,1.00] n=4 | 0.75 [0.25,1.00] n=4 | 0.00 [0.00,0.00] n=4 | 0.88 [0.81,0.96] n=4 |
| en cross (ar passage) (q=4) | 0.81 [0.62,1.00] n=4 | 0.75 [0.64,0.89] n=4 | 0.76 [0.62,0.91] n=4 | 0.87 [0.62,1.00] n=4 | 1.00 [1.00,1.00] n=4 | 0.30 [0.00,0.75] n=4 | 0.25 [0.00,0.75] n=4 | 0.62 [0.51,0.72] n=4 |
| ar mono (q=4) | 1.00 [1.00,1.00] n=4 | 0.74 [0.63,0.84] n=4 | 0.76 [0.62,0.90] n=4 | 0.87 [0.62,1.00] n=4 | 1.00 [1.00,1.00] n=4 | 0.62 [0.36,0.88] n=4 | 0.17 [0.00,0.50] n=4 | 0.85 [0.74,0.93] n=4 |
| ar cross (en passage) (q=4) | 1.00 [1.00,1.00] n=4 | 0.79 [0.61,0.97] n=4 | 0.76 [0.57,0.96] n=4 | 0.87 [0.62,1.00] n=4 | 1.00 [1.00,1.00] n=4 | 0.45 [0.00,0.90] n=4 | 0.00 [0.00,0.00] n=4 | 0.78 [0.64,0.93] n=4 |
| unanswerable (abstained) (q=4) | 0.75 [0.25,1.00] n=4 | 0.00 [0.00,0.00] n=4 | 0.00 [0.00,0.00] n=4 | - | - | - | - | - |
| answerable, abstained (q=0) | - | - | - | - | - | - | - | - |
| answerable, answered (q=16) | 0.95 [0.88,1.00] n=16 | 0.76 [0.70,0.83] n=16 | 0.74 [0.66,0.82] n=16 | 0.91 [0.81,1.00] n=16 | 1.00 [1.00,1.00] n=16 | 0.53 [0.32,0.74] n=16 | 0.10 [0.00,0.25] n=16 | 0.78 [0.70,0.85] n=16 |

### Does Ragas notice a real retrieval failure? (all 26 scored rows, answerable only; includes the targeted misses)

| slice | faithfulness | answer relevancy bge | answer relevancy minilm | context precision | context recall | factual correctness | factual correctness span ref | semantic similarity |
|---|---|---|---|---|---|---|---|---|
| gold passage retrieved (q=15) | 0.95 [0.87,1.00] n=15 | 0.78 [0.71,0.84] n=15 | 0.75 [0.67,0.84] n=15 | 0.90 [0.80,1.00] n=15 | 1.00 [1.00,1.00] n=15 | 0.57 [0.35,0.77] n=15 | 0.11 [0.00,0.27] n=15 | 0.79 [0.71,0.87] n=15 |
| gold passage NOT retrieved (model refused or guessed) (q=7) | 0.71 [0.43,1.00] n=7 | 0.08 [0.00,0.25] n=7 | 0.08 [0.00,0.24] n=7 | 0.29 [0.00,0.71] n=7 | 0.29 [0.00,0.71] n=7 | 0.00 [0.00,0.00] n=7 | 0.00 [0.00,0.00] n=7 | 0.37 [0.29,0.49] n=7 |
|   of which refused (q=6) | 0.67 [0.33,1.00] n=6 | 0.00 [0.00,0.00] n=6 | 0.00 [0.00,0.00] n=6 | 0.17 [0.00,0.50] n=6 | 0.17 [0.00,0.50] n=6 | 0.00 [0.00,0.00] n=6 | 0.00 [0.00,0.00] n=6 | 0.32 [0.28,0.36] n=6 |

### Diagnostics

- Correct abstentions on unanswerable questions: ragas answer_relevancy mean = 0.00 (ragas forces relevancy to 0 for 'noncommittal' answers; a correct refusal is scored as a failure), faithfulness mean = 0.75.
- Context recall tracks the real retrieval outcome? mean context_recall when the gold passage WAS retrieved = 1.00 (n=15), when it was NOT = 0.29 (n=7).
- FactualCorrectness with the bare XQuAD span as reference: mean 0.10 vs 0.53 with the reference expanded to its sentence (n=16 answered answerable questions; the span-reference version collapses).
- Ragas FC vs simple string match: mean FC when the reference string IS in the answer = 0.59 (n=11), when it is NOT = 0.40 (n=5).
- Wrong-language answers (n=4) vs right-language (n=12): answer_relevancy(bge) 0.75 vs 0.77; answer_relevancy(minilm) 0.76 vs 0.74; faithfulness 0.81 vs 1.00.
- Metric errors / None (NaN) per metric: errors={}; none-counts={'faithfulness': 0, 'answer_relevancy_bge': 0, 'answer_relevancy_minilm': 0, 'context_precision': 0, 'context_recall': 0, 'factual_correctness': 0, 'factual_correctness_span_ref': 0, 'semantic_similarity': 0}