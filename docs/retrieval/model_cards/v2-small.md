---
license: mit  # TBD - MIT or cc-by-sa-4.0 (the user decides before upload)
language: [ar, en]
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: intfloat/multilingual-e5-small
tags: [sentence-transformers, retrieval, cross-lingual, arabic, rag, hard-negatives]
datasets: [google-research-datasets/tydiqa, hsseinmz/arcd, rajpurkar/squad]
metrics: [mrr]
---

# multilingual-e5-small, fine-tuned for Arabic-English retrieval (v2-small)  (DRAFT - not uploaded, licence TBD)

A 118M-parameter embedder for search over mixed Arabic and English text, [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) **distilled from v2-m3** (listwise KL on in-batch scores, KD only, 2 epochs; 3-seed weight soup). It is the successor of the v1 checkpoint: +5.0 on Arabic-query -> English-passage retrieval, same size; it does not reach bge-m3 on that direction.

Code, benchmark and full results: the `arabic-english-retrieval-eval` repository (`docs/retrieval/RESULTS_v2.md`). Companion reranker: [`mzahran1/arabic-english-reranker-mminilm`](https://huggingface.co/mzahran1/arabic-english-reranker-mminilm). Intended repository name: `mzahran1/arabic-english-e5-small-retrieval-v2` (to be confirmed).

## Results

Test MRR@10 x 100 on the bilingual benchmark of v1 (TyDi-QA, ARCD, XQuAD; 6,823 test queries; disjoint train/dev/test articles; leakage-audited). Mean ± sample std over seeds where several are listed.

| model | params | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|---:|
| multilingual-e5-small (base) | 118M | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 |
| bge-m3 | 568M | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 |
| v1 e5-small (3 seeds) | 118M | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 |
| **this model** (3-seed soup) | 118M | 80.3 | 88.0 | 59.7 | 79.7 | 76.9 |
| same recipe, the 3 individual seeds | 118M | 80.4 ± 0.0 | 87.9 ± 0.2 | 59.3 ± 0.3 | 79.7 ± 0.2 | 76.8 ± 0.0 |
| this model + the companion reranker (top-50) | 118M + 118M | 87.0 | 90.6 | 73.4 | 83.4 | 83.6 |

Paired bootstrap, per-query (2,000 resamples, point estimate [95% CI], * = CI excludes 0):

| comparison (system minus reference) | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| this model vs multilingual-e5-small | +3.5 [+2.7, +4.3]* | +0.2 [-0.5, +1.0] | +18.3 [+16.4, +20.3]* | +18.0 [+16.2, +19.9]* | +10.0 [+9.3, +10.8]* |
| this model vs bge-m3 | +0.2 [-0.8, +1.2] | +0.2 [-0.8, +1.2] | -3.6 [-5.6, -1.7]* | +11.9 [+9.9, +13.9]* | +2.2 [+1.4, +3.0]* |
| this model vs v1 (seed 13) | +0.2 [-0.4, +0.8] | +0.7 [+0.1, +1.4]* | +4.6 [+3.3, +5.9]* | -0.7 [-1.8, +0.4] | +1.2 [+0.7, +1.7]* |

### MTEB(ara, v1) (19 tasks, same pipeline for every model)

| MTEB(ara, v1) | multilingual-e5-small | this model |
|---|---:|---:|
| Mean (Task), 19 tasks | 65.63 | **66.20** |
| Mean (TaskType) | 65.70 | 65.42 |
| ar-en cross-lingual subsets | 63.9 | 69.7 |
| ar-only subsets | 65.0 | 64.1 |

Mean (Task) +0.57 vs e5-small (CI [-1.24, +2.19], includes 0); vs v1 +1.19 (CI [+0.43, +1.92], excludes 0).

### Arabic-only regression guards (MTEB(ara))

| Arabic-only guard task | multilingual-e5-small | this model |
|---|---:|---:|
| MIRACLRetrievalHardNegatives.v2 | 72.4 | 68.3 |
| NamaaMrTydiReranking | 79.2 | 68.9 |
| SemRel24STS | 51.0 | 46.1 |

## Usage

The e5 prefixes are required: `query: ` for queries and `passage: ` for documents.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mzahran1/arabic-english-e5-small-retrieval-v2")
query = model.encode(["query: متى تأسست جامعة الملك عبدالله للعلوم والتقنية؟"])
docs = model.encode(
    [
        "passage: The King Abdullah University of Science and Technology was founded in 2009 in Thuwal.",
        "passage: Riyadh is the capital of Saudi Arabia and its largest city.",
    ]
)
print(model.similarity(query, docs))  # the KAUST passage scores highest
```

## Training

- **Data**: the same 39,377 filtered examples as v2-m3.
- **Teacher**: v2-m3; **recipe**: listwise KL on in-batch similarities (T = 20), no cross-entropy term, 2 epochs, lr 3e-5, batch 64, bf16; the released weights are the uniform average of three seeds.

## Limitations

- Evaluated on Wikipedia-style factoid QA; enterprise documents are not measured.
- **It forgets Arabic-only skills relative to e5-small:** Mr.TyDi Arabic reranking 79.2 -> 68.9, MIRACL-HN 72.4 -> 68.3, SemRel24 STS 51.0 -> 46.1. Against v1 these are all better (65.9 / 66.3 / 41.2).
- MTEB(ara) Mean (Task) +0.57 over e5-small is within noise; its Mean (TaskType) (65.42) is below e5-small's (65.70).
- e5 prefixes are required; text beyond 256 tokens was not seen in training.
- **Disclosed training-data leak:** a post-hoc exact-match check found 9 kept training rows (8 distinct TyDi-ar questions) that equal queries of the MTEB MIRACL-HN.v2 ar dev set (0.8% of its queries; 0 for Mr.TyDi reranking). Not retrained; the effect on the reported MIRACL-HN score is small and the score is below the base model's anyway.

## License

**TBD (MIT or CC BY-SA 4.0 - the user decides).** The base model is MIT; the training data keeps its own licences: TyDi-QA Apache-2.0, ARCD MIT, SQuAD v1.1 CC BY-SA 4.0 (questions machine-translated with Gemma-4-26B-A4B).
