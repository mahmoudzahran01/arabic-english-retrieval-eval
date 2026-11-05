---
license: mit  # TBD - MIT or cc-by-sa-4.0 (the user decides before upload)
language: [ar, en]
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: intfloat/multilingual-e5-base
tags: [sentence-transformers, retrieval, cross-lingual, arabic, rag, hard-negatives]
datasets: [google-research-datasets/tydiqa, hsseinmz/arcd, rajpurkar/squad]
metrics: [mrr]
---

# multilingual-e5-base, fine-tuned for Arabic-English retrieval (v2-base)  (DRAFT - not uploaded, licence TBD)

A 278M-parameter embedder for search over mixed Arabic and English text, [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) **distilled from v2-m3** (listwise KL on in-batch scores, KD only, 2 epochs; 3-seed weight soup). It handles **an Arabic query finding an English passage** well above its base and above off-the-shelf bge-m3.

Code, benchmark and full results: the `arabic-english-retrieval-eval` repository (`docs/retrieval/RESULTS_v2.md`). Companion reranker: [`mzahran1/arabic-english-reranker-mminilm`](https://huggingface.co/mzahran1/arabic-english-reranker-mminilm). Intended repository name: `mzahran1/arabic-english-e5-base-retrieval` (to be confirmed).

## Results

Test MRR@10 x 100 on the bilingual benchmark of v1 (TyDi-QA, ARCD, XQuAD; 6,823 test queries; disjoint train/dev/test articles; leakage-audited). Mean ± sample std over seeds where several are listed.

| model | params | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|---:|
| multilingual-e5-base (base) | 278M | 78.0 | 89.8 | 49.9 | 71.9 | 72.4 |
| bge-m3 | 568M | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 |
| v1 e5-small (3 seeds) | 118M | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 |
| **this model** (3-seed soup) | 278M | 82.2 | 89.7 | 66.6 | 82.2 | 80.2 |
| same recipe, the 3 individual seeds | 278M | 82.1 ± 0.1 | 89.6 ± 0.1 | 65.9 ± 0.2 | 81.9 ± 0.3 | 79.9 ± 0.1 |
| this model + the companion reranker (top-50) | 278M + 118M | 87.1 | 90.7 | 74.4 | 83.8 | 84.0 |

Paired bootstrap, per-query (2,000 resamples, point estimate [95% CI], * = CI excludes 0):

| comparison (system minus reference) | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| this model vs multilingual-e5-base | +4.1 [+3.4, +4.9]* | -0.1 [-0.8, +0.6] | +16.7 [+14.8, +18.6]* | +10.3 [+8.8, +11.9]* | +7.8 [+7.1, +8.5]* |
| this model vs bge-m3 | +2.0 [+1.2, +3.0]* | +1.9 [+1.0, +2.7]* | +3.2 [+1.4, +5.1]* | +14.5 [+12.6, +16.2]* | +5.4 [+4.7, +6.1]* |
| this model vs v1 (seed 13) | +2.0 [+1.2, +2.8]* | +2.4 [+1.5, +3.3]* | +11.5 [+9.6, +13.4]* | +1.9 [+0.3, +3.4]* | +4.4 [+3.7, +5.2]* |

### MTEB(ara, v1) (19 tasks, same pipeline for every model)

| MTEB(ara, v1) | multilingual-e5-base | this model |
|---|---:|---:|
| Mean (Task), 19 tasks | 67.89 | **68.91** |
| Mean (TaskType) | 67.28 | 67.62 |
| ar-en cross-lingual subsets | 69.1 | 74.2 |
| ar-only subsets | 66.5 | 66.3 |

Mean (Task) +1.02 vs e5-base (CI [-0.52, +2.37], 15 wins / 4 losses): wins on 15 of 19 tasks but the CI over tasks includes 0.

### Arabic-only regression guards (MTEB(ara))

| Arabic-only guard task | multilingual-e5-base | this model |
|---|---:|---:|
| MIRACLRetrievalHardNegatives.v2 | 73.0 | 70.7 |
| NamaaMrTydiReranking | 77.1 | 68.2 |
| SemRel24STS | 59.2 | 59.0 |

## Usage

The e5 prefixes are required: `query: ` for queries and `passage: ` for documents.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mzahran1/arabic-english-e5-base-retrieval")
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
- **Teacher**: v2-m3 (this release's sibling); **recipe**: listwise KL on in-batch similarities (T = 20), no cross-entropy term, 2 epochs, lr 3e-5, batch 64, bf16; the released weights are the uniform average of three seeds.

## Limitations

- Evaluated on Wikipedia-style factoid QA; enterprise documents are not measured.
- **It forgets Arabic-only reranking:** the Mr.TyDi Arabic reranking task drops from 77.1 (e5-base) to 68.2, MIRACL-HN from 73.0 to 70.7. Use the base model or bge-m3 for pure-Arabic reranking-style workloads.
- MTEB(ara) Mean (Task) +1.02 over e5-base is not statistically distinguishable at 19 tasks (CI includes 0), although it wins 15 of 19 tasks.
- e5 prefixes are required; text beyond 256 tokens was not seen in training.
- **Disclosed training-data leak:** a post-hoc exact-match check found 9 kept training rows (8 distinct TyDi-ar questions) that equal queries of the MTEB MIRACL-HN.v2 ar dev set (0.8% of its queries; 0 for Mr.TyDi reranking). Not retrained; the effect on the reported MIRACL-HN score is small and the score is below the base model's anyway.

## License

**TBD (MIT or CC BY-SA 4.0 - the user decides).** The base model is MIT; the training data keeps its own licences: TyDi-QA Apache-2.0, ARCD MIT, SQuAD v1.1 CC BY-SA 4.0 (questions machine-translated with Gemma-4-26B-A4B).
