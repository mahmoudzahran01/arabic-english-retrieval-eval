# MTEB(ara, v1) local leaderboard

19 tasks; scores x100; sorted by Mean (TaskType). eval_min = summed evaluation_time of the 19 task results (GPU shared with other jobs; precision/batch size differ for some models, see notes).

| Rank | Model | Mean (Task) | Mean (TaskType) | Retrieval | Reranking | STS | PairClassification | Classification | Clustering | BitextMining | Mean ar-only subsets | Mean ar-en cross-lingual subsets | eval_min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | BAAI/bge-m3 | 72.0 | 70.9 | 73.5 | 79.4 | 66.3 | 83.5 | 64.5 | 34.8 | 94.2 | 70.4 | 73.3 | 39.0 |
| 2 | intfloat/multilingual-e5-large | 71.4 | 70.3 | 74.0 | 79.6 | 68.1 | 79.5 | 61.8 | 35.1 | 94.1 | 69.5 | 75.3 | 45.4 |
| 3 | intfloat/multilingual-e5-base | 67.9 | 67.3 | 68.8 | 77.1 | 64.9 | 74.0 | 59.6 | 34.2 | 92.4 | 66.5 | 69.1 | 23.8 |
| 4 | intfloat/multilingual-e5-small | 65.6 | 65.7 | 66.0 | 79.2 | 60.1 | 74.1 | 58.3 | 34.7 | 87.4 | 65.0 | 63.9 | 15.3 |
| 5 | mzahran1/arabic-english-e5-small-retrieval | 65.0 | 64.4 | 66.3 | 65.9 | 57.4 | 71.8 | 59.0 | 41.8 | 88.9 | 63.1 | 68.1 | 16.1 |
| 6 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 58.7 | 62.7 | 45.2 | 70.0 | 63.1 | 76.1 | 52.0 | 39.1 | 93.1 | 54.6 | 66.5 | 12.5 |
| 7 | silma-ai/silma-embeddding-matryoshka-v0.1 | 60.3 | 61.9 | 47.1 | 71.4 | 59.2 | 86.3 | 63.2 | 24.5 | 81.5 | 62.2 | 47.9 | 21.4 |
| 8 | sentence-transformers/LaBSE | 58.2 | 58.7 | 46.2 | 51.6 | 62.5 | 69.1 | 58.3 | 28.8 | 94.4 | 54.2 | 66.2 | 20.8 |
| 9 | omarelshehy/Arabic-Retrieval-v1.0 | 58.3 | 56.8 | 55.8 | 65.2 | 62.8 | 82.0 | 64.0 | 36.0 | 31.9 | 67.4 | 27.9 | 20.8 |
| 10 | Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2 | 55.7 | 54.8 | 51.0 | 64.5 | 60.3 | 88.8 | 63.4 | 35.4 | 20.4 | 66.3 | 19.5 | 18.2 |

## Per task

| Model | MIRACLRetrievalHardNegatives.v2 | XQuADRetrieval | MLQARetrieval | BelebeleRetrieval | XPQARetrieval | WebFAQRetrieval | NamaaMrTydiReranking | STS17 | STS22.v2 | SemRel24STS | XNLI | ArEntail | AJGT.v2 | TweetEmotionClassification.v2 | TweetSarcasmClassification.v2 | HotelReviewSentimentClassification.v2 | SIB200ClusteringS2S.v2 | FloresBitextMining | Tatoeba |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BAAI/bge-m3 | 78.9 | 94.4 | 60.8 | 88.7 | 38.0 | 80.3 | 79.4 | 75.1 | 64.7 | 59.1 | 77.9 | 89.1 | 84.1 | 57.8 | 61.7 | 54.4 | 34.8 | 100.0 | 88.3 |
| intfloat/multilingual-e5-large | 76.6 | 96.2 | 63.2 | 90.9 | 37.9 | 79.2 | 79.6 | 77.9 | 63.7 | 62.7 | 72.7 | 86.2 | 81.6 | 55.9 | 57.7 | 52.1 | 35.1 | 99.8 | 88.5 |
| intfloat/multilingual-e5-base | 73.0 | 94.6 | 55.1 | 84.0 | 30.6 | 75.7 | 77.1 | 74.0 | 61.3 | 59.2 | 67.3 | 80.6 | 78.4 | 52.4 | 57.5 | 50.1 | 34.2 | 99.2 | 85.6 |
| intfloat/multilingual-e5-small | 72.4 | 93.5 | 50.7 | 80.1 | 29.9 | 69.6 | 79.2 | 68.0 | 61.4 | 51.0 | 68.9 | 79.4 | 75.4 | 50.4 | 58.6 | 48.9 | 34.7 | 98.0 | 76.8 |
| mzahran1/arabic-english-e5-small-retrieval | 66.3 | 91.4 | 54.4 | 82.6 | 31.7 | 71.4 | 65.9 | 71.1 | 59.8 | 41.2 | 66.2 | 77.5 | 78.4 | 50.0 | 61.2 | 46.4 | 41.8 | 98.7 | 79.1 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 21.9 | 77.3 | 36.8 | 72.4 | 21.9 | 40.9 | 70.0 | 80.2 | 49.1 | 60.1 | 78.6 | 73.7 | 70.5 | 39.9 | 54.0 | 43.6 | 39.1 | 98.2 | 87.9 |
| silma-ai/silma-embeddding-matryoshka-v0.1 | 35.1 | 86.8 | 30.8 | 56.9 | 21.0 | 52.0 | 71.4 | 63.5 | 60.0 | 54.0 | 81.4 | 91.3 | 87.5 | 49.7 | 66.3 | 49.5 | 24.5 | 96.8 | 66.3 |
| sentence-transformers/LaBSE | 12.1 | 82.0 | 37.1 | 71.3 | 29.4 | 45.3 | 51.6 | 71.8 | 61.0 | 54.6 | 65.5 | 72.7 | 76.5 | 46.1 | 61.7 | 49.0 | 28.8 | 100.0 | 88.8 |
| omarelshehy/Arabic-Retrieval-v1.0 | 61.6 | 89.8 | 31.2 | 52.2 | 24.1 | 76.0 | 65.2 | 57.7 | 62.6 | 68.2 | 73.9 | 90.1 | 87.7 | 53.0 | 65.3 | 50.1 | 36.0 | 42.7 | 21.0 |
| Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2 | 62.6 | 84.7 | 25.7 | 44.8 | 20.0 | 67.9 | 64.5 | 55.6 | 64.0 | 61.4 | 85.4 | 92.2 | 87.4 | 51.3 | 64.6 | 50.1 | 35.4 | 26.7 | 14.1 |
