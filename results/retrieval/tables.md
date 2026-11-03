
### Baselines on TEST - MRR@10 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 raw tokens | 67.4 | 79.5 | 3.2 | 1.2 | 37.8 |
| BM25 + Arabic normalization | 71.3 | 79.5 | 3.1 | 1.2 | 38.8 |
| BM25 + normalization + light stemmer | 76.1 | 79.5 | 3.2 | 1.2 | 40.0 |
| MiniLM-L12 paraphrase-multilingual (dense) | 40.2 | 70.1 | 35.0 | 52.8 | 49.5 |
| multilingual-e5-small (dense) | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 |
| multilingual-e5-small, Arabic-normalized input | 74.2 | 87.8 | 34.8 | 61.7 | 64.6 |
| bge-m3 (dense) | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 |
| bge-m3, Arabic-normalized input | 79.0 | 87.8 | 59.0 | 64.3 | 72.5 |
| RRF e5-small + BM25(stem) | 80.4 | 86.4 | 36.9 | 26.9 | 57.7 |
| RRF e5-small + BM25(stem), language-gated | 80.4 | 86.4 | 41.4 | 61.7 | 67.5 |
| RRF bge-m3 + BM25(raw) | 77.6 | 86.8 | 56.1 | 34.8 | 63.8 |
| RRF bge-m3 + BM25(norm) | 79.7 | 86.8 | 55.8 | 34.8 | 64.3 |
| RRF bge-m3 + BM25(stem) | 82.7 | 86.8 | 54.5 | 34.8 | 64.7 |
| RRF bge-m3 + BM25(stem), language-gated | 82.7 | 86.8 | 63.3 | 67.8 | 75.1 |


### Baselines on TEST - HIT@1 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 raw tokens | 60.0 | 72.3 | 2.4 | 0.7 | 33.8 |
| BM25 + Arabic normalization | 63.3 | 72.3 | 2.3 | 0.7 | 34.6 |
| BM25 + normalization + light stemmer | 68.3 | 72.3 | 2.4 | 0.7 | 35.9 |
| MiniLM-L12 paraphrase-multilingual (dense) | 32.7 | 62.3 | 24.9 | 43.9 | 41.0 |
| multilingual-e5-small (dense) | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 |
| multilingual-e5-small, Arabic-normalized input | 65.3 | 81.7 | 26.4 | 50.9 | 56.1 |
| bge-m3 (dense) | 73.1 | 82.1 | 52.4 | 59.2 | 66.7 |
| bge-m3, Arabic-normalized input | 71.7 | 82.1 | 47.8 | 55.4 | 64.3 |
| RRF e5-small + BM25(stem) | 72.2 | 80.5 | 25.0 | 10.6 | 47.1 |
| RRF e5-small + BM25(stem), language-gated | 72.2 | 80.5 | 31.7 | 51.4 | 59.0 |
| RRF bge-m3 + BM25(raw) | 69.4 | 81.0 | 41.4 | 16.2 | 52.0 |
| RRF bge-m3 + BM25(norm) | 71.9 | 81.0 | 41.1 | 16.2 | 52.6 |
| RRF bge-m3 + BM25(stem) | 75.4 | 81.0 | 38.8 | 16.2 | 52.9 |
| RRF bge-m3 + BM25(stem), language-gated | 75.4 | 81.0 | 52.4 | 59.2 | 67.0 |


### Baselines on TEST - HIT@10 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 raw tokens | 82.2 | 91.8 | 5.3 | 2.9 | 45.5 |
| BM25 + Arabic normalization | 86.1 | 91.8 | 5.3 | 2.9 | 46.5 |
| BM25 + normalization + light stemmer | 90.2 | 91.8 | 5.5 | 2.9 | 47.6 |
| MiniLM-L12 paraphrase-multilingual (dense) | 55.8 | 86.6 | 59.1 | 70.9 | 68.1 |
| multilingual-e5-small (dense) | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 |
| multilingual-e5-small, Arabic-normalized input | 89.3 | 97.5 | 53.8 | 82.5 | 80.8 |
| bge-m3 (dense) | 91.9 | 97.0 | 84.7 | 84.4 | 89.5 |
| bge-m3, Arabic-normalized input | 90.7 | 97.0 | 81.8 | 81.3 | 87.7 |
| RRF e5-small + BM25(stem) | 93.5 | 96.3 | 63.1 | 73.3 | 81.5 |
| RRF e5-small + BM25(stem), language-gated | 93.5 | 96.3 | 62.9 | 82.7 | 83.8 |
| RRF bge-m3 + BM25(raw) | 91.8 | 96.2 | 83.6 | 78.5 | 87.5 |
| RRF bge-m3 + BM25(norm) | 92.8 | 96.2 | 83.6 | 78.5 | 87.8 |
| RRF bge-m3 + BM25(stem) | 94.3 | 96.2 | 83.6 | 78.5 | 88.2 |
| RRF bge-m3 + BM25(stem), language-gated | 94.3 | 96.2 | 84.7 | 84.4 | 89.9 |


### Baselines on TEST - NDCG@10 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 raw tokens | 71.0 | 82.5 | 3.7 | 1.6 | 39.7 |
| BM25 + Arabic normalization | 74.9 | 82.5 | 3.6 | 1.6 | 40.7 |
| BM25 + normalization + light stemmer | 79.5 | 82.5 | 3.8 | 1.6 | 41.8 |
| MiniLM-L12 paraphrase-multilingual (dense) | 44.0 | 74.1 | 40.8 | 57.1 | 54.0 |
| multilingual-e5-small (dense) | 80.2 | 90.2 | 46.5 | 66.8 | 70.9 |
| multilingual-e5-small, Arabic-normalized input | 77.9 | 90.2 | 39.3 | 66.8 | 68.6 |
| bge-m3 (dense) | 83.0 | 90.1 | 68.5 | 71.8 | 78.4 |
| bge-m3, Arabic-normalized input | 81.9 | 90.1 | 64.6 | 68.4 | 76.2 |
| RRF e5-small + BM25(stem) | 83.7 | 88.9 | 43.2 | 37.8 | 63.4 |
| RRF e5-small + BM25(stem), language-gated | 83.7 | 88.9 | 46.5 | 66.8 | 71.5 |
| RRF bge-m3 + BM25(raw) | 81.1 | 89.1 | 62.8 | 45.3 | 69.5 |
| RRF bge-m3 + BM25(norm) | 82.9 | 89.1 | 62.6 | 45.3 | 70.0 |
| RRF bge-m3 + BM25(stem) | 85.6 | 89.1 | 61.6 | 45.3 | 70.4 |
| RRF bge-m3 + BM25(stem), language-gated | 85.6 | 89.1 | 68.5 | 71.8 | 78.7 |


### Baselines on TEST - every slice, every metric (x100)

| System | Slice | n | hit@1 | hit@5 | hit@10 | mrr@10 | ndcg@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 raw tokens | arcd_ar | 702 | 41.2 | 65.2 | 73.1 | 51.5 | 56.7 |
| BM25 raw tokens | tydi_ar | 921 | 64.8 | 79.3 | 83.0 | 71.0 | 73.9 |
| BM25 raw tokens | tydi_en | 440 | 69.3 | 87.3 | 89.1 | 76.7 | 79.8 |
| BM25 raw tokens | xquad_ar2ar | 1190 | 74.1 | 86.8 | 90.4 | 79.8 | 82.4 |
| BM25 raw tokens | xquad_ar2en | 1190 | 2.4 | 4.4 | 5.3 | 3.2 | 3.7 |
| BM25 raw tokens | xquad_en2ar | 1190 | 0.7 | 1.6 | 2.9 | 1.2 | 1.6 |
| BM25 raw tokens | xquad_en2en | 1190 | 75.2 | 91.8 | 94.5 | 82.3 | 85.3 |
| BM25 + Arabic normalization | arcd_ar | 702 | 49.1 | 76.9 | 82.2 | 61.2 | 66.3 |
| BM25 + Arabic normalization | tydi_ar | 921 | 66.3 | 82.3 | 85.8 | 72.9 | 76.0 |
| BM25 + Arabic normalization | tydi_en | 440 | 69.3 | 87.3 | 89.1 | 76.7 | 79.8 |
| BM25 + Arabic normalization | xquad_ar2ar | 1190 | 74.3 | 87.0 | 90.3 | 79.9 | 82.4 |
| BM25 + Arabic normalization | xquad_ar2en | 1190 | 2.3 | 4.4 | 5.3 | 3.1 | 3.6 |
| BM25 + Arabic normalization | xquad_en2ar | 1190 | 0.7 | 1.6 | 2.9 | 1.2 | 1.6 |
| BM25 + Arabic normalization | xquad_en2en | 1190 | 75.2 | 91.8 | 94.5 | 82.3 | 85.3 |
| BM25 + normalization + light stemmer | arcd_ar | 702 | 52.7 | 80.8 | 87.5 | 65.1 | 70.6 |
| BM25 + normalization + light stemmer | tydi_ar | 921 | 71.7 | 86.0 | 89.5 | 77.8 | 80.6 |
| BM25 + normalization + light stemmer | tydi_en | 440 | 69.3 | 87.3 | 89.1 | 76.7 | 79.8 |
| BM25 + normalization + light stemmer | xquad_ar2ar | 1190 | 80.4 | 91.7 | 93.7 | 85.3 | 87.4 |
| BM25 + normalization + light stemmer | xquad_ar2en | 1190 | 2.4 | 4.6 | 5.5 | 3.2 | 3.8 |
| BM25 + normalization + light stemmer | xquad_en2ar | 1190 | 0.7 | 1.6 | 2.9 | 1.2 | 1.6 |
| BM25 + normalization + light stemmer | xquad_en2en | 1190 | 75.2 | 91.8 | 94.5 | 82.3 | 85.3 |
| MiniLM-L12 paraphrase-multilingual (dense) | arcd_ar | 702 | 20.8 | 38.9 | 43.3 | 28.3 | 31.9 |
| MiniLM-L12 paraphrase-multilingual (dense) | tydi_ar | 921 | 36.2 | 51.5 | 57.7 | 42.8 | 46.4 |
| MiniLM-L12 paraphrase-multilingual (dense) | tydi_en | 440 | 76.8 | 89.8 | 92.5 | 82.3 | 84.8 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_ar2ar | 1190 | 41.1 | 60.6 | 66.4 | 49.5 | 53.6 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_ar2en | 1190 | 24.9 | 48.2 | 59.1 | 35.0 | 40.8 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_en2ar | 1190 | 43.9 | 65.1 | 70.9 | 52.8 | 57.1 |
| MiniLM-L12 paraphrase-multilingual (dense) | xquad_en2en | 1190 | 47.8 | 72.4 | 80.7 | 58.0 | 63.4 |
| multilingual-e5-small (dense) | arcd_ar | 702 | 50.3 | 79.3 | 84.0 | 63.4 | 68.5 |
| multilingual-e5-small (dense) | tydi_ar | 921 | 86.3 | 95.1 | 97.0 | 90.2 | 91.8 |
| multilingual-e5-small (dense) | tydi_en | 440 | 88.6 | 98.0 | 98.9 | 92.9 | 94.4 |
| multilingual-e5-small (dense) | xquad_ar2ar | 1190 | 69.2 | 86.9 | 91.1 | 76.9 | 80.4 |
| multilingual-e5-small (dense) | xquad_ar2en | 1190 | 31.7 | 54.0 | 62.9 | 41.4 | 46.5 |
| multilingual-e5-small (dense) | xquad_en2ar | 1190 | 51.4 | 75.5 | 82.7 | 61.7 | 66.8 |
| multilingual-e5-small (dense) | xquad_en2en | 1190 | 74.7 | 93.3 | 96.2 | 82.7 | 86.0 |
| multilingual-e5-small, Arabic-normalized input | arcd_ar | 702 | 48.9 | 80.1 | 84.8 | 62.5 | 68.0 |
| multilingual-e5-small, Arabic-normalized input | tydi_ar | 921 | 83.6 | 93.9 | 95.8 | 88.2 | 90.0 |
| multilingual-e5-small, Arabic-normalized input | tydi_en | 440 | 88.6 | 98.0 | 98.9 | 92.9 | 94.4 |
| multilingual-e5-small, Arabic-normalized input | xquad_ar2ar | 1190 | 63.4 | 82.9 | 87.2 | 71.9 | 75.6 |
| multilingual-e5-small, Arabic-normalized input | xquad_ar2en | 1190 | 26.4 | 46.6 | 53.8 | 34.8 | 39.3 |
| multilingual-e5-small, Arabic-normalized input | xquad_en2ar | 1190 | 50.9 | 76.1 | 82.5 | 61.7 | 66.8 |
| multilingual-e5-small, Arabic-normalized input | xquad_en2en | 1190 | 74.7 | 93.3 | 96.2 | 82.7 | 86.0 |
| bge-m3 (dense) | arcd_ar | 702 | 56.8 | 82.6 | 86.9 | 68.5 | 73.1 |
| bge-m3 (dense) | tydi_ar | 921 | 89.9 | 95.8 | 96.9 | 92.5 | 93.6 |
| bge-m3 (dense) | tydi_en | 440 | 91.6 | 99.1 | 99.5 | 95.0 | 96.2 |
| bge-m3 (dense) | xquad_ar2ar | 1190 | 72.6 | 88.1 | 91.9 | 79.4 | 82.5 |
| bge-m3 (dense) | xquad_ar2en | 1190 | 52.4 | 77.8 | 84.7 | 63.3 | 68.5 |
| bge-m3 (dense) | xquad_en2ar | 1190 | 59.2 | 79.1 | 84.4 | 67.8 | 71.8 |
| bge-m3 (dense) | xquad_en2en | 1190 | 72.7 | 91.1 | 94.5 | 80.6 | 84.0 |
| bge-m3, Arabic-normalized input | arcd_ar | 702 | 55.4 | 83.5 | 86.9 | 68.1 | 72.8 |
| bge-m3, Arabic-normalized input | tydi_ar | 921 | 89.5 | 95.3 | 96.3 | 92.1 | 93.1 |
| bge-m3, Arabic-normalized input | tydi_en | 440 | 91.6 | 99.1 | 99.5 | 95.0 | 96.2 |
| bge-m3, Arabic-normalized input | xquad_ar2ar | 1190 | 70.3 | 85.7 | 89.0 | 76.8 | 79.7 |
| bge-m3, Arabic-normalized input | xquad_ar2en | 1190 | 47.8 | 74.8 | 81.8 | 59.0 | 64.6 |
| bge-m3, Arabic-normalized input | xquad_en2ar | 1190 | 55.4 | 76.1 | 81.3 | 64.3 | 68.4 |
| bge-m3, Arabic-normalized input | xquad_en2en | 1190 | 72.6 | 91.1 | 94.5 | 80.6 | 84.0 |
| RRF e5-small + BM25(stem) | arcd_ar | 702 | 55.7 | 86.8 | 90.2 | 69.5 | 74.7 |
| RRF e5-small + BM25(stem) | tydi_ar | 921 | 80.5 | 92.1 | 94.6 | 85.7 | 87.9 |
| RRF e5-small + BM25(stem) | tydi_en | 440 | 80.5 | 92.5 | 95.5 | 85.9 | 88.3 |
| RRF e5-small + BM25(stem) | xquad_ar2ar | 1190 | 80.4 | 93.4 | 95.8 | 86.1 | 88.5 |
| RRF e5-small + BM25(stem) | xquad_ar2en | 1190 | 25.0 | 52.3 | 63.1 | 36.9 | 43.2 |
| RRF e5-small + BM25(stem) | xquad_en2ar | 1190 | 10.6 | 52.0 | 73.3 | 26.9 | 37.8 |
| RRF e5-small + BM25(stem) | xquad_en2en | 1190 | 80.6 | 95.2 | 97.1 | 86.9 | 89.5 |
| RRF e5-small + BM25(stem), language-gated | arcd_ar | 702 | 55.7 | 86.8 | 90.2 | 69.5 | 74.7 |
| RRF e5-small + BM25(stem), language-gated | tydi_ar | 921 | 80.5 | 92.1 | 94.6 | 85.7 | 87.9 |
| RRF e5-small + BM25(stem), language-gated | tydi_en | 440 | 80.5 | 92.5 | 95.5 | 85.9 | 88.3 |
| RRF e5-small + BM25(stem), language-gated | xquad_ar2ar | 1190 | 80.4 | 93.4 | 95.8 | 86.1 | 88.5 |
| RRF e5-small + BM25(stem), language-gated | xquad_ar2en | 1190 | 31.7 | 54.0 | 62.9 | 41.4 | 46.5 |
| RRF e5-small + BM25(stem), language-gated | xquad_en2ar | 1190 | 51.4 | 75.5 | 82.7 | 61.7 | 66.8 |
| RRF e5-small + BM25(stem), language-gated | xquad_en2en | 1190 | 80.6 | 95.2 | 97.1 | 86.9 | 89.5 |
| RRF bge-m3 + BM25(raw) | arcd_ar | 702 | 52.8 | 82.9 | 87.7 | 65.8 | 71.2 |
| RRF bge-m3 + BM25(raw) | tydi_ar | 921 | 76.0 | 89.1 | 92.6 | 81.8 | 84.4 |
| RRF bge-m3 + BM25(raw) | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(raw) | xquad_ar2ar | 1190 | 79.5 | 92.3 | 95.0 | 85.2 | 87.6 |
| RRF bge-m3 + BM25(raw) | xquad_ar2en | 1190 | 41.4 | 75.7 | 83.6 | 56.1 | 62.8 |
| RRF bge-m3 + BM25(raw) | xquad_en2ar | 1190 | 16.2 | 63.3 | 78.5 | 34.8 | 45.3 |
| RRF bge-m3 + BM25(raw) | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |
| RRF bge-m3 + BM25(norm) | arcd_ar | 702 | 58.4 | 85.9 | 89.9 | 70.6 | 75.3 |
| RRF bge-m3 + BM25(norm) | tydi_ar | 921 | 77.9 | 90.6 | 93.3 | 83.4 | 85.8 |
| RRF bge-m3 + BM25(norm) | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(norm) | xquad_ar2ar | 1190 | 79.6 | 92.2 | 95.1 | 85.2 | 87.6 |
| RRF bge-m3 + BM25(norm) | xquad_ar2en | 1190 | 41.1 | 75.7 | 83.6 | 55.8 | 62.6 |
| RRF bge-m3 + BM25(norm) | xquad_en2ar | 1190 | 16.2 | 63.3 | 78.5 | 34.8 | 45.3 |
| RRF bge-m3 + BM25(norm) | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |
| RRF bge-m3 + BM25(stem) | arcd_ar | 702 | 62.0 | 88.7 | 91.7 | 73.8 | 78.3 |
| RRF bge-m3 + BM25(stem) | tydi_ar | 921 | 81.8 | 92.7 | 95.3 | 86.7 | 88.8 |
| RRF bge-m3 + BM25(stem) | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(stem) | xquad_ar2ar | 1190 | 82.4 | 93.6 | 95.8 | 87.5 | 89.5 |
| RRF bge-m3 + BM25(stem) | xquad_ar2en | 1190 | 38.8 | 75.6 | 83.6 | 54.5 | 61.6 |
| RRF bge-m3 + BM25(stem) | xquad_en2ar | 1190 | 16.2 | 63.4 | 78.5 | 34.8 | 45.3 |
| RRF bge-m3 + BM25(stem) | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |
| RRF bge-m3 + BM25(stem), language-gated | arcd_ar | 702 | 62.0 | 88.7 | 91.7 | 73.8 | 78.3 |
| RRF bge-m3 + BM25(stem), language-gated | tydi_ar | 921 | 81.8 | 92.7 | 95.3 | 86.7 | 88.8 |
| RRF bge-m3 + BM25(stem), language-gated | tydi_en | 440 | 83.2 | 92.5 | 95.9 | 87.7 | 89.7 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_ar2ar | 1190 | 82.4 | 93.6 | 95.8 | 87.5 | 89.5 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_ar2en | 1190 | 52.4 | 77.8 | 84.7 | 63.3 | 68.5 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_en2ar | 1190 | 59.2 | 79.1 | 84.4 | 67.8 | 71.8 |
| RRF bge-m3 + BM25(stem), language-gated | xquad_en2en | 1190 | 78.9 | 94.3 | 96.6 | 85.9 | 88.5 |


### HEADLINE - TEST MRR@10 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 (normalized + light stemmer) | 76.1 | 79.5 | 3.2 | 1.2 | 40.0 |
| multilingual-e5-small, off the shelf (118M) | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 |
| bge-m3 dense, off the shelf (568M) | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 |
| e5-small + BM25(stem), language-gated RRF | 80.4 | 86.4 | 41.4 | 61.7 | 67.5 |
| **fine-tuned e5-small** (hard negatives + cross-lingual pairs; 3 seeds) | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 |
| baseline e5-small -> top-50 -> off-the-shelf mMiniLM reranker | 85.3 | 90.9 | 65.7 | 80.0 | 80.5 |
| baseline e5-small -> top-50 -> **fine-tuned reranker** | 87.2 | 91.0 | 68.9 | 83.8 | 82.7 |
| **fine-tuned e5-small -> top-50 -> fine-tuned reranker** | 86.7 | 90.6 | 72.3 | 83.7 | 83.3 |
| gated RRF(bge-m3, BM25) -> top-50 -> fine-tuned reranker | 89.0 | 91.3 | 75.3 | 84.4 | 85.0 |


### HEADLINE_H1 - TEST Hit@1 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| BM25 (normalized + light stemmer) | 68.3 | 72.3 | 2.4 | 0.7 | 35.9 |
| multilingual-e5-small, off the shelf (118M) | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 |
| bge-m3 dense, off the shelf (568M) | 73.1 | 82.1 | 52.4 | 59.2 | 66.7 |
| e5-small + BM25(stem), language-gated RRF | 72.2 | 80.5 | 31.7 | 51.4 | 59.0 |
| **fine-tuned e5-small** (hard negatives + cross-lingual pairs; 3 seeds) | 72.8 ± 0.2 | 81.4 ± 0.3 | 42.8 ± 0.8 | 72.7 ± 0.5 | 67.4 ± 0.2 |
| baseline e5-small -> top-50 -> off-the-shelf mMiniLM reranker | 80.3 | 86.9 | 58.7 | 74.9 | 75.2 |
| baseline e5-small -> top-50 -> **fine-tuned reranker** | 82.8 | 86.8 | 62.1 | 78.8 | 77.6 |
| **fine-tuned e5-small -> top-50 -> fine-tuned reranker** | 82.3 | 86.4 | 64.2 | 77.7 | 77.7 |
| gated RRF(bge-m3, BM25) -> top-50 -> fine-tuned reranker | 84.3 | 87.0 | 67.6 | 79.3 | 79.6 |


### EXP A_negatives - TEST MRR@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 | 1 |
| fine-tuned, in-batch negatives only | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| + 1 RANDOM negative per query | 78.6 | 86.8 | 46.6 | 71.7 | 70.9 | 1 |
| + 1 mined hard negative, NO false-negative filter | 79.9 | 86.7 | 45.1 | 75.4 | 71.8 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| + 1 mined hard negative, margin 0.05 | 79.4 | 86.8 | 47.5 | 72.4 | 71.5 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 79.3 | 87.2 | 47.3 | 72.5 | 71.6 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 79.7 | 86.8 | 46.6 | 72.8 | 71.5 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 81.2 | 87.9 | 48.2 | 75.1 | 73.1 | 1 |
| + 3 mined hard negatives, margin 0.02 | 80.2 | 87.6 | 49.7 | 73.8 | 72.8 | 1 |


### EXP A_negatives - TEST Hit@1

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 | 1 |
| fine-tuned, in-batch negatives only | 70.7 ± 0.3 | 79.6 ± 0.3 | 35.6 ± 1.2 | 61.2 ± 1.3 | 61.8 ± 0.2 | 3 |
| + 1 RANDOM negative per query | 70.2 | 80.5 | 35.7 | 61.8 | 62.0 | 1 |
| + 1 mined hard negative, NO false-negative filter | 72.7 | 80.4 | 34.8 | 66.2 | 63.5 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| + 1 mined hard negative, margin 0.05 | 71.2 | 80.4 | 36.2 | 62.8 | 62.6 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 71.4 | 81.1 | 36.1 | 62.9 | 62.9 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 71.7 | 80.5 | 35.7 | 63.4 | 62.8 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 74.2 | 82.1 | 37.2 | 66.4 | 65.0 | 1 |
| + 3 mined hard negatives, margin 0.02 | 72.3 | 81.6 | 38.9 | 64.4 | 64.3 | 1 |


### EXP A_negatives - TEST Hit@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 | 1 |
| fine-tuned, in-batch negatives only | 92.2 ± 0.4 | 96.9 ± 0.1 | 71.0 ± 1.0 | 89.7 ± 0.8 | 87.4 ± 0.1 | 3 |
| + 1 RANDOM negative per query | 92.0 | 96.7 | 70.5 | 90.1 | 87.3 | 1 |
| + 1 mined hard negative, NO false-negative filter | 91.9 | 96.6 | 69.2 | 91.4 | 87.3 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| + 1 mined hard negative, margin 0.05 | 92.5 | 96.9 | 72.4 | 90.3 | 88.0 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 92.2 | 97.0 | 71.3 | 90.5 | 87.8 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 92.6 | 96.9 | 70.2 | 90.3 | 87.5 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 92.8 | 97.1 | 72.4 | 91.2 | 88.4 | 1 |
| + 3 mined hard negatives, margin 0.02 | 92.8 | 97.0 | 72.9 | 91.1 | 88.5 | 1 |


### EXP A_negatives - DEV MRR@10

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 72.6 | 87.3 | 55.5 | 67.7 | 79.9 | 1 |
| fine-tuned, in-batch negatives only | 76.9 ± 0.5 | 86.5 ± 0.1 | n/a | n/a | 81.7 ± 0.2 | 3 |
| + 1 RANDOM negative per query | 76.9 | 87.0 | 58.0 | 73.9 | 81.9 | 1 |
| + 1 mined hard negative, NO false-negative filter | 76.7 | 87.6 | 59.2 | 72.1 | 82.1 | 1 |
| + 1 mined hard negative, margin 0.02 (default) | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| + 1 mined hard negative, margin 0.05 | 78.5 | 86.8 | 58.7 | 74.6 | 82.6 | 1 |
| + 1 mined hard negative, margin 0.02, skip ranks 1-9 | 78.0 | 86.9 | 59.5 | 73.8 | 82.4 | 1 |
| + 1 mined hard negative, margin 0.02, drop same-article | 77.4 | 86.9 | 59.3 | 73.8 | 82.1 | 1 |
| + 1 mined hard negative (mined by bge-m3), margin 0.02 | 78.4 | 88.2 | 61.1 | 74.4 | 83.3 | 1 |
| + 3 mined hard negatives, margin 0.02 | 79.3 | 87.3 | n/a | n/a | 83.3 | 1 |


### EXP B_datasize - TEST MRR@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 25% of queries | 77.8 | 85.0 | 45.7 | 72.4 | 70.2 | 1 |
| in-batch, 50% | 78.5 | 85.8 | 46.2 | 71.8 | 70.6 | 1 |
| in-batch, 100% | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| in-batch, 25% x 4 epochs (same #steps as 100% x 1) | 77.4 | 85.8 | 47.2 | 71.4 | 70.5 | 1 |
| hard neg, 25% of queries | 79.0 | 86.5 | 48.4 | 73.1 | 71.7 | 1 |
| hard neg, 50% | 80.1 | 87.0 | 49.5 | 73.5 | 72.5 | 1 |
| hard neg, 100% | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| hard neg, 25% x 4 epochs (same #steps as 100% x 1) | 79.1 | 86.5 | 48.9 | 72.6 | 71.8 | 1 |


### EXP B_datasize - TEST Hit@1

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 25% of queries | 69.6 | 78.3 | 34.5 | 63.0 | 61.3 | 1 |
| in-batch, 50% | 70.3 | 79.2 | 35.1 | 62.0 | 61.7 | 1 |
| in-batch, 100% | 70.7 ± 0.3 | 79.6 ± 0.3 | 35.6 ± 1.2 | 61.2 ± 1.3 | 61.8 ± 0.2 | 3 |
| in-batch, 25% x 4 epochs (same #steps as 100% x 1) | 69.0 | 79.5 | 36.3 | 61.8 | 61.6 | 1 |
| hard neg, 25% of queries | 71.0 | 80.5 | 37.3 | 64.0 | 63.2 | 1 |
| hard neg, 50% | 72.3 | 80.6 | 38.6 | 64.0 | 63.9 | 1 |
| hard neg, 100% | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| hard neg, 25% x 4 epochs (same #steps as 100% x 1) | 71.4 | 80.3 | 37.7 | 63.5 | 63.2 | 1 |


### EXP B_datasize - TEST Hit@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 25% of queries | 91.5 | 96.3 | 70.3 | 90.3 | 87.1 | 1 |
| in-batch, 50% | 92.1 | 96.7 | 70.3 | 90.6 | 87.4 | 1 |
| in-batch, 100% | 92.2 ± 0.4 | 96.9 ± 0.1 | 71.0 ± 1.0 | 89.7 ± 0.8 | 87.4 ± 0.1 | 3 |
| in-batch, 25% x 4 epochs (same #steps as 100% x 1) | 91.2 | 96.5 | 71.8 | 89.4 | 87.2 | 1 |
| hard neg, 25% of queries | 92.1 | 96.5 | 72.1 | 90.5 | 87.8 | 1 |
| hard neg, 50% | 92.7 | 97.1 | 72.9 | 91.6 | 88.6 | 1 |
| hard neg, 100% | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| hard neg, 25% x 4 epochs (same #steps as 100% x 1) | 91.8 | 96.4 | 73.0 | 89.9 | 87.8 | 1 |


### EXP B_datasize - DEV MRR@10

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 25% of queries | 75.4 | 85.2 | n/a | n/a | 80.3 | 1 |
| in-batch, 50% | 75.6 | 85.6 | 58.1 | 74.7 | 80.6 | 1 |
| in-batch, 100% | 76.9 ± 0.5 | 86.5 ± 0.1 | n/a | n/a | 81.7 ± 0.2 | 3 |
| in-batch, 25% x 4 epochs (same #steps as 100% x 1) | 75.6 | 85.8 | 59.5 | 74.8 | 80.7 | 1 |
| hard neg, 25% of queries | 78.7 | 86.4 | n/a | n/a | 82.5 | 1 |
| hard neg, 50% | 79.1 | 87.2 | 61.1 | 76.0 | 83.2 | 1 |
| hard neg, 100% | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| hard neg, 25% x 4 epochs (same #steps as 100% x 1) | 77.4 | 86.8 | 60.7 | 73.8 | 82.1 | 1 |


### EXP C_epochs - TEST MRR@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 1 epoch | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| in-batch, 3 epochs | 78.3 | 86.7 | 46.5 | 71.7 | 70.8 | 1 |
| hard neg, 1 epoch | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| hard neg, 3 epochs | 80.6 | 87.8 | 48.9 | 72.1 | 72.4 | 1 |


### EXP C_epochs - TEST Hit@1

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 1 epoch | 70.7 ± 0.3 | 79.6 ± 0.3 | 35.6 ± 1.2 | 61.2 ± 1.3 | 61.8 ± 0.2 | 3 |
| in-batch, 3 epochs | 69.8 | 80.1 | 34.9 | 61.8 | 61.7 | 1 |
| hard neg, 1 epoch | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| hard neg, 3 epochs | 73.0 | 82.4 | 37.3 | 62.6 | 63.8 | 1 |


### EXP C_epochs - TEST Hit@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 1 epoch | 92.2 ± 0.4 | 96.9 ± 0.1 | 71.0 ± 1.0 | 89.7 ± 0.8 | 87.4 ± 0.1 | 3 |
| in-batch, 3 epochs | 92.5 | 96.9 | 72.1 | 90.0 | 87.9 | 1 |
| hard neg, 1 epoch | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| hard neg, 3 epochs | 92.6 | 96.9 | 73.6 | 89.7 | 88.2 | 1 |


### EXP C_epochs - DEV MRR@10

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| in-batch, 1 epoch | 76.9 ± 0.5 | 86.5 ± 0.1 | n/a | n/a | 81.7 ± 0.2 | 3 |
| in-batch, 3 epochs | 75.9 | 86.7 | n/a | n/a | 81.3 | 1 |
| hard neg, 1 epoch | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| hard neg, 3 epochs | 78.3 | 87.7 | n/a | n/a | 83.0 | 1 |


### EXP D_hyperparams - TEST MRR@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| default: bs 64, lr 3e-5, scale 20 | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| scale 50 (temperature 0.02) | 81.3 | 88.2 | 47.1 | 73.9 | 72.6 | 1 |
| lr 1e-5 | 79.8 | 86.8 | 48.1 | 73.1 | 71.9 | 1 |
| lr 6e-5 | 80.5 | 87.7 | 49.4 | 74.2 | 72.9 | 1 |
| batch 32 | 80.8 | 87.8 | 49.1 | 73.5 | 72.8 | 1 |
| batch 128 (CachedMNRL) | 80.1 | 87.2 | 48.5 | 73.4 | 72.3 | 1 |
| batch 512 (CachedMNRL, 32 steps) | 77.3 | 85.0 | 47.2 | 71.9 | 70.3 | 1 |


### EXP D_hyperparams - TEST Hit@1

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| default: bs 64, lr 3e-5, scale 20 | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| scale 50 (temperature 0.02) | 73.7 | 82.1 | 36.2 | 64.8 | 64.2 | 1 |
| lr 1e-5 | 72.1 | 80.4 | 37.2 | 63.6 | 63.3 | 1 |
| lr 6e-5 | 73.0 | 81.6 | 37.7 | 65.1 | 64.4 | 1 |
| batch 32 | 73.3 | 81.8 | 37.9 | 64.7 | 64.4 | 1 |
| batch 128 (CachedMNRL) | 72.4 | 80.9 | 38.0 | 64.2 | 63.9 | 1 |
| batch 512 (CachedMNRL, 32 steps) | 69.1 | 78.3 | 36.0 | 62.2 | 61.4 | 1 |


### EXP D_hyperparams - TEST Hit@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| default: bs 64, lr 3e-5, scale 20 | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| scale 50 (temperature 0.02) | 93.2 | 97.4 | 71.4 | 91.0 | 88.3 | 1 |
| lr 1e-5 | 92.2 | 96.9 | 72.3 | 90.9 | 88.1 | 1 |
| lr 6e-5 | 92.7 | 97.2 | 74.4 | 90.7 | 88.7 | 1 |
| batch 32 | 92.9 | 97.2 | 73.5 | 90.8 | 88.6 | 1 |
| batch 128 (CachedMNRL) | 92.5 | 97.1 | 72.2 | 90.4 | 88.1 | 1 |
| batch 512 (CachedMNRL, 32 steps) | 91.2 | 96.1 | 71.9 | 90.0 | 87.3 | 1 |


### EXP D_hyperparams - DEV MRR@10

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| default: bs 64, lr 3e-5, scale 20 | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| scale 50 (temperature 0.02) | 77.4 | 88.3 | 59.5 | 75.6 | 82.8 | 1 |
| lr 1e-5 | 78.4 | 87.0 | 60.0 | 74.8 | 82.7 | 1 |
| lr 6e-5 | 78.9 | 87.4 | 61.6 | 74.1 | 83.2 | 1 |
| batch 32 | 78.5 | 87.8 | 62.2 | 73.9 | 83.2 | 1 |
| batch 128 (CachedMNRL) | 78.3 | 87.2 | 59.7 | 74.9 | 82.7 | 1 |
| batch 512 (CachedMNRL, 32 steps) | 77.3 | 84.6 | 59.3 | 74.2 | 81.0 | 1 |


### EXP E_crosslingual - TEST MRR@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 | 1 |
| in-batch, monolingual data | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| in-batch, + translated cross-lingual queries | 79.0 | 86.1 | 53.3 | 80.1 | 74.6 | 1 |
| hard neg, monolingual data | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 | 3 |
| hard neg x3 (cached), + cross-lingual | 79.9 | 87.2 | 55.2 | 80.3 | 75.6 | 1 |
| hard neg, + cross-lingual, 2 epochs | 80.2 | 87.3 | 54.9 | 80.8 | 75.8 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 80.9 | 87.1 | 54.1 | 80.8 | 75.7 | 1 |
| bge-m3 dense (568M, off the shelf) | 80.2 | 87.8 | 63.3 | 67.8 | 74.8 | 1 |


### EXP E_crosslingual - TEST Hit@1

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 | 1 |
| in-batch, monolingual data | 70.7 ± 0.3 | 79.6 ± 0.3 | 35.6 ± 1.2 | 61.2 ± 1.3 | 61.8 ± 0.2 | 3 |
| in-batch, + translated cross-lingual queries | 71.0 | 79.6 | 41.7 | 72.4 | 66.2 | 1 |
| hard neg, monolingual data | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 72.8 ± 0.2 | 81.4 ± 0.3 | 42.8 ± 0.8 | 72.7 ± 0.5 | 67.4 ± 0.2 | 3 |
| hard neg x3 (cached), + cross-lingual | 72.3 | 81.4 | 43.1 | 72.4 | 67.3 | 1 |
| hard neg, + cross-lingual, 2 epochs | 72.8 | 81.6 | 43.2 | 73.2 | 67.7 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 73.9 | 81.0 | 42.4 | 73.0 | 67.6 | 1 |
| bge-m3 dense (568M, off the shelf) | 73.1 | 82.1 | 52.4 | 59.2 | 66.7 | 1 |


### EXP E_crosslingual - TEST Hit@10

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 | 1 |
| in-batch, monolingual data | 92.2 ± 0.4 | 96.9 ± 0.1 | 71.0 ± 1.0 | 89.7 ± 0.8 | 87.4 ± 0.1 | 3 |
| in-batch, + translated cross-lingual queries | 91.8 | 96.8 | 77.0 | 94.0 | 89.9 | 1 |
| hard neg, monolingual data | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 92.3 ± 0.2 | 96.9 ± 0.1 | 79.6 ± 0.5 | 94.1 ± 0.3 | 90.7 ± 0.1 | 3 |
| hard neg x3 (cached), + cross-lingual | 92.1 | 96.9 | 80.2 | 93.9 | 90.8 | 1 |
| hard neg, + cross-lingual, 2 epochs | 92.2 | 96.8 | 79.6 | 94.3 | 90.7 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 92.0 | 97.0 | 78.4 | 94.7 | 90.5 | 1 |
| bge-m3 dense (568M, off the shelf) | 91.9 | 97.0 | 84.7 | 84.4 | 89.5 | 1 |


### EXP E_crosslingual - DEV MRR@10

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small, no fine-tuning | 72.6 | 87.3 | 55.5 | 67.7 | 79.9 | 1 |
| in-batch, monolingual data | 76.9 ± 0.5 | 86.5 ± 0.1 | n/a | n/a | 81.7 ± 0.2 | 3 |
| in-batch, + translated cross-lingual queries | 76.5 | 86.5 | 65.4 | 78.1 | 81.5 | 1 |
| hard neg, monolingual data | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| hard neg, + translated cross-lingual queries  (FINAL RECIPE) | 78.1 ± 0.5 | 87.1 ± 0.1 | 66.1 ± 0.2 | 78.5 ± 0.4 | 82.6 ± 0.2 | 3 |
| hard neg x3 (cached), + cross-lingual | 78.0 | 87.1 | 66.0 | 78.5 | 82.6 | 1 |
| hard neg, + cross-lingual, 2 epochs | 77.5 | 87.0 | 67.3 | 80.4 | 82.3 | 1 |
| hard neg mined by bge-m3, + cross-lingual | 79.0 | 87.9 | 66.9 | 77.6 | 83.5 | 1 |


### Paired bootstrap, TEST MRR@10 difference [95% CI] in points (* = CI excludes 0)

| Comparison | ar->ar | en->en | ar->en | en->ar |
|---|---:|---:|---:|---:|
| hard neg (3 seeds, seed 13) vs in-batch only (seed 13) | +1.8 [+1.2, +2.4]* | +1.1 [+0.5, +1.7]* | +3.3 [+2.3, +4.3]* | +2.4 [+1.5, +3.3]* |
| random neg (seed 13) vs in-batch only (seed 13) | -0.3 [-0.7, +0.1] | +0.4 [-0.0, +0.8] | +0.7 [-0.1, +1.5] | +0.1 [-0.6, +0.6] |
| hard neg margin 0.02 vs no filter (seed 13) | +0.8 [-0.1, +1.7] | +0.8 [-0.0, +1.8] | +4.1 [+2.8, +5.3]* | -1.3 [-2.4, -0.3]* |
| + cross-lingual data vs mono, hard neg (seed 13) | -0.5 [-1.0, +0.1] | -0.2 [-0.6, +0.2] | +5.9 [+4.7, +7.2]* | +6.4 [+5.2, +7.5]* |
| FINAL (hard neg + xling, seed 13) vs e5-small baseline | +3.3 [+2.5, +4.2]* | -0.5 [-1.4, +0.4] | +13.7 [+11.8, +15.7]* | +18.7 [+16.9, +20.6]* |
| FINAL (hard neg + xling, seed 13) vs bge-m3 dense | +0.0 [-1.0, +1.1] | -0.5 [-1.6, +0.5] | -8.2 [-10.4, -6.2]* | +12.6 [+10.5, +14.7]* |


### Fine-tuned embedders - TEST MRR@10 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small baseline | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 | 1 |
| e5s_hard1_bgem3_e1 | 81.2 | 87.9 | 48.2 | 75.1 | 73.1 | 1 |
| e5s_hard1_bs128c_e1 | 80.1 | 87.2 | 48.5 | 73.4 | 72.3 | 1 |
| e5s_hard1_bs32_e1 | 80.8 | 87.8 | 49.1 | 73.5 | 72.8 | 1 |
| e5s_hard1_cached512_e1 | 77.3 | 85.0 | 47.2 | 71.9 | 70.3 | 1 |
| e5s_hard1_e1 | 80.8 ± 0.2 | 87.7 ± 0.2 | 49.4 ± 0.6 | 74.1 ± 0.1 | 73.0 ± 0.2 | 3 |
| e5s_hard1_e3 | 80.6 | 87.8 | 48.9 | 72.1 | 72.4 | 1 |
| e5s_hard1_f25_e1 | 79.0 | 86.5 | 48.4 | 73.1 | 71.7 | 1 |
| e5s_hard1_f25_e4 | 79.1 | 86.5 | 48.9 | 72.6 | 71.8 | 1 |
| e5s_hard1_f50_e1 | 80.1 | 87.0 | 49.5 | 73.5 | 72.5 | 1 |
| e5s_hard1_lr1e5_e1 | 79.8 | 86.8 | 48.1 | 73.1 | 71.9 | 1 |
| e5s_hard1_lr6e5_e1 | 80.5 | 87.7 | 49.4 | 74.2 | 72.9 | 1 |
| e5s_hard1_m05_e1 | 79.4 | 86.8 | 47.5 | 72.4 | 71.5 | 1 |
| e5s_hard1_nomargin_e1 | 79.9 | 86.7 | 45.1 | 75.4 | 71.8 | 1 |
| e5s_hard1_notitle_e1 | 79.7 | 86.8 | 46.6 | 72.8 | 71.5 | 1 |
| e5s_hard1_rank10_e1 | 79.3 | 87.2 | 47.3 | 72.5 | 71.6 | 1 |
| e5s_hard1_sc50_e1 | 81.3 | 88.2 | 47.1 | 73.9 | 72.6 | 1 |
| e5s_hard1_xl_bgem3_e1 | 80.9 | 87.1 | 54.1 | 80.8 | 75.7 | 1 |
| e5s_hard1_xl_e1 | 80.3 ± 0.2 | 87.3 ± 0.2 | 54.7 ± 0.6 | 80.4 ± 0.4 | 75.7 ± 0.1 | 3 |
| e5s_hard1_xl_e1_s13>>rr_ft_bce | 86.7 | 90.6 | 72.3 | 83.7 | 83.3 | 1 |
| e5s_hard1_xl_e2 | 80.2 | 87.3 | 54.9 | 80.8 | 75.8 | 1 |
| e5s_hard3_e1 | 80.2 | 87.6 | 49.7 | 73.8 | 72.8 | 1 |
| e5s_hard3c_xl_e1 | 79.9 | 87.2 | 55.2 | 80.3 | 75.6 | 1 |
| e5s_inb_e1 | 78.8 ± 0.1 | 86.4 ± 0.3 | 46.7 ± 1.0 | 71.1 ± 1.1 | 70.8 ± 0.2 | 3 |
| e5s_inb_e3 | 78.3 | 86.7 | 46.5 | 71.7 | 70.8 | 1 |
| e5s_inb_f25_e1 | 77.8 | 85.0 | 45.7 | 72.4 | 70.2 | 1 |
| e5s_inb_f25_e4 | 77.4 | 85.8 | 47.2 | 71.4 | 70.5 | 1 |
| e5s_inb_f50_e1 | 78.5 | 85.8 | 46.2 | 71.8 | 70.6 | 1 |
| e5s_inb_xl_e1 | 79.0 | 86.1 | 53.3 | 80.1 | 74.6 | 1 |
| e5s_randneg1_e1 | 78.6 | 86.8 | 46.6 | 71.7 | 70.9 | 1 |


### Fine-tuned embedders - TEST Hit@1 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small baseline | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 | 1 |
| e5s_hard1_bgem3_e1 | 74.2 | 82.1 | 37.2 | 66.4 | 65.0 | 1 |
| e5s_hard1_bs128c_e1 | 72.4 | 80.9 | 38.0 | 64.2 | 63.9 | 1 |
| e5s_hard1_bs32_e1 | 73.3 | 81.8 | 37.9 | 64.7 | 64.4 | 1 |
| e5s_hard1_cached512_e1 | 69.1 | 78.3 | 36.0 | 62.2 | 61.4 | 1 |
| e5s_hard1_e1 | 73.5 ± 0.3 | 81.8 ± 0.3 | 38.2 ± 0.4 | 65.0 ± 0.3 | 64.6 ± 0.2 | 3 |
| e5s_hard1_e3 | 73.0 | 82.4 | 37.3 | 62.6 | 63.8 | 1 |
| e5s_hard1_f25_e1 | 71.0 | 80.5 | 37.3 | 64.0 | 63.2 | 1 |
| e5s_hard1_f25_e4 | 71.4 | 80.3 | 37.7 | 63.5 | 63.2 | 1 |
| e5s_hard1_f50_e1 | 72.3 | 80.6 | 38.6 | 64.0 | 63.9 | 1 |
| e5s_hard1_lr1e5_e1 | 72.1 | 80.4 | 37.2 | 63.6 | 63.3 | 1 |
| e5s_hard1_lr6e5_e1 | 73.0 | 81.6 | 37.7 | 65.1 | 64.4 | 1 |
| e5s_hard1_m05_e1 | 71.2 | 80.4 | 36.2 | 62.8 | 62.6 | 1 |
| e5s_hard1_nomargin_e1 | 72.7 | 80.4 | 34.8 | 66.2 | 63.5 | 1 |
| e5s_hard1_notitle_e1 | 71.7 | 80.5 | 35.7 | 63.4 | 62.8 | 1 |
| e5s_hard1_rank10_e1 | 71.4 | 81.1 | 36.1 | 62.9 | 62.9 | 1 |
| e5s_hard1_sc50_e1 | 73.7 | 82.1 | 36.2 | 64.8 | 64.2 | 1 |
| e5s_hard1_xl_bgem3_e1 | 73.9 | 81.0 | 42.4 | 73.0 | 67.6 | 1 |
| e5s_hard1_xl_e1 | 72.8 ± 0.2 | 81.4 ± 0.3 | 42.8 ± 0.8 | 72.7 ± 0.5 | 67.4 ± 0.2 | 3 |
| e5s_hard1_xl_e1_s13>>rr_ft_bce | 82.3 | 86.4 | 64.2 | 77.7 | 77.7 | 1 |
| e5s_hard1_xl_e2 | 72.8 | 81.6 | 43.2 | 73.2 | 67.7 | 1 |
| e5s_hard3_e1 | 72.3 | 81.6 | 38.9 | 64.4 | 64.3 | 1 |
| e5s_hard3c_xl_e1 | 72.3 | 81.4 | 43.1 | 72.4 | 67.3 | 1 |
| e5s_inb_e1 | 70.7 ± 0.3 | 79.6 ± 0.3 | 35.6 ± 1.2 | 61.2 ± 1.3 | 61.8 ± 0.2 | 3 |
| e5s_inb_e3 | 69.8 | 80.1 | 34.9 | 61.8 | 61.7 | 1 |
| e5s_inb_f25_e1 | 69.6 | 78.3 | 34.5 | 63.0 | 61.3 | 1 |
| e5s_inb_f25_e4 | 69.0 | 79.5 | 36.3 | 61.8 | 61.6 | 1 |
| e5s_inb_f50_e1 | 70.3 | 79.2 | 35.1 | 62.0 | 61.7 | 1 |
| e5s_inb_xl_e1 | 71.0 | 79.6 | 41.7 | 72.4 | 66.2 | 1 |
| e5s_randneg1_e1 | 70.2 | 80.5 | 35.7 | 61.8 | 62.0 | 1 |


### Fine-tuned embedders - TEST Hit@10 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small baseline | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 | 1 |
| e5s_hard1_bgem3_e1 | 92.8 | 97.1 | 72.4 | 91.2 | 88.4 | 1 |
| e5s_hard1_bs128c_e1 | 92.5 | 97.1 | 72.2 | 90.4 | 88.1 | 1 |
| e5s_hard1_bs32_e1 | 92.9 | 97.2 | 73.5 | 90.8 | 88.6 | 1 |
| e5s_hard1_cached512_e1 | 91.2 | 96.1 | 71.9 | 90.0 | 87.3 | 1 |
| e5s_hard1_e1 | 92.7 ± 0.0 | 97.1 ± 0.2 | 73.9 ± 0.4 | 91.1 ± 0.2 | 88.7 ± 0.2 | 3 |
| e5s_hard1_e3 | 92.6 | 96.9 | 73.6 | 89.7 | 88.2 | 1 |
| e5s_hard1_f25_e1 | 92.1 | 96.5 | 72.1 | 90.5 | 87.8 | 1 |
| e5s_hard1_f25_e4 | 91.8 | 96.4 | 73.0 | 89.9 | 87.8 | 1 |
| e5s_hard1_f50_e1 | 92.7 | 97.1 | 72.9 | 91.6 | 88.6 | 1 |
| e5s_hard1_lr1e5_e1 | 92.2 | 96.9 | 72.3 | 90.9 | 88.1 | 1 |
| e5s_hard1_lr6e5_e1 | 92.7 | 97.2 | 74.4 | 90.7 | 88.7 | 1 |
| e5s_hard1_m05_e1 | 92.5 | 96.9 | 72.4 | 90.3 | 88.0 | 1 |
| e5s_hard1_nomargin_e1 | 91.9 | 96.6 | 69.2 | 91.4 | 87.3 | 1 |
| e5s_hard1_notitle_e1 | 92.6 | 96.9 | 70.2 | 90.3 | 87.5 | 1 |
| e5s_hard1_rank10_e1 | 92.2 | 97.0 | 71.3 | 90.5 | 87.8 | 1 |
| e5s_hard1_sc50_e1 | 93.2 | 97.4 | 71.4 | 91.0 | 88.3 | 1 |
| e5s_hard1_xl_bgem3_e1 | 92.0 | 97.0 | 78.4 | 94.7 | 90.5 | 1 |
| e5s_hard1_xl_e1 | 92.3 ± 0.2 | 96.9 ± 0.1 | 79.6 ± 0.5 | 94.1 ± 0.3 | 90.7 ± 0.1 | 3 |
| e5s_hard1_xl_e1_s13>>rr_ft_bce | 94.1 | 97.2 | 86.1 | 94.8 | 93.0 | 1 |
| e5s_hard1_xl_e2 | 92.2 | 96.8 | 79.6 | 94.3 | 90.7 | 1 |
| e5s_hard3_e1 | 92.8 | 97.0 | 72.9 | 91.1 | 88.5 | 1 |
| e5s_hard3c_xl_e1 | 92.1 | 96.9 | 80.2 | 93.9 | 90.8 | 1 |
| e5s_inb_e1 | 92.2 ± 0.4 | 96.9 ± 0.1 | 71.0 ± 1.0 | 89.7 ± 0.8 | 87.4 ± 0.1 | 3 |
| e5s_inb_e3 | 92.5 | 96.9 | 72.1 | 90.0 | 87.9 | 1 |
| e5s_inb_f25_e1 | 91.5 | 96.3 | 70.3 | 90.3 | 87.1 | 1 |
| e5s_inb_f25_e4 | 91.2 | 96.5 | 71.8 | 89.4 | 87.2 | 1 |
| e5s_inb_f50_e1 | 92.1 | 96.7 | 70.3 | 90.6 | 87.4 | 1 |
| e5s_inb_xl_e1 | 91.8 | 96.8 | 77.0 | 94.0 | 89.9 | 1 |
| e5s_randneg1_e1 | 92.0 | 96.7 | 70.5 | 90.1 | 87.3 | 1 |


### Fine-tuned embedders - TEST nDCG@10 (x100)

| System | ar->ar | en->en | ar->en | en->ar | macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small baseline | 80.2 | 90.2 | 46.5 | 66.8 | 70.9 | 1 |
| e5s_hard1_bgem3_e1 | 84.1 | 90.2 | 54.0 | 79.0 | 76.8 | 1 |
| e5s_hard1_bs128c_e1 | 83.2 | 89.7 | 54.2 | 77.6 | 76.1 | 1 |
| e5s_hard1_bs32_e1 | 83.8 | 90.1 | 54.9 | 77.7 | 76.6 | 1 |
| e5s_hard1_cached512_e1 | 80.8 | 87.7 | 53.1 | 76.3 | 74.5 | 1 |
| e5s_hard1_e1 | 83.8 ± 0.1 | 90.0 ± 0.2 | 55.3 ± 0.5 | 78.3 ± 0.1 | 76.8 ± 0.2 | 3 |
| e5s_hard1_e3 | 83.6 | 90.1 | 54.9 | 76.4 | 76.2 | 1 |
| e5s_hard1_f25_e1 | 82.2 | 89.0 | 54.1 | 77.4 | 75.6 | 1 |
| e5s_hard1_f25_e4 | 82.3 | 89.0 | 54.6 | 76.9 | 75.7 | 1 |
| e5s_hard1_f50_e1 | 83.2 | 89.5 | 55.1 | 77.9 | 76.4 | 1 |
| e5s_hard1_lr1e5_e1 | 82.9 | 89.3 | 53.9 | 77.4 | 75.9 | 1 |
| e5s_hard1_lr6e5_e1 | 83.5 | 90.0 | 55.4 | 78.3 | 76.8 | 1 |
| e5s_hard1_m05_e1 | 82.7 | 89.3 | 53.5 | 76.7 | 75.6 | 1 |
| e5s_hard1_nomargin_e1 | 82.8 | 89.1 | 50.9 | 79.3 | 75.5 | 1 |
| e5s_hard1_notitle_e1 | 82.9 | 89.3 | 52.3 | 77.1 | 75.4 | 1 |
| e5s_hard1_rank10_e1 | 82.5 | 89.6 | 53.1 | 76.9 | 75.5 | 1 |
| e5s_hard1_sc50_e1 | 84.3 | 90.5 | 52.9 | 78.1 | 76.4 | 1 |
| e5s_hard1_xl_bgem3_e1 | 83.6 | 89.5 | 60.0 | 84.2 | 79.3 | 1 |
| e5s_hard1_xl_e1 | 83.2 ± 0.2 | 89.6 ± 0.2 | 60.7 ± 0.5 | 83.8 ± 0.3 | 79.3 ± 0.0 | 3 |
| e5s_hard1_xl_e1_s13>>rr_ft_bce | 88.5 | 92.2 | 75.7 | 86.4 | 85.7 | 1 |
| e5s_hard1_xl_e2 | 83.1 | 89.6 | 60.8 | 84.1 | 79.4 | 1 |
| e5s_hard3_e1 | 83.3 | 90.0 | 55.3 | 78.0 | 76.6 | 1 |
| e5s_hard3c_xl_e1 | 82.9 | 89.6 | 61.2 | 83.6 | 79.3 | 1 |
| e5s_inb_e1 | 82.1 ± 0.0 | 89.0 ± 0.2 | 52.5 ± 1.0 | 75.7 ± 1.0 | 74.8 ± 0.1 | 3 |
| e5s_inb_e3 | 81.8 | 89.3 | 52.6 | 76.2 | 74.9 | 1 |
| e5s_inb_f25_e1 | 81.2 | 87.8 | 51.6 | 76.8 | 74.3 | 1 |
| e5s_inb_f25_e4 | 80.8 | 88.4 | 53.0 | 75.8 | 74.5 | 1 |
| e5s_inb_f50_e1 | 81.8 | 88.5 | 52.0 | 76.3 | 74.7 | 1 |
| e5s_inb_xl_e1 | 82.1 | 88.7 | 59.0 | 83.5 | 78.4 | 1 |
| e5s_randneg1_e1 | 81.9 | 89.3 | 52.3 | 76.1 | 74.9 | 1 |


### Fine-tuned embedders - DEV MRR@10 (x100, model selection only; mono macro = mean of ar->ar and en->en, the only directions every run was scored on; ar->en / en->ar are shown where the run was evaluated after the translated dev queries existed)

| System | ar->ar | en->en | ar->en | en->ar | mono macro | seeds |
|---|---:|---:|---:|---:|---:|---:|
| e5-small baseline | 72.6 | 87.3 | 55.5 | 67.7 | 79.9 | 1 |
| e5s_hard1_bgem3_e1 | 78.4 | 88.2 | 61.1 | 74.4 | 83.3 | 1 |
| e5s_hard1_bs128c_e1 | 78.3 | 87.2 | 59.7 | 74.9 | 82.7 | 1 |
| e5s_hard1_bs32_e1 | 78.5 | 87.8 | 62.2 | 73.9 | 83.2 | 1 |
| e5s_hard1_cached512_e1 | 77.3 | 84.6 | 59.3 | 74.2 | 81.0 | 1 |
| e5s_hard1_e1 | 78.4 ± 0.4 | 87.6 ± 0.1 | n/a | n/a | 83.0 ± 0.3 | 3 |
| e5s_hard1_e3 | 78.3 | 87.7 | n/a | n/a | 83.0 | 1 |
| e5s_hard1_f25_e1 | 78.7 | 86.4 | n/a | n/a | 82.5 | 1 |
| e5s_hard1_f25_e4 | 77.4 | 86.8 | 60.7 | 73.8 | 82.1 | 1 |
| e5s_hard1_f50_e1 | 79.1 | 87.2 | 61.1 | 76.0 | 83.2 | 1 |
| e5s_hard1_lr1e5_e1 | 78.4 | 87.0 | 60.0 | 74.8 | 82.7 | 1 |
| e5s_hard1_lr6e5_e1 | 78.9 | 87.4 | 61.6 | 74.1 | 83.2 | 1 |
| e5s_hard1_m05_e1 | 78.5 | 86.8 | 58.7 | 74.6 | 82.6 | 1 |
| e5s_hard1_nomargin_e1 | 76.7 | 87.6 | 59.2 | 72.1 | 82.1 | 1 |
| e5s_hard1_notitle_e1 | 77.4 | 86.9 | 59.3 | 73.8 | 82.1 | 1 |
| e5s_hard1_rank10_e1 | 78.0 | 86.9 | 59.5 | 73.8 | 82.4 | 1 |
| e5s_hard1_sc50_e1 | 77.4 | 88.3 | 59.5 | 75.6 | 82.8 | 1 |
| e5s_hard1_xl_bgem3_e1 | 79.0 | 87.9 | 66.9 | 77.6 | 83.5 | 1 |
| e5s_hard1_xl_e1 | 78.1 ± 0.5 | 87.1 ± 0.1 | 66.1 ± 0.2 | 78.5 ± 0.4 | 82.6 ± 0.2 | 3 |
| e5s_hard1_xl_e1_s13>>rr_ft_bce | n/a | n/a | n/a | n/a | n/a | 0 |
| e5s_hard1_xl_e2 | 77.5 | 87.0 | 67.3 | 80.4 | 82.3 | 1 |
| e5s_hard3_e1 | 79.3 | 87.3 | n/a | n/a | 83.3 | 1 |
| e5s_hard3c_xl_e1 | 78.0 | 87.1 | 66.0 | 78.5 | 82.6 | 1 |
| e5s_inb_e1 | 76.9 ± 0.5 | 86.5 ± 0.1 | n/a | n/a | 81.7 ± 0.2 | 3 |
| e5s_inb_e3 | 75.9 | 86.7 | n/a | n/a | 81.3 | 1 |
| e5s_inb_f25_e1 | 75.4 | 85.2 | n/a | n/a | 80.3 | 1 |
| e5s_inb_f25_e4 | 75.6 | 85.8 | 59.5 | 74.8 | 80.7 | 1 |
| e5s_inb_f50_e1 | 75.6 | 85.6 | 58.1 | 74.7 | 80.6 | 1 |
| e5s_inb_xl_e1 | 76.5 | 86.5 | 65.4 | 78.1 | 81.5 | 1 |
| e5s_randneg1_e1 | 76.9 | 87.0 | 58.0 | 73.9 | 81.9 | 1 |


### Training cost

| Run | examples | steps | train s | peak GPU GB | first loss | last loss | padded negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| e5s_hard1_bgem3_e1_s13 | 16419 | 256 | 720 | 8.95 | 1.360 | 0.262 | 558 |
| e5s_hard1_bs128c_e1_s13 | 16419 | 128 | 487 | 5.36 | 1.386 | 0.264 | 996 |
| e5s_hard1_bs32_e1_s13 | 16419 | 513 | 393 | 5.22 | 1.299 | 0.150 | 996 |
| e5s_hard1_cached512_e1_s13 | 16419 | 32 | 492 | 5.36 | 1.666 | 1.666 | 996 |
| e5s_hard1_e1_s13 | 16419 | 256 | 305 | 8.99 | 1.348 | 0.199 | 996 |
| e5s_hard1_e1_s14 | 16419 | 256 | 259 | 8.99 | 1.349 | 0.217 | 996 |
| e5s_hard1_e1_s15 | 16419 | 256 | 272 | 8.99 | 1.354 | 0.218 | 996 |
| e5s_hard1_e3_s13 | 16419 | 768 | 852 | 8.99 | 1.864 | 0.099 | 996 |
| e5s_hard1_f25_e1_s13 | 4105 | 64 | 63 | 8.99 | 0.902 | 0.269 | 259 |
| e5s_hard1_f25_e4_s13 | 4105 | 256 | 474 | 8.99 | 1.347 | 0.091 | 259 |
| e5s_hard1_f50_e1_s13 | 8209 | 128 | 268 | 8.99 | 1.095 | 0.228 | 494 |
| e5s_hard1_lr1e5_e1_s13 | 16419 | 256 | 383 | 8.95 | 1.858 | 0.221 | 996 |
| e5s_hard1_lr6e5_e1_s13 | 16419 | 256 | 401 | 8.95 | 1.090 | 0.192 | 996 |
| e5s_hard1_m05_e1_s13 | 16419 | 256 | 514 | 8.95 | 1.299 | 0.132 | 3760 |
| e5s_hard1_nomargin_e1_s13 | 16419 | 256 | 510 | 8.99 | 1.454 | 0.437 | 0 |
| e5s_hard1_notitle_e1_s13 | 16419 | 256 | 400 | 8.95 | 1.326 | 0.153 | 1264 |
| e5s_hard1_rank10_e1_s13 | 16419 | 256 | 405 | 8.95 | 1.307 | 0.153 | 996 |
| e5s_hard1_sc50_e1_s13 | 16419 | 256 | 387 | 8.95 | 0.363 | 0.113 | 996 |
| e5s_hard1_xl_bgem3_e1_s13 | 20918 | 326 | 434 | 8.95 | 1.827 | 0.320 | 965 |
| e5s_hard1_xl_e1_s13 | 20918 | 326 | 611 | 8.99 | 1.817 | 0.267 | 2480 |
| e5s_hard1_xl_e1_s14 | 20918 | 326 | 753 | 8.99 | 1.836 | 0.286 | 2480 |
| e5s_hard1_xl_e1_s15 | 20918 | 326 | 699 | 8.95 | 1.810 | 0.276 | 2480 |
| e5s_hard1_xl_e2_s13 | 20918 | 652 | 889 | 8.95 | 2.176 | 0.183 | 2480 |
| e5s_hard3_e1_s13 | 16419 | 256 | 688 | 15.85 | 1.847 | 0.315 | 3053 |
| e5s_hard3c_xl_e1_s13 | 20918 | 326 | 1764 | 5.36 | 2.386 | 0.422 | 7546 |
| e5s_inb_e1_s13 | 16419 | 256 | 281 | 5.56 | 0.871 | 0.078 | 0 |
| e5s_inb_e1_s14 | 16419 | 256 | 240 | 5.57 | 0.870 | 0.089 | 0 |
| e5s_inb_e1_s15 | 16419 | 256 | 161 | 5.56 | 0.885 | 0.076 | 0 |
| e5s_inb_e3_s13 | 16419 | 768 | 635 | 5.56 | 1.263 | 0.029 | 0 |
| e5s_inb_f25_e1_s13 | 4105 | 64 | 35 | 5.56 | 0.539 | 0.103 | 0 |
| e5s_inb_f25_e4_s13 | 4105 | 256 | 142 | 5.56 | 0.867 | 0.032 | 0 |
| e5s_inb_f50_e1_s13 | 8209 | 128 | 156 | 5.57 | 0.689 | 0.083 | 0 |
| e5s_inb_xl_e1_s13 | 20918 | 326 | 326 | 5.56 | 1.269 | 0.123 | 0 |
| e5s_randneg1_e1_s13 | 16419 | 256 | 546 | 8.95 | 1.238 | 0.123 | 0 |


### Dev curves (evaluated at the end of every epoch)

- e5s_hard1_e3_s13: epoch 1: dev macro MRR@10 83.3, epoch 2: dev macro MRR@10 82.8, epoch 3: dev macro MRR@10 83.0
- e5s_inb_e3_s13: epoch 1: dev macro MRR@10 82.0, epoch 2: dev macro MRR@10 81.0, epoch 3: dev macro MRR@10 81.2


### TEST MRR@10 change vs the e5-small baseline, paired bootstrap 95% CI (points; * = CI excludes 0)

| Comparison | ar->ar | en->en | ar->en | en->ar |
|---|---:|---:|---:|---:|
| e5s_hard1_bgem3_e1 (seed 13) - e5-small | +4.4 [+3.6, +5.2]* | +0.1 [-0.8, +0.9] | +6.8 [+5.1, +8.7]* | +13.4 [+11.8, +15.0]* |
| e5s_hard1_bs128c_e1 (seed 13) - e5-small | +3.3 [+2.4, +4.1]* | -0.6 [-1.5, +0.2] | +7.1 [+5.5, +8.8]* | +11.7 [+10.1, +13.4]* |
| e5s_hard1_bs32_e1 (seed 13) - e5-small | +4.0 [+3.2, +4.8]* | -0.0 [-0.9, +0.8] | +7.7 [+6.1, +9.5]* | +11.8 [+10.1, +13.5]* |
| e5s_hard1_cached512_e1 (seed 13) - e5-small | +0.5 [-0.3, +1.3] | -2.8 [-3.8, -1.9]* | +5.8 [+4.1, +7.5]* | +10.2 [+8.5, +11.9]* |
| e5s_hard1_e1 (seed 13) - e5-small | +3.8 [+3.0, +4.6]* | -0.3 [-1.1, +0.5] | +7.8 [+6.2, +9.5]* | +12.4 [+10.6, +14.1]* |
| e5s_hard1_e3 (seed 13) - e5-small | +3.8 [+3.0, +4.6]* | +0.0 [-0.8, +0.9] | +7.5 [+5.7, +9.4]* | +10.5 [+8.8, +12.1]* |
| e5s_hard1_f25_e1 (seed 13) - e5-small | +2.1 [+1.4, +2.9]* | -1.3 [-2.2, -0.4]* | +7.0 [+5.3, +8.6]* | +11.5 [+9.8, +13.2]* |
| e5s_hard1_f25_e4 (seed 13) - e5-small | +2.3 [+1.5, +3.1]* | -1.3 [-2.2, -0.4]* | +7.5 [+5.7, +9.2]* | +11.0 [+9.3, +12.6]* |
| e5s_hard1_f50_e1 (seed 13) - e5-small | +3.2 [+2.5, +4.0]* | -0.8 [-1.7, -0.0]* | +8.1 [+6.5, +9.7]* | +11.8 [+10.2, +13.5]* |
| e5s_hard1_lr1e5_e1 (seed 13) - e5-small | +3.0 [+2.2, +3.7]* | -1.0 [-1.8, -0.2]* | +6.7 [+5.1, +8.3]* | +11.4 [+9.8, +13.1]* |
| e5s_hard1_lr6e5_e1 (seed 13) - e5-small | +3.7 [+2.9, +4.5]* | -0.1 [-1.0, +0.7] | +8.0 [+6.3, +9.7]* | +12.6 [+10.9, +14.3]* |
| e5s_hard1_m05_e1 (seed 13) - e5-small | +2.6 [+1.8, +3.4]* | -1.0 [-1.9, -0.2]* | +6.1 [+4.5, +7.8]* | +10.7 [+9.1, +12.4]* |
| e5s_hard1_nomargin_e1 (seed 13) - e5-small | +3.0 [+2.0, +4.2]* | -1.1 [-2.2, +0.0] | +3.7 [+1.9, +5.8]* | +13.7 [+11.9, +15.5]* |
| e5s_hard1_notitle_e1 (seed 13) - e5-small | +2.9 [+2.1, +3.8]* | -1.0 [-2.0, -0.1]* | +5.2 [+3.6, +7.0]* | +11.2 [+9.5, +13.0]* |
| e5s_hard1_rank10_e1 (seed 13) - e5-small | +2.5 [+1.6, +3.4]* | -0.6 [-1.4, +0.3] | +5.9 [+4.2, +7.7]* | +10.8 [+9.1, +12.6]* |
| e5s_hard1_sc50_e1 (seed 13) - e5-small | +4.5 [+3.8, +5.2]* | +0.4 [-0.3, +1.0] | +5.7 [+4.3, +7.2]* | +12.2 [+10.7, +13.7]* |
| e5s_hard1_xl_bgem3_e1 (seed 13) - e5-small | +4.1 [+3.2, +5.0]* | -0.7 [-1.6, +0.2] | +12.7 [+10.9, +14.7]* | +19.1 [+17.2, +21.0]* |
| e5s_hard1_xl_e1 (seed 13) - e5-small | +3.3 [+2.5, +4.2]* | -0.5 [-1.4, +0.4] | +13.7 [+11.8, +15.7]* | +18.7 [+16.9, +20.6]* |
| e5s_hard1_xl_e1_s13>>rr_ft_bce (seed 13) - e5-small | +9.8 [+8.6, +11.1]* | +2.8 [+1.4, +4.2]* | +30.9 [+28.4, +33.4]* | +22.0 [+19.5, +24.6]* |
| e5s_hard1_xl_e2 (seed 13) - e5-small | +3.3 [+2.5, +4.3]* | -0.5 [-1.4, +0.4] | +13.5 [+11.7, +15.5]* | +19.2 [+17.4, +21.1]* |
| e5s_hard3_e1 (seed 13) - e5-small | +3.4 [+2.5, +4.1]* | -0.2 [-1.0, +0.7] | +8.3 [+6.6, +10.0]* | +12.1 [+10.4, +13.8]* |
| e5s_hard3c_xl_e1 (seed 13) - e5-small | +3.1 [+2.2, +4.0]* | -0.6 [-1.5, +0.3] | +13.8 [+11.9, +15.8]* | +18.6 [+16.7, +20.5]* |
| e5s_inb_e1 (seed 13) - e5-small | +2.0 [+1.1, +2.9]* | -1.4 [-2.3, -0.5]* | +4.5 [+2.7, +6.3]* | +9.9 [+8.2, +11.7]* |
| e5s_inb_e3 (seed 13) - e5-small | +1.5 [+0.6, +2.4]* | -1.1 [-2.0, -0.1]* | +5.1 [+3.3, +7.0]* | +10.0 [+8.3, +11.9]* |
| e5s_inb_f25_e1 (seed 13) - e5-small | +1.0 [+0.1, +1.8]* | -2.8 [-3.8, -1.9]* | +4.3 [+2.6, +6.1]* | +10.7 [+9.0, +12.6]* |
| e5s_inb_f25_e4 (seed 13) - e5-small | +0.6 [-0.3, +1.4] | -2.0 [-3.0, -1.1]* | +5.8 [+4.0, +7.6]* | +9.8 [+8.0, +11.6]* |
| e5s_inb_f50_e1 (seed 13) - e5-small | +1.6 [+0.8, +2.5]* | -2.0 [-3.1, -1.1]* | +4.8 [+3.1, +6.6]* | +10.1 [+8.4, +11.9]* |
| e5s_inb_xl_e1 (seed 13) - e5-small | +2.2 [+1.3, +3.0]* | -1.7 [-2.6, -0.8]* | +11.9 [+10.1, +13.9]* | +18.4 [+16.5, +20.3]* |
| e5s_randneg1_e1 (seed 13) - e5-small | +1.7 [+0.8, +2.6]* | -1.0 [-1.9, -0.1]* | +5.2 [+3.4, +7.0]* | +10.0 [+8.3, +11.8]* |


### RERANK - TEST MRR@10 (x100); every reranker sees the first stage's top-50

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 76.8 | 87.8 | 41.4 | 61.7 | 66.9 |
| (a) + mMiniLM cross-encoder, off the shelf | 85.3 | 90.9 | 65.7 | 80.0 | 80.5 |
| (a) + fine-tuned reranker (mined hard negatives, + cross-lingual) | 87.2 | 91.0 | 68.9 | 83.8 | 82.7 |
| (a) + fine-tuned reranker, RANDOM negatives | 85.3 | 85.9 | 66.2 | 83.1 | 80.1 |
| (a) + fine-tuned reranker, no cross-lingual training data | 86.8 | 90.7 | 67.3 | 82.3 | 81.7 |
| (a) + fine-tuned reranker, rerank only top-10 | 85.9 | 91.7 | 58.0 | 78.1 | 78.4 |
| (a) + fine-tuned reranker, rerank only top-20 | 86.9 | 91.5 | 64.7 | 82.1 | 81.3 |
| (b) first stage fine-tuned e5-small, no reranker | 80.2 | 87.3 | 55.1 | 80.4 | 75.7 |
| (b) + mMiniLM cross-encoder, off the shelf | 85.5 | 90.7 | 69.8 | 79.9 | 81.5 |
| (b) + fine-tuned reranker | 86.7 | 90.6 | 72.3 | 83.7 | 83.3 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 82.7 | 86.8 | 63.3 | 67.8 | 75.1 |
| (c) + mMiniLM cross-encoder, off the shelf | 86.7 | 91.3 | 70.9 | 80.1 | 82.2 |
| (c) + fine-tuned reranker | 89.0 | 91.3 | 75.3 | 84.4 | 85.0 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 81.7 | 86.5 | 55.1 | 80.4 | 75.9 |
| (d) + fine-tuned reranker | 88.4 | 90.9 | 72.3 | 83.7 | 83.8 |


### RERANK - TEST HIT@1 (x100); every reranker sees the first stage's top-50

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 68.6 | 81.7 | 31.7 | 51.4 | 58.3 |
| (a) + mMiniLM cross-encoder, off the shelf | 80.3 | 86.9 | 58.7 | 74.9 | 75.2 |
| (a) + fine-tuned reranker (mined hard negatives, + cross-lingual) | 82.8 | 86.8 | 62.1 | 78.8 | 77.6 |
| (a) + fine-tuned reranker, RANDOM negatives | 78.5 | 78.2 | 58.0 | 77.1 | 72.9 |
| (a) + fine-tuned reranker, no cross-lingual training data | 82.5 | 86.6 | 60.3 | 76.5 | 76.5 |
| (a) + fine-tuned reranker, rerank only top-10 | 82.1 | 87.8 | 54.6 | 74.8 | 74.8 |
| (a) + fine-tuned reranker, rerank only top-20 | 82.7 | 87.5 | 60.2 | 77.9 | 77.1 |
| (b) first stage fine-tuned e5-small, no reranker | 72.7 | 81.6 | 43.4 | 72.8 | 67.6 |
| (b) + mMiniLM cross-encoder, off the shelf | 80.5 | 86.6 | 61.8 | 73.4 | 75.6 |
| (b) + fine-tuned reranker | 82.3 | 86.4 | 64.2 | 77.7 | 77.7 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 75.4 | 81.0 | 52.4 | 59.2 | 67.0 |
| (c) + mMiniLM cross-encoder, off the shelf | 81.5 | 87.2 | 62.6 | 74.9 | 76.5 |
| (c) + fine-tuned reranker | 84.3 | 87.0 | 67.6 | 79.3 | 79.6 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 74.1 | 80.8 | 43.4 | 72.8 | 67.8 |
| (d) + fine-tuned reranker | 83.7 | 86.8 | 64.2 | 77.7 | 78.1 |


### RERANK - TEST HIT@10 (x100); every reranker sees the first stage's top-50

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 |
| (a) + mMiniLM cross-encoder, off the shelf | 92.9 | 97.4 | 78.2 | 90.0 | 89.6 |
| (a) + fine-tuned reranker (mined hard negatives, + cross-lingual) | 94.0 | 97.4 | 80.2 | 93.4 | 91.2 |
| (a) + fine-tuned reranker, RANDOM negatives | 94.4 | 98.3 | 79.9 | 93.0 | 91.4 |
| (a) + fine-tuned reranker, no cross-lingual training data | 93.8 | 97.5 | 79.7 | 92.8 | 90.9 |
| (a) + fine-tuned reranker, rerank only top-10 | 90.7 | 97.5 | 62.9 | 82.7 | 83.4 |
| (a) + fine-tuned reranker, rerank only top-20 | 92.7 | 97.7 | 72.2 | 88.7 | 87.9 |
| (b) first stage fine-tuned e5-small, no reranker | 92.1 | 96.9 | 79.3 | 93.9 | 90.6 |
| (b) + mMiniLM cross-encoder, off the shelf | 93.5 | 97.5 | 84.5 | 92.8 | 92.0 |
| (b) + fine-tuned reranker | 94.1 | 97.2 | 86.1 | 94.8 | 93.0 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 94.3 | 96.2 | 84.7 | 84.4 | 89.9 |
| (c) + mMiniLM cross-encoder, off the shelf | 95.0 | 98.1 | 86.4 | 90.0 | 92.4 |
| (c) + fine-tuned reranker | 96.6 | 97.9 | 89.0 | 92.9 | 94.1 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 94.2 | 96.4 | 79.3 | 93.9 | 91.0 |
| (d) + fine-tuned reranker | 96.3 | 97.4 | 86.1 | 94.8 | 93.6 |


### RERANK - TEST NDCG@10 (x100); every reranker sees the first stage's top-50

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 80.2 | 90.2 | 46.5 | 66.8 | 70.9 |
| (a) + mMiniLM cross-encoder, off the shelf | 87.2 | 92.5 | 68.8 | 82.4 | 82.7 |
| (a) + fine-tuned reranker (mined hard negatives, + cross-lingual) | 88.9 | 92.6 | 71.6 | 86.1 | 84.8 |
| (a) + fine-tuned reranker, RANDOM negatives | 87.6 | 89.0 | 69.5 | 85.5 | 82.9 |
| (a) + fine-tuned reranker, no cross-lingual training data | 88.5 | 92.3 | 70.3 | 84.8 | 84.0 |
| (a) + fine-tuned reranker, rerank only top-10 | 87.1 | 93.2 | 59.2 | 79.2 | 79.7 |
| (a) + fine-tuned reranker, rerank only top-20 | 88.4 | 93.1 | 66.5 | 83.7 | 82.9 |
| (b) first stage fine-tuned e5-small, no reranker | 83.1 | 89.7 | 60.9 | 83.7 | 79.4 |
| (b) + mMiniLM cross-encoder, off the shelf | 87.5 | 92.4 | 73.3 | 83.0 | 84.1 |
| (b) + fine-tuned reranker | 88.5 | 92.2 | 75.7 | 86.4 | 85.7 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 85.6 | 89.1 | 68.5 | 71.8 | 78.7 |
| (c) + mMiniLM cross-encoder, off the shelf | 88.7 | 93.0 | 74.7 | 82.5 | 84.7 |
| (c) + fine-tuned reranker | 90.9 | 92.9 | 78.6 | 86.5 | 87.2 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 84.8 | 88.9 | 60.9 | 83.7 | 79.6 |
| (d) + fine-tuned reranker | 90.4 | 92.5 | 75.7 | 86.4 | 86.2 |


### RERANK - candidate recall: Hit@50 of each first stage (x100) = ceiling for reranking top-50

| System | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| (a) first stage e5-small, no reranker | 95.3 | 99.4 | 83.1 | 94.1 | 93.0 |
| (b) first stage fine-tuned e5-small, no reranker | 96.4 | 99.1 | 91.3 | 98.4 | 96.3 |
| (c) first stage bge-m3 + BM25(stem), language-gated RRF | 98.1 | 99.8 | 93.8 | 94.0 | 96.4 |
| (d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF | 98.3 | 99.4 | 91.3 | 98.4 | 96.8 |


### RERANK - paired bootstrap on TEST MRR@10 [95% CI], points

| Comparison | ar->ar | en->en | ar->en | en->ar |
|---|---:|---:|---:|---:|
| (a) FT reranker vs off-the-shelf mMiniLM | +1.9 [+1.1, +2.6]* | +0.0 [-0.9, +1.0] | +3.1 [+1.8, +4.5]* | +3.8 [+2.6, +5.1]* |
| (a) FT reranker (mined negs) vs FT reranker (random negs) | +1.8 [+1.0, +2.7]* | +5.0 [+3.7, +6.4]* | +2.7 [+1.4, +4.0]* | +0.8 [-0.3, +1.9] |
| (a) FT reranker vs no reranker | +10.3 [+9.1, +11.5]* | +3.2 [+1.8, +4.6]* | +27.5 [+25.1, +29.9]* | +22.1 [+19.8, +24.6]* |
| (b) FT reranker behind FT embedder vs FT embedder alone | +6.5 [+5.2, +7.6]* | +3.3 [+1.8, +4.8]* | +17.2 [+14.8, +19.7]* | +3.3 [+1.1, +5.4]* |
| (b) vs (a): FT embedder + FT reranker vs baseline embedder + FT reranker | -0.5 [-1.0, +0.0] | -0.4 [-0.8, +0.0] | +3.5 [+1.8, +5.0]* | -0.1 [-1.2, +1.0] |


### RERANK - strong 568M reference reranker on a 1050-query subset, MRR@10 (x100)

| System (same 1050 test queries: 150 per slice) | ar->ar | en->en | ar->en | en->ar | macro |
|---|---:|---:|---:|---:|---:|
| baseline e5-small, no reranker | 77.6 | 87.4 | 41.7 | 61.3 | 67.0 |
| + mMiniLM cross-encoder (118M), off the shelf | 87.0 | 91.8 | 67.2 | 81.2 | 81.8 |
| + mMiniLM cross-encoder, fine-tuned (mined negatives) | 87.8 | 91.3 | 69.7 | 84.4 | 83.3 |
| + mMiniLM cross-encoder, fine-tuned (random negatives) | 87.0 | 87.0 | 66.5 | 84.4 | 81.2 |
| + bge-reranker-v2-m3 (568M), off the shelf | 86.5 | 93.7 | 75.1 | 79.3 | 83.7 |


### Per-query latency

| Component | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|
| mmarco-mMiniLM (off-the-shelf), cuda, top20 | 38.5 | 47.1 | 36.3 |
| mmarco-mMiniLM (off-the-shelf), cuda, top50 | 94.3 | 120.4 | 93.7 |
| mmarco-mMiniLM (off-the-shelf), cpu 1thr, top20 | 3066.6 | 3261.1 | 3068.8 |
| mmarco-mMiniLM (off-the-shelf), cpu 1thr, top50 | 7206.6 | 7406.1 | 7136.3 |
| mmarco-mMiniLM (off-the-shelf), cpu 4thr, top20 | 933.6 | 1280.2 | 986.8 |
| mmarco-mMiniLM (off-the-shelf), cpu 4thr, top50 | 3160.7 | 3459.6 | 3142.3 |
| mmarco-mMiniLM (off-the-shelf), cpu 8thr, top20 | 1242.1 | 1304.8 | 1247.5 |
| mmarco-mMiniLM (off-the-shelf), cpu 8thr, top50 | 3214.7 | 3491.3 | 3237.7 |
| bge-reranker-v2-m3 (off-the-shelf), cuda, top20 | 262.0 | 302.9 | 263.9 |
| bge-reranker-v2-m3 (off-the-shelf), cuda, top50 | 559.9 | 594.6 | 555.3 |
| bge-reranker-v2-m3 (off-the-shelf), cpu 8thr, top20 | 10737.3 | 12102.2 | 10871.8 |
| embed one query, e5-small, cuda | 16.0 | 19.5 | 15.9 |
| embed one query, e5-small, cpu 8thr | 17.2 | 33.8 | 20.4 |
| embed one query, bge-m3, cuda | 39.3 | 55.2 | 38.9 |
| embed one query, bge-m3, cpu 8thr | 172.1 | 218.0 | 167.8 |


### Sensitivity to near-duplicate leakage (test queries whose gold passage overlaps a training passage removed)

Excluded test queries (gold passage near-duplicated in train): 14 of 6823.

| System | macro MRR@10, all queries | macro MRR@10, excluding flagged | difference |
|---|---:|---:|---:|
| e5-small baseline | 66.92 | 66.93 | +0.01 |
| bge-m3 baseline | 74.78 | 74.77 | -0.00 |
| e5s_inb_e1_s13 | 70.69 | 70.71 | +0.01 |
| e5s_hard1_e1_s13 | 72.86 | 72.87 | +0.01 |
| e5s_hard1_xl_e1_s13 | 75.74 | 75.76 | +0.02 |


### Orthographic-noise robustness (ar->ar slices tydi_ar, arcd_ar, xquad_ar2ar; macro MRR@10 x100)

Fraction of Arabic test queries changed by the corruption: 0.634

| System | clean MRR@10 | corrupted-query MRR@10 | change |
|---|---:|---:|---:|
| bm25-raw | 67.4 | 61.8 | -5.7 |
| bm25-norm | 71.3 | 71.3 | +0.0 |
| bm25-stem | 76.1 | 76.1 | +0.0 |
| e5-small | 76.8 | 75.7 | -1.1 |
| bge-m3 | 80.2 | 79.6 | -0.5 |
| e5s_hard1_xl_e1_s13 | 80.2 | 79.2 | -1.0 |


### Mining analysis (mining_e5-small)

```json
{
 "model": "e5-small",
 "n_queries": 16419,
 "gold_rank_on_train_pool": {
  "top1": 0.7541872221207138,
  "top10": 0.957488275778062,
  "top30": 0.9785005176929168,
  "beyond_100": 0.01090200377611304
 },
 "score_gap_by_negative_rank": {
  "rank1": {
   "median": 0.0259,
   "p10": -0.0231,
   "p90": 0.0814,
   "frac_beating_gold": 0.246
  },
  "rank2": {
   "median": 0.04,
   "p10": -0.0061,
   "p90": 0.0922,
   "frac_beating_gold": 0.135
  },
  "rank3": {
   "median": 0.0466,
   "p10": 0.0004,
   "p90": 0.0978,
   "frac_beating_gold": 0.098
  },
  "rank5": {
   "median": 0.0539,
   "p10": 0.0076,
   "p90": 0.1047,
   "frac_beating_gold": 0.068
  },
  "rank10": {
   "median": 0.0634,
   "p10": 0.0168,
   "p90": 0.1133,
   "frac_beating_gold": 0.043
  },
  "rank20": {
   "median": 0.073,
   "p10": 0.0259,
   "p90": 0.1219,
   "frac_beating_gold": 0.028
  },
  "rank30": {
   "median": 0.0787,
   "p10": 0.0311,
   "p90": 0.1267,
   "frac_beating_gold": 0.021
  }
 },
 "filters(counts over 30 candidates per query)": {
  "no filter (ranks 1-30)": {
   "gold": 16240,
   "rank_window": 1149156,
   "margin": 0,
   "near_dup": 98,
   "same_title": 0,
   "kept": 49257,
   "short_lists": 0
  },
  "margin 0.00": {
   "gold": 16240,
   "rank_window": 1149156,
   "margin": 23639,
   "near_dup": 55,
   "same_title": 0,
   "kept": 48162,
   "short_lists": 372
  },
  "margin 0.02": {
   "gold": 16240,
   "rank_window": 1149156,
   "margin": 58403,
   "near_dup": 11,
   "same_title": 0,
   "kept": 46204,
   "short_lists": 1040
  },
  "margin 0.05": {
   "gold": 16240,
   "rank_window": 1149156,
   "margin": 165772,
   "near_dup": 0,
   "same_title": 0,
   "kept": 37783,
   "short_lists": 3891
  },
  "margin 0.02 + drop same article": {
   "gold": 16240,
   "rank_window": 1149156,
   "margin": 58403,
   "near_dup": 0,
   "same_title": 79699,
   "kept": 44734,
   "short_lists": 1754
  },
  "skip top-9 (ranks 10-30)": {
   "gold": 16240,
   "rank_window": 1281251,
   "margin": 0,
   "near_dup": 0,
   "same_title": 0,
   "kept": 49257,
   "short_lists": 0
  }
 },
 "judge": {
  "n_judged": 638,
  "gold_agreement(yes rate on labelled gold)": 0.912,
  "by_band": {
   "rank 1-3": {
    "n": 159,
    "yes_rate": 0.088
   },
   "rank 11-30": {
    "n": 160,
    "yes_rate": 0.006
   },
   "rank 4-10": {
    "n": 160,
    "yes_rate": 0.019
   }
  },
  "by_lang": {
   "ar": {
    "n": 240,
    "yes_rate": 0.042
   },
   "en": {
    "n": 239,
    "yes_rate": 0.033
   }
  },
  "by_score_gap(gold - candidate)": {
   "(-9.0, 0.0]": {
    "n": 31,
    "yes_rate": 0.194
   },
   "(0.0, 0.02]": {
    "n": 44,
    "yes_rate": 0.114
   },
   "(0.02, 0.05]": {
    "n": 127,
    "yes_rate": 0.047
   },
   "(0.05, 0.1]": {
    "n": 189,
    "yes_rate": 0.005
   },
   "(0.1, 9.0]": {
    "n": 88,
    "yes_rate": 0.0
   }
  }
 }
}
```


### Leakage audit

```json
{
  "gold_outside_pool": 0,
  "duplicate_pids_across_pools": 0,
  "questions_in_multiple_splits": 1,
  "question_split_pairs": {
    "dev+train": "1"
  },
  "title_overlap_all_passages(incl. distractors)": {
    "train&test": 10,
    "train&dev": 16,
    "dev&test": 0
  },
  "title_overlap_gold_passages_only": {
    "train&test": 0,
    "train&dev": 9,
    "dev&test": 0
  },
  "note_title_overlap": "Overlap with distractor-only articles comes from the per-source title hash (the same Wikipedia article can be a TyDi distractor in the test pool and an ARCD train article); identical paragraphs are removed by text key. Gold test articles are excluded from train/dev by construction (see build_data).",
  "near_duplicate_passages(6-gram containment)": {
    "test_vs_train": {
      "passages_checked": 9817,
      "containment>=0.5": 26,
      "containment>=0.8": 14,
      "gold_queries_affected": 14,
      "examples": [
        {
          "a": "ar-ae77a4b50516",
          "b": "ar-80de3bf9c6d3",
          "frac": 0.65
        },
        {
          "a": "ar-49ef52e929f3",
          "b": "ar-e9e040494650",
          "frac": 0.6
        },
        {
          "a": "en-5f1251d802d5",
          "b": "en-e4f96fe24c8f",
          "frac": 0.9
        }
      ]
    },
    "dev_vs_train": {
      "passages_checked": 4037,
      "containment>=0.5": 23,
      "containment>=0.8": 13,
      "gold_queries_affected": 50,
      "examples": [
        {
          "a": "ar-0adca5c65f83",
          "b": "ar-38cd2f0ce34f",
          "frac": 0.93
        },
        {
          "a": "ar-dc45f2a62f0a",
          "b": "ar-025454a9571d",
          "frac": 0.64
        },
        {
          "a": "ar-4b65214810c6",
          "b": "ar-9277e582e03e",
          "frac": 0.81
        }
      ]
    },
    "test_vs_dev": {
      "passages_checked": 9817,
      "containment>=0.5": 6,
      "containment>=0.8": 3,
      "gold_queries_affected": 6,
      "examples": [
        {
          "a": "ar-aca2aa84a564",
          "b": "ar-6db3ff04802e",
          "frac": 0.7
        },
        {
          "a": "ar-5267b6445f78",
          "b": "ar-6db3ff04802e",
          "frac": 0.71
        },
        {
          "a": "ar-98c25479c247",
          "b": "ar-3d280a0c7715",
          "frac": 0.55
        }
      ]
    }
  },
  "near_duplicate_questions": {
    "test": {
      "queries_checked": 6823,
      "near_dup_of_a_train_question(J>=0.8)": 8,
      "examples": [
        [
          "ما هو الحمض النووي الريبوزي الصغير النووي ؟",
          "ما هو الحمض النووي الريبوزي؟"
        ],
        [
          "متى تأسست حركة حماس؟",
          "متى تأسست حركة حماس العراق؟"
        ],
        [
          "Where was Zheng He born?",
          "Where was he born?"
        ]
      ]
    },
    "dev": {
      "queries_checked": 3163,
      "near_dup_of_a_train_question(J>=0.8)": 9,
      "examples": [
        [
          "من هو جون ستيوارت ؟",
          "من هو جون ستيوارت مل؟"
        ],
        [
          "ماهي أعلى رتبة عسكرية في سلم الرتب العسكرية الإيراني؟",
          "ماهي أعلى رتبة عسكرية في سلم الرتب العسكرية المصري؟"
        ],
        [
          "ما هو الحمض النووي المؤتلف؟",
          "ما هو الحمض النووي؟"
        ]
      ]
    }
  },
  "translated_queries_with_foreign_source_split": 0,
  "translated_queries": {
    "dev": 500,
    "train": 4499
  }
}
```


### Data manifest

```json
{
 "seed": 13,
 "max_q_per_ctx": {
  "squad_train": 2
 },
 "licenses": {
  "tydi": "Apache-2.0 (google-research-datasets/tydiqa, secondary_task = GoldP)",
  "arcd": "MIT (hsseinmz/arcd)",
  "squad": "CC BY-SA 4.0 (rajpurkar/squad v1.1)",
  "xquad": "CC BY-SA 4.0 (google/xquad)"
 },
 "dropped": {
  "title_in_test_tydi_ar_train": 1094,
  "title_in_test_tydi_en_train": 6,
  "title_in_test_arcd_train": 306,
  "passage_in_other_pool_arcd_train": 12,
  "title_in_test_squad_train": 1610,
  "passage_in_other_pool_tydi_ar_train": 4,
  "dup_question_train_tydi_ar": 6,
  "dup_question_train_arcd_ar": 3,
  "dup_question_train_squad_en": 1
 },
 "passages_by_pool_lang": {
  "dev/ar": 1982,
  "dev/en": 2055,
  "test/ar": 4325,
  "test/en": 5492,
  "train/ar": 5425,
  "train/en": 5227
 },
 "queries_by_slice": {
  "dev/dev_arcd_ar": 111,
  "dev/dev_squad_en": 1661,
  "dev/dev_tydi_ar": 1037,
  "dev/dev_tydi_en": 354,
  "dev/dev_xl_ar2en": 300,
  "dev/dev_xl_en2ar": 200,
  "test/arcd_ar": 702,
  "test/tydi_ar": 921,
  "test/tydi_en": 440,
  "test/xquad_ar2ar": 1190,
  "test/xquad_ar2en": 1190,
  "test/xquad_en2ar": 1190,
  "test/xquad_en2en": 1190,
  "train/train_arcd_ar": 414,
  "train/train_squad_en": 6653,
  "train/train_tydi_ar": 7328,
  "train/train_tydi_en": 2024,
  "train/train_xl_ar2en": 3000,
  "train/train_xl_en2ar": 1499
 },
 "xling_candidates": {
  "dev/ar2en": 200,
  "dev/en2ar": 300,
  "train/ar2en": 1500,
  "train/en2ar": 3000
 },
 "translation_stats": {
  "candidates": 5000,
  "missing": 0,
  "invalid": 0,
  "dropped_duplicate_of_higher_priority_split": 1
 },
 "queries_by_split": {
  "dev": 3663,
  "test": 6823,
  "train": 20918
 },
 "passages_total": 24506,
 "passages_by_pool_kind": {
  "dev/background": 2054,
  "dev/gold": 1983,
  "test/background": 7834,
  "test/gold": 1983,
  "train/background": 1,
  "train/gold": 10651
 }
}
```
