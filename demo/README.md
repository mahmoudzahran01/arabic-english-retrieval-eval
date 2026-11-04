---
title: Arabic-English retrieval, v2 demo
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Arabic and English search over a mixed corpus, v2 vs bge-m3, on CPU
---

# Arabic-English retrieval: v2 demo

Type a question in **Arabic or English**; the Space shows the top-k passages of a small mixed Arabic + English corpus, with the
language of every passage marked, for the **v2** model next to a baseline (**bge-m3**, or the v1 e5-small fine-tune).
Everything runs on the CPU of a free Space (2 vCPU): corpus embeddings are precomputed offline, so a query costs one
forward pass of a short text.

| | what runs at query time |
|---|---|
| v2 | ONNX fp32 graph of **v2-base** (e5-base distilled from the fine-tuned bge-m3), ONNX Runtime, no PyTorch needed for this path |
| bge-m3 | `BAAI/bge-m3` with transformers + PyTorch (CPU), loaded on first use; CLS pooling, no prefix |
| v1 | `mzahran1/arabic-english-e5-small-retrieval`, mean pooling, `query: ` prefix |

## Read this as a demo, not a benchmark

* The corpus is **4,000 paragraphs of the training pool** (2,000 Arabic, 2,000 English) of the benchmark in this repository
  (TyDi QA GoldP, SQuAD v1.1 train, ARCD train). The v2 models, and v1, saw many of these paragraphs during fine-tuning, so the side-by-side view
  flatters them. The held-out comparison is in the technical report (`report/`) and `docs/retrieval/RESULTS.md`.
* The corpus is small: a question about a topic that is not in it returns the nearest unrelated paragraphs.
* v2 queries use the **fp32** ONNX graph (1,059 MiB), i.e. exactly the model that was evaluated: it is identical to PyTorch fp32 on the test split.
  Compressed graphs were measured and not used: dynamic int8 costs several macro points on this model, and the weight-only 8-bit graph (`w8`, 270 MiB) is lossless
  but several times slower per query on a CPU (see `deploy/README.md`). Set `V2_ONNX_VARIANT=w8` and ship `models/v2-base/w8/` instead if download size matters more than latency.

## Files

| file | role |
|---|---|
| `app.py` | Gradio UI (side by side, language badges, language filter, examples) |
| `engines.py` | ONNX and transformers query encoders, top-k |
| `build_index.py` | offline: pick the corpus, embed it with v2 / bge-m3 / v1 |
| `smoke_test.py`, `smoke_test.md` | 5 Arabic + 5 English queries through the same code path, and the output of the last run |
| `requirements.txt` | pinned to the versions the smoke test ran with |
| `index/` (not in git) | `passages.parquet`, `examples.json`, `meta.json`, `emb_v2.npy`, `emb_bge-m3.npy`, `emb_v1.npy` (fp16) |
| `models/v2-base/` (not in git) | `fp32/model.onnx` (about 1.1 GB; `w8/model.onnx`, about 270 MB, is the smaller alternative), `tokenizer.json`, `config.json` |

`index/` and `models/` are generated or exported artefacts and are **not committed to this repository**; they are uploaded together with the Space
(the ONNX file through Git LFS, as every Space file above 10 MB).

## Build and run

```bash
# 1. export the model (deploy/): ONNX fp32 graph (embeds the corpus and answers the queries)
python -m deploy.export_onnx --model <v2-base folder> --out demo/models/v2-base --variants fp32

# 2. corpus + embeddings (CPU; bge-m3 is the slow one: it dominates the build time)
python -m demo.build_index --out demo/index --data-dir runs/retrieval/data --select
python -m demo.build_index --out demo/index --data-dir runs/retrieval/data --model v2 --onnx-root demo/models/v2-base --variant fp32
python -m demo.build_index --out demo/index --data-dir runs/retrieval/data --model v1 --hf mzahran1/arabic-english-e5-small-retrieval
python -m demo.build_index --out demo/index --data-dir runs/retrieval/data --model bge-m3 --hf BAAI/bge-m3

# 3. run it
pip install -r demo/requirements.txt
python demo/app.py            # http://localhost:7860
python demo/smoke_test.py     # writes demo/smoke_test.md
```

`runs/retrieval/data` is produced by the data stage of this repository (`docs/retrieval/REPRODUCING.md`).

## Publishing (not done yet)

Create a Gradio Space, copy this folder's `app.py`, `engines.py`, `requirements.txt`, `README.md`, `index/` and `models/v2-base/fp32/` + `tokenizer.json` + `config.json` into it.
Once the v2 weights are on the Hub, `V2_ONNX_DIR` can point at a downloaded snapshot instead of the bundled folder.
The first query of a session loads bge-m3 (2.2 GB) and takes a while on a cold Space.

## Licences and attribution

Code: MIT (this repository). Corpus text: Wikipedia paragraphs distributed with TyDi QA (Apache-2.0), SQuAD v1.1 (CC BY-SA 4.0) and ARCD (MIT);
the paragraph text itself is Wikipedia content (CC BY-SA), so the `index/passages.parquet` file inherits CC BY-SA.
bge-m3 is MIT licensed (BAAI).
