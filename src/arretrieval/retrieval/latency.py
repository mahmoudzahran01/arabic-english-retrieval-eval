"""Latency of the query-time components, per query, on GPU and CPU.

  reranker:  score N candidate passages for ONE query (one batch of N pairs), tokenization included
  embedder:  encode ONE query (what a dense first stage pays per request; passages are pre-embedded)

Warm-up first, then the median / p95 over `--n-queries` real test queries.  The GPU/CPU were shared with other
jobs when these numbers were taken, so absolute numbers are noisy upper-ish bounds; ratios between models are the meaningful part.

Run: python -m arretrieval.retrieval.latency --first-stage e5-small
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import evalkit as E
from .paths import RESULTS

RERANKERS = {
    "mmarco-mMiniLM (off-the-shelf)": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "bge-reranker-v2-m3 (off-the-shelf)": "BAAI/bge-reranker-v2-m3",
}


def stats(ts: list[float]) -> dict:
    a = np.array(ts) * 1000
    return {"p50_ms": round(float(np.median(a)), 1), "p95_ms": round(float(np.quantile(a, 0.95)), 1), "mean_ms": round(float(a.mean()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-stage", default="e5-small")
    ap.add_argument("--n-queries", type=int, default=60, help="queries timed on GPU")
    ap.add_argument("--n-queries-cpu", type=int, default=12, help="queries timed on CPU (one query = seconds there)")
    ap.add_argument("--top-n", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--extra-reranker", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    args = ap.parse_args()
    import torch

    from .rerank import Reranker

    passages, queries = E.load_passages(), E.load_queries()
    text = dict(zip(passages.pid, passages.text))
    runs = E.load_runs(args.first_stage, "test")["xquad_en2en"]
    sample = queries[queries.slice == "xquad_en2en"].sample(args.n_queries, random_state=13)
    out: dict = {
        "first_stage": args.first_stage,
        "n_queries_gpu": args.n_queries,
        "n_queries_cpu": args.n_queries_cpu,
        "note": "shared machine; see RESULTS.md",
        "cpu_cores_visible": __import__("os").cpu_count(),
    }
    rerankers = {**RERANKERS, **{l: p for l, p in args.extra_reranker}}
    for label, path in rerankers.items():
        big = "bge-reranker" in label  # 568M: CPU only at 8 threads / top-20, otherwise it takes hours
        for device in ("cuda", "cpu"):
            for threads in [None] if device == "cuda" else ([8] if big else args.threads):
                if threads:
                    torch.set_num_threads(threads)
                rr = Reranker(path, max_len=256, bs=128 if not big else 32, device=device)
                for top_n in [20] if (big and device == "cpu") else args.top_n:
                    nq = args.n_queries if device == "cuda" else args.n_queries_cpu
                    pairs = [[(q, text[p]) for p in runs[qid][0][:top_n]] for qid, q in list(zip(sample.qid, sample.text))[:nq]]
                    for ps in pairs[:3]:
                        rr.score(ps)  # warm-up
                    ts = []
                    for ps in pairs:
                        if device == "cuda":
                            torch.cuda.synchronize()
                        t = time.perf_counter()
                        rr.score(ps)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t)
                    key = f"{label} | {device}{'' if threads is None else f' {threads}thr'} | top{top_n}"
                    out[key] = stats(ts)
                    print(key, out[key], flush=True)
                del rr
                torch.cuda.empty_cache()
    from sentence_transformers import SentenceTransformer

    for label, path in (("e5-small", "intfloat/multilingual-e5-small"), ("bge-m3", "BAAI/bge-m3")):
        for device in ("cuda", "cpu"):
            if device == "cpu":
                torch.set_num_threads(8)
            m = SentenceTransformer(path, device=device)
            qs = ["query: " + q if label == "e5-small" else q for q in sample.text][
                : (args.n_queries if device == "cuda" else args.n_queries_cpu)
            ]
            for q in qs[:3]:
                m.encode([q])
            ts = []
            for q in qs:
                t = time.perf_counter()
                m.encode([q])
                ts.append(time.perf_counter() - t)
            key = f"embed one query | {label} | {device}{' 8thr' if device == 'cpu' else ''}"
            out[key] = stats(ts)
            print(key, out[key], flush=True)
            del m
    (RESULTS / "latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
