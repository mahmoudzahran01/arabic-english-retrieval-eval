"""Retrieval quality of one deployment variant: MRR@10 on the published TEST split, same protocol as the repo's evaluation.

Every slice is searched in the passages of the split's own pool in the passage language (arretrieval.retrieval.evalkit).
Corpus embeddings are computed once per (pool, language) and reused by all slices.

    python -m deploy.eval_quality --name v2-small --variant onnx-int8 --model MODEL_DIR --onnx-root ARTIFACTS/v2-small --out results/quality

`--corpus-variant onnx-fp32` encodes the corpus with that variant and the queries with --variant (the "int8 queries against an fp32 index" setup used by
the demo). The reference for all deltas is `--variant torch-fp32`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .common import load_encoder, peak_rss_mb, set_threads, wait_for_memory


class SplitEncoder:
    """Queries with one encoder, passages with another (identical object twice for the symmetric case)."""

    def __init__(self, q_enc, p_enc):
        self.q_enc, self.p_enc = q_enc, p_enc

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        return (self.q_enc if kind == "q" else self.p_enc).encode(texts, kind)


class CachedRetriever:
    """Dense retriever with the E.run_system interface; corpus embeddings are cached per pool/language (in memory, and on disk
    under `disk_cache/<tag>__<hash of the passage ids>.npy` so that a later run with the same corpus encoder can reuse them)."""

    def __init__(self, enc, E, disk_cache: Path | None = None, tag: str = ""):
        self.enc, self.E, self.cache, self.corpus_seconds, self.query_seconds = enc, E, {}, 0.0, 0.0
        self.disk_cache, self.tag = disk_cache, tag

    def _disk_path(self, c) -> Path | None:
        if self.disk_cache is None:
            return None
        h = hashlib.sha1("\n".join(c.pid).encode()).hexdigest()[:16]
        return self.disk_cache / f"{self.tag}__{h}.npy"

    def rank(self, q, c, top: int = 100):
        key = (c.pid.iloc[0], len(c), c.pid.iloc[-1])
        if key not in self.cache:
            path = self._disk_path(c)
            if path is not None and path.exists():
                self.cache[key] = np.load(path)
            else:
                t0 = time.time()
                self.cache[key] = self.enc.encode(list(c.text), "p")
                self.corpus_seconds += time.time() - t0
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, self.cache[key])
        t0 = time.time()
        qe = self.enc.encode(list(q.text), "q")
        self.query_seconds += time.time() - t0
        scores = qe @ self.cache[key].T
        return dict(zip(q.qid, self.E._topk_from_scores(scores, c.pid.to_numpy(), top)))


def evaluate(enc, split: str = "test", disk_cache: Path | None = None, tag: str = "") -> dict:
    from arretrieval.retrieval import evalkit as E

    passages, queries = E.load_passages(), E.load_queries()
    ret = CachedRetriever(enc, E, disk_cache, tag)
    runs = E.run_system(ret, queries, passages, split)
    res = E.evaluate_runs(runs, queries, split)
    res["corpus_seconds"], res["query_seconds"] = ret.corpus_seconds, ret.query_seconds
    res["n_corpus_passages"] = int(sum(len(v) for v in ret.cache.values()))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="model label, e.g. v2-small")
    ap.add_argument("--variant", required=True, help="torch-fp32 | onnx-fp32 | onnx-int8 | onnx-int8-mm")
    ap.add_argument("--corpus-variant", default=None, help="encode the corpus with this variant instead (asymmetric setup)")
    ap.add_argument("--model", required=True, help="sentence-transformers folder")
    ap.add_argument("--onnx-root", default=None)
    ap.add_argument("--prefix", default="e5", choices=["e5", "none"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-mem-gb", type=float, default=15.0)
    args = ap.parse_args()
    set_threads(args.threads)
    wait_for_memory(args.min_mem_gb)

    q_enc = load_encoder(args.variant, args.model, args.onnx_root, args.prefix, args.threads, args.bs)
    p_enc = (
        q_enc
        if args.corpus_variant is None
        else load_encoder(args.corpus_variant, args.model, args.onnx_root, args.prefix, args.threads, args.bs)
    )
    t0 = time.time()
    corpus_label = args.corpus_variant or args.variant
    res = evaluate(SplitEncoder(q_enc, p_enc), args.split, Path(args.out).parent / "emb", f"{args.name}__{corpus_label}")
    label = args.variant if args.corpus_variant is None else f"{args.variant}+q-only"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}__{label}.{args.split}"
    payload = {
        "model": args.name,
        "variant": label,
        "split": args.split,
        "threads": args.threads,
        "slices": res["slices"],
        "directions": res["directions"],
        "corpus_seconds": round(res["corpus_seconds"], 1),
        "query_seconds": round(res["query_seconds"], 1),
        "n_corpus_passages": res["n_corpus_passages"],
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=1))
    np.savez_compressed(out / f"{stem}.perq.npz", **{sl: d["mrr@10"].astype(np.float32) for sl, d in res["perq"].items()})
    d = res["directions"]
    print(
        f"[{args.name} {label}] " + " ".join(f"{k}={d[k]['mrr@10'] * 100:.2f}" for k in d), f"| wall {payload['wall_seconds']}s", flush=True
    )


if __name__ == "__main__":
    main()
