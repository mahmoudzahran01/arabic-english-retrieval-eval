"""Latency / throughput / peak-memory benchmark of one deployment variant on CPU (one process per variant => clean peak RSS).

Workloads (texts sampled from the published TEST split, fixed seed):
  query    batch 1 and batch 32   (short search queries, Arabic and English)
  passage  batch 1 and batch 32   (passages truncated to 512 tokens, as in the evaluation)
Latency is end-to-end per call: tokenisation + model + pooling. Throughput = sentences / wall time of the timed calls.

    python -m deploy.bench --name v2-small --variant onnx-int8 --model MODEL_DIR --onnx-root ARTIFACTS/v2-small --threads 8 --out results/bench
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .common import dir_size_mb, load_encoder, peak_rss_mb, percentile, set_threads, variant_size_mb, wait_for_memory

WORKLOADS = {  # name: (kind, batch size, timed calls, warm-up calls)
    "query_b1": ("q", 1, 200, 20),
    "query_b32": ("q", 32, 20, 3),
    "passage_b1": ("p", 1, 60, 5),
    "passage_b32": ("p", 32, 8, 2),
}


def sample_texts(data_dir: Path, n_q: int = 800, n_p: int = 512, seed: int = 13) -> tuple[list[str], list[str]]:
    import pandas as pd

    rng = np.random.default_rng(seed)
    q = pd.read_parquet(data_dir / "queries.parquet")
    p = pd.read_parquet(data_dir / "passages.parquet")
    q = q[q.split == "test"].drop_duplicates("text")
    p = p[p.pool == "test"]
    qs = q.text.to_numpy()[rng.permutation(len(q))[:n_q]]
    ps = p.text.to_numpy()[rng.permutation(len(p))[:n_p]]
    return list(qs), list(ps)


def mean_tokens(tok_dir: Path, texts: list[str], max_len: int = 512) -> float:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=max_len)
    return float(np.mean([len(e.ids) for e in tok.encode_batch(texts)]))


def run_workload(enc, texts: list[str], kind: str, bs: int, calls: int, warm: int) -> dict:
    batches = [texts[(i * bs) % len(texts) :][:bs] for i in range(calls + warm)]
    batches = [b if len(b) == bs else (b + texts)[:bs] for b in batches]
    for b in batches[:warm]:
        enc.encode(b, kind, bs=bs)
    lat = []
    t_all = time.perf_counter()
    for b in batches[warm:]:
        t0 = time.perf_counter()
        enc.encode(b, kind, bs=bs)
        lat.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - t_all
    return {
        "calls": calls,
        "batch": bs,
        "p50_ms": round(percentile(lat, 50), 2),
        "p95_ms": round(percentile(lat, 95), 2),
        "mean_ms": round(float(np.mean(lat)), 2),
        "throughput_sent_per_s": round(calls * bs / wall, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx-root", default=None)
    ap.add_argument("--prefix", default="e5", choices=["e5", "none"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--data-dir", default=os.environ.get("ARR_RUNS_DIR", "runs") + "/retrieval/data")
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-mem-gb", type=float, default=15.0)
    args = ap.parse_args()
    set_threads(args.threads)
    wait_for_memory(args.min_mem_gb)

    qs, ps = sample_texts(Path(args.data_dir))
    tok_dir = Path(args.onnx_root) if args.onnx_root else Path(args.model)
    load0 = os.getloadavg()[0]
    t0 = time.perf_counter()
    enc = load_encoder(args.variant, args.model, args.onnx_root, args.prefix, args.threads)
    load_s = time.perf_counter() - t0
    rss_after_load = peak_rss_mb()  # includes the python + pandas + data baseline of this process
    res = {}
    for name in args.workloads:
        kind, bs, calls, warm = WORKLOADS[name]
        res[name] = run_workload(enc, qs if kind == "q" else ps, kind, bs, calls, warm)
        print(f"[{args.name} {args.variant} t{args.threads}] {name}: {res[name]}", flush=True)
    payload = {
        "model": args.name,
        "variant": args.variant,
        "threads": args.threads,
        "platform": f"{platform.machine()} {platform.system()} ({os.cpu_count()} logical CPUs)",
        "loadavg_start": round(load0, 2),
        "loadavg_end": round(os.getloadavg()[0], 2),
        "size_mb": round(variant_size_mb(args.variant, args.model, args.onnx_root), 1),
        "model_dir_mb": round(dir_size_mb(Path(args.model) / "model.safetensors"), 1) if Path(args.model).exists() else None,
        "load_seconds": round(load_s, 2),
        "rss_after_load_mb": round(rss_after_load, 0),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "mean_tokens": {"query": round(mean_tokens(tok_dir, qs), 1), "passage": round(mean_tokens(tok_dir, ps), 1)},
        "workloads": res,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}__{args.variant}__t{args.threads}.bench.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
