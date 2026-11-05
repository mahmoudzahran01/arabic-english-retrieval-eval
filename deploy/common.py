"""Shared helpers for the deployment benchmark: encoders (PyTorch / ONNX Runtime), the test-set evaluation, memory gate.

Everything here is CPU only. Both encoders return L2-normalised float32 embeddings so that the retrieval code is
identical for every variant. Imports of torch / onnxruntime are lazy, so the pure helpers can be unit-tested without them.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

MAX_LEN = 512  # the length used by the published evaluation (arretrieval.retrieval.evalkit.Dense)
E5_PREFIXES = {"e5": ("query: ", "passage: "), "none": ("", "")}


# ------------------------------------------------------------------------------------------------ resources
def mem_available_gb(meminfo: str = "/proc/meminfo") -> float | None:
    """MemAvailable in GiB (Linux), None where /proc/meminfo does not exist (macOS)."""
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


MIN_MEM_GB = float(os.environ.get("V2_MIN_MEM_GB", "25"))  # the Spark rule: MemAvailable >= 25 GB before each job


def wait_for_memory(min_gb: float = MIN_MEM_GB, poll_s: int = 30, timeout_s: int = 6 * 3600) -> None:
    """Block until MemAvailable >= min_gb (the Spark is shared with a training job; see deploy/README.md)."""
    t0 = time.time()
    while True:
        avail = mem_available_gb()
        if avail is None or avail >= min_gb:
            return
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"MemAvailable stayed below {min_gb} GiB for {timeout_s} s (last {avail:.1f})")
        print(f"[memgate] MemAvailable {avail:.1f} GiB < {min_gb}; waiting", file=sys.stderr, flush=True)
        time.sleep(poll_s)


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (ru_maxrss is KiB on Linux, bytes on macOS)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def set_threads(n: int) -> None:
    """Pin the math libraries to n threads. Call before importing torch / onnxruntime."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = str(n)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def dir_size_mb(path: str | Path) -> float:
    """Size of a file, or the summed size of a directory's files, in MiB (1 MiB = 2**20 bytes)."""
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 2**20


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


# ------------------------------------------------------------------------------------------------ encoders
class TorchEncoder:
    """Reference: the sentence-transformers model in float32 on CPU."""

    def __init__(self, path: str, prefix: str = "e5", threads: int = 8, max_len: int = MAX_LEN, bs: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.q_prefix, self.p_prefix = E5_PREFIXES[prefix]
        self.bs = bs
        self.model = SentenceTransformer(path, device="cpu")
        self.model.max_seq_length = max_len

    def encode(self, texts: list[str], kind: str, bs: int | None = None) -> np.ndarray:
        pre = self.q_prefix if kind == "q" else self.p_prefix
        out = self.model.encode(
            [pre + t for t in texts], batch_size=bs or self.bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return out.astype(np.float32)


class OnnxEncoder:
    """ONNX Runtime session over a graph that already contains pooling and L2 normalisation (see export_onnx.py)."""

    def __init__(self, onnx_path: str, tok_dir: str, prefix: str = "e5", threads: int = 8, max_len: int = MAX_LEN, bs: int = 32):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.q_prefix, self.p_prefix = E5_PREFIXES[prefix]
        self.bs = bs
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
        pad_id = int(json.loads((Path(tok_dir) / "config.json").read_text()).get("pad_token_id", 1))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=pad_id, pad_token=self.tok.id_to_token(pad_id) or "<pad>")

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
        return self.sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    def encode(self, texts: list[str], kind: str, bs: int | None = None) -> np.ndarray:
        pre = self.q_prefix if kind == "q" else self.p_prefix
        texts = [pre + t for t in texts]
        bs = bs or self.bs
        order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths per batch -> little padding
        out = None
        for i in range(0, len(texts), bs):
            idx = order[i : i + bs]
            emb = self.embed_batch([texts[j] for j in idx]).astype(np.float32)
            if out is None:
                out = np.empty((len(texts), emb.shape[1]), dtype=np.float32)
            out[idx] = emb
        return out if out is not None else np.zeros((0, 0), dtype=np.float32)


def load_encoder(variant: str, model: str, onnx_root: str | None, prefix: str, threads: int, bs: int = 32):
    """variant: 'torch-fp32' (model = sentence-transformers folder) or 'onnx-<name>' (onnx_root/<name>/model.onnx)."""
    if variant == "torch-fp32":
        return TorchEncoder(model, prefix, threads, bs=bs)
    if variant.startswith("onnx-"):
        root = Path(onnx_root or ".")
        return OnnxEncoder(str(root / variant[5:] / "model.onnx"), str(root), prefix, threads, bs=bs)
    raise ValueError(f"unknown variant {variant!r}")


def variant_size_mb(variant: str, model: str, onnx_root: str | None) -> float:
    if variant == "torch-fp32":
        return dir_size_mb(Path(model) / "model.safetensors")
    return dir_size_mb(Path(onnx_root or ".") / variant[5:])
