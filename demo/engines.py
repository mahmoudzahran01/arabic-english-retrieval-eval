"""Query/passage encoders used by the demo (CPU only, no sentence-transformers needed).

OnnxEmbedder  the v2 model: ONNX Runtime graph with pooling + L2 normalisation inside (see deploy/export_onnx.py)
HFEmbedder    baselines (bge-m3, v1 e5-small): transformers + torch, CLS or mean pooling, L2-normalised

Both return float32 arrays of unit vectors, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

E5_PREFIXES = ("query: ", "passage: ")
NO_PREFIXES = ("", "")


def _batches(texts: list[str], bs: int):
    order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths together -> little padding
    for i in range(0, len(texts), bs):
        yield order[i : i + bs]


class OnnxEmbedder:
    def __init__(self, root: str | Path, variant: str = "fp32", prefixes=E5_PREFIXES, max_len: int = 256, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(root)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(root / variant / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        pad_id = int(json.loads((root / "config.json").read_text()).get("pad_token_id", 1))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=pad_id, pad_token=self.tok.id_to_token(pad_id) or "<pad>")
        self.q_prefix, self.p_prefix = prefixes

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
        return self.sess.run(None, {"input_ids": ids, "attention_mask": mask})[0].astype(np.float32)

    def encode(self, texts: list[str], kind: str = "q", bs: int = 16) -> np.ndarray:
        pre = self.q_prefix if kind == "q" else self.p_prefix
        texts = [pre + t for t in texts]
        out = None
        for idx in _batches(texts, bs):
            emb = self._embed([texts[j] for j in idx])
            if out is None:
                out = np.empty((len(texts), emb.shape[1]), dtype=np.float32)
            out[idx] = emb
        return out if out is not None else np.zeros((0, 0), dtype=np.float32)


class HFEmbedder:
    def __init__(self, model_id: str, pooling: str, prefixes=NO_PREFIXES, max_len: int = 256, threads: int = 2):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.pooling, self.max_len = pooling, max_len
        self.q_prefix, self.p_prefix = prefixes

    def _embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        b = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        with torch.inference_mode():
            h = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            if self.pooling == "cls":
                e = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            e = torch.nn.functional.normalize(e, dim=-1)
        return e.numpy().astype(np.float32)

    def encode(self, texts: list[str], kind: str = "q", bs: int = 8) -> np.ndarray:
        pre = self.q_prefix if kind == "q" else self.p_prefix
        texts = [pre + t for t in texts]
        out = None
        for idx in _batches(texts, bs):
            emb = self._embed([texts[j] for j in idx])
            if out is None:
                out = np.empty((len(texts), emb.shape[1]), dtype=np.float32)
            out[idx] = emb
        return out if out is not None else np.zeros((0, 0), dtype=np.float32)


def top_k(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Indices and cosine scores of the k best corpus rows; `allowed` is an optional boolean mask (language filter).
    `corpus_emb` should already be float32 (converted once at load time)."""
    scores = corpus_emb @ query_emb.astype(np.float32)
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    idx = np.argpartition(-scores, k - 1)[:k] if k > 0 else np.array([], dtype=int)
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]
