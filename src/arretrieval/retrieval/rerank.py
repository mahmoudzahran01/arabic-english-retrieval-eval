"""Cross-encoder reranking: score (query, passage) pairs jointly, re-order a first-stage candidate list."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Reranker:
    def __init__(self, path: str, max_len: int = 256, bs: int = 128, device: str = "cuda", fp16: bool = True):
        import torch
        from sentence_transformers.cross_encoder import CrossEncoder

        kw = {"torch_dtype": torch.float16} if (fp16 and device == "cuda") else {}
        self.model = CrossEncoder(path, max_length=max_len, device=device, model_kwargs=kw)
        self.bs = bs
        self.device = device

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(
            self.model.predict(pairs, batch_size=self.bs, show_progress_bar=False, convert_to_numpy=True), dtype=np.float32
        ).reshape(-1)


def rerank_slice(rr: Reranker, q: pd.DataFrame, passages_text: dict[str, str], run: dict, top_n: int, top_keep: int = 100) -> dict:
    """run[qid] = (pids, scores) from the first stage. Returns the same structure, re-ordered inside the top_n prefix;
    candidates beyond top_n keep their first-stage order after the reranked prefix."""
    pairs, owners = [], []
    for qid, text in zip(q.qid, q.text):
        for pid in run[qid][0][:top_n]:
            pairs.append((text, passages_text[pid]))
            owners.append((qid, pid))
    scores = rr.score(pairs) if pairs else np.zeros(0)
    by_q: dict[str, list[tuple[str, float]]] = {}
    for (qid, pid), s in zip(owners, scores):
        by_q.setdefault(qid, []).append((pid, float(s)))
    out = {}
    for qid in q.qid:
        ranked = sorted(by_q.get(qid, []), key=lambda t: -t[1])
        pids = [p for p, _ in ranked] + list(run[qid][0][top_n:])
        out[qid] = (pids[:top_keep], [s for _, s in ranked] + [0.0] * (len(pids[:top_keep]) - len(ranked)))
    return out
