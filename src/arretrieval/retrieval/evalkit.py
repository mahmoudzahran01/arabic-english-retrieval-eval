"""Shared evaluation harness: data access, dense / BM25 retrievers, per-slice + per-direction metrics.

A "slice" is one (query source, query language, passage language) combination. Every slice is evaluated against
the passages of the *split's own pool in the passage language*, e.g. test/xquad_ar2en = Arabic XQuAD questions
searched in the English test-pool passages (5.5k passages), gold = the parallel English paragraph.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .paths import CACHE, DATA, RESULTS
from .textnorm import normalize_ar, tokenize

TOP = 100  # depth of stored rankings

# slice -> (retrieval direction label). Direction = query language -> passage language.
DIRECTION_OF = {
    "tydi_ar": "ar->ar",
    "arcd_ar": "ar->ar",
    "xquad_ar2ar": "ar->ar",
    "tydi_en": "en->en",
    "xquad_en2en": "en->en",
    "xquad_ar2en": "ar->en",
    "xquad_en2ar": "en->ar",
    "dev_tydi_ar": "ar->ar",
    "dev_arcd_ar": "ar->ar",
    "dev_tydi_en": "en->en",
    "dev_squad_en": "en->en",
    "dev_xl_ar2en": "ar->en",
    "dev_xl_en2ar": "en->ar",
}
DIRECTIONS = ("ar->ar", "en->en", "ar->en", "en->ar")


# ----------------------------------------------------------------------------------------------- data
def load_passages() -> pd.DataFrame:
    return pd.read_parquet(DATA / "passages.parquet")


def load_queries(mono_only: bool = False) -> pd.DataFrame:
    """The final query table (monolingual + translated cross-lingual); `mono_only` loads the pre-translation table."""
    final = DATA / "queries.parquet"
    return pd.read_parquet(final if final.exists() and not mono_only else DATA / "queries_mono.parquet")


def pool_corpus(passages: pd.DataFrame, pool: str, lang: str) -> pd.DataFrame:
    return passages[(passages.pool == pool) & (passages.lang == lang)].reset_index(drop=True)


def eval_units(queries: pd.DataFrame, passages: pd.DataFrame, split: str):
    """Yield (slice, query_df, corpus_df) for every slice of a split."""
    qs = queries[queries.split == split]
    for sl, g in qs.groupby("slice"):
        assert g.plang.nunique() == 1, sl
        yield sl, g.reset_index(drop=True), pool_corpus(passages, split, g.plang.iloc[0])


# --------------------------------------------------------------------------------------------- retrievers
def _topk_from_scores(scores: np.ndarray, pids: np.ndarray, k: int = TOP, positive_only: bool = False):
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    idx, part = np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)
    out = []
    for i in range(len(idx)):
        keep = part[i] > 0 if positive_only else np.ones(k, dtype=bool)
        out.append(([pids[j] for j in idx[i][keep]], part[i][keep].tolist()))
    return out


class Dense:
    """Sentence-transformers bi-encoder with optional query/passage prefixes (e5) and Arabic normalization."""

    def __init__(
        self,
        path: str | None,
        tag: str,
        q_prefix: str = "",
        p_prefix: str = "",
        max_len: int = 512,
        bs: int = 64,
        normalize_text: bool = False,
        cache: bool = False,
        fp16: bool = True,
        model=None,
    ):
        """Load `path`, or wrap an already-loaded SentenceTransformer (used for dev curves during training)."""
        import torch
        from sentence_transformers import SentenceTransformer

        self.tag, self.q_prefix, self.p_prefix, self.bs = tag, q_prefix, p_prefix, bs
        self.normalize_text, self.cache = normalize_text, cache
        if model is None:
            kw = {"torch_dtype": torch.float16} if fp16 else {}
            model = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu", model_kwargs=kw)
        self.model = model
        self.max_len = max_len
        self._train_len = model.max_seq_length
        self.model.max_seq_length = max_len

    def restore_max_len(self) -> None:
        self.model.max_seq_length = self._train_len

    def _prep(self, texts: list[str], prefix: str) -> list[str]:
        if self.normalize_text:
            texts = [normalize_ar(t) for t in texts]
        return [prefix + t for t in texts]

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        prefix = self.q_prefix if kind == "q" else self.p_prefix
        texts = self._prep(texts, prefix)
        path = None
        if self.cache:
            key = hashlib.sha1(("\n".join(texts) + f"|{self.max_len}").encode()).hexdigest()[:16]
            path = CACHE / self.tag / f"{kind}_{key}.npy"
            if path.exists():
                return np.load(path)
        emb = self.model.encode(
            texts, batch_size=self.bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def rank(self, q: pd.DataFrame, c: pd.DataFrame, top: int = TOP):
        import torch

        qe, ce = self.encode(list(q.text), "q"), self.encode(list(c.text), "p")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        scores = (torch.from_numpy(qe).to(dev) @ torch.from_numpy(ce).to(dev).T).float().cpu().numpy()
        return dict(zip(q.qid, _topk_from_scores(scores, c.pid.to_numpy(), top)))


class BM25:
    """rank-bm25 (Okapi) over a chosen tokenization: 'raw' | 'norm' | 'stem'."""

    def __init__(self, mode: str):
        self.mode, self._idx = mode, {}

    def _index(self, c: pd.DataFrame):
        from rank_bm25 import BM25Okapi

        key = (len(c), c.pid.iloc[0], c.pid.iloc[-1])
        if key not in self._idx:
            self._idx[key] = BM25Okapi([tokenize(t, self.mode) for t in c.text])
        return self._idx[key]

    def rank(self, q: pd.DataFrame, c: pd.DataFrame, top: int = TOP):
        bm, pids = self._index(c), c.pid.to_numpy()
        out = {}
        for qid, text in zip(q.qid, q.text):
            toks = tokenize(text, self.mode)
            scores = bm.get_scores(toks) if toks else np.zeros(len(pids))
            out[qid] = _topk_from_scores(scores[None, :], pids, top, positive_only=True)[0]
        return out


def rrf_runs(runs: list[dict], k: int = 60, top: int = TOP) -> dict:
    out = {}
    for qid in runs[0]:
        fused = M.rrf([r[qid][0] for r in runs], k=k, top=top)
        out[qid] = (fused, [0.0] * len(fused))
    return out


def fuse_slice(bm25_run: dict, dense_run: dict, cross_lingual: bool, gate_language: bool, k: int = 60) -> dict:
    """RRF of a BM25 run and a dense run for one slice.

    With `gate_language`, a cross-lingual slice (query language != passage language) keeps the dense run unchanged:
    BM25 has no shared vocabulary across scripts, so its rank-1 there is noise that rank-based fusion cannot discount.
    """
    if gate_language and cross_lingual:
        return dense_run
    return rrf_runs([bm25_run, dense_run], k=k)


# --------------------------------------------------------------------------------------------- evaluation
def evaluate_runs(runs_by_slice: dict[str, dict], queries: pd.DataFrame, split: str) -> dict:
    """runs_by_slice[slice][qid] = (pids, scores). Returns slice summaries, direction macros and per-query arrays."""
    qs = queries[queries.split == split]
    slices, perq = {}, {}
    for sl, run in runs_by_slice.items():
        g = qs[qs.slice == sl]
        qrels = {q: {p} for q, p in zip(g.qid, g.pos_pid)}
        pq = M.evaluate_run({q: v[0] for q, v in run.items()}, qrels, extra_hits=(50, 100))
        slices[sl] = M.summarize(pq)
        perq[sl] = pq
    return {"slices": slices, "directions": direction_macro(slices), "perq": perq}


def direction_macro(slices: dict[str, dict]) -> dict:
    by_dir = defaultdict(list)
    for sl, s in slices.items():
        by_dir[DIRECTION_OF[sl]].append(s)
    out = {}
    for d in DIRECTIONS:
        if d in by_dir:
            keys = [k for k in by_dir[d][0] if k != "n"]
            out[d] = {k: float(np.mean([s[k] for s in by_dir[d]])) for k in keys}
            out[d]["n_slices"] = len(by_dir[d])
    if out:
        keys = [k for k in next(iter(out.values())) if k != "n_slices"]
        out["macro"] = {k: float(np.mean([out[d][k] for d in out if d != "macro"])) for k in keys}
    return out


def run_system(retriever, queries: pd.DataFrame, passages: pd.DataFrame, split: str) -> dict[str, dict]:
    return {sl: retriever.rank(q, c) for sl, q, c in eval_units(queries, passages, split)}


def result_path(name: str, split: str, suffix: str, base: Path | None = None) -> Path:
    """File of a stored result. A pipeline system is named like `e5-small>>rr_ft_bce_s13` (first stage >> reranker);
    the `>>` becomes `__` in file names so that the files are valid on every OS."""
    return (base or RESULTS) / f"{name.replace('>>', '__')}.{split}.{suffix}"


def save_result(name: str, split: str, res: dict, runs: dict | None = None, extra: dict | None = None) -> None:
    payload = {"system": name, "split": split, "slices": res["slices"], "directions": res["directions"], **(extra or {})}
    result_path(name, split, "json").write_text(json.dumps(payload, indent=1))
    np.savez_compressed(result_path(name, split, "perq.npz"), **{f"{sl}|{m}": v for sl, d in res["perq"].items() for m, v in d.items()})
    if runs is not None:
        slim = {sl: {q: v[0] for q, v in r.items()} for sl, r in runs.items()}
        result_path(name, split, "runs.json").write_text(json.dumps(slim))


def load_runs(name: str, split: str) -> dict[str, dict]:
    raw = json.loads(result_path(name, split, "runs.json").read_text())
    return {sl: {q: (p, [0.0] * len(p)) for q, p in r.items()} for sl, r in raw.items()}


def load_perq(name: str, split: str, base: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Per-query metric arrays of a stored result (`base` = results directory, default: this pipeline's own)."""
    z = np.load(result_path(name, split, "perq.npz", base))
    out: dict = defaultdict(dict)
    for k in z.files:
        sl, m = k.split("|")
        out[sl][m] = z[k]
    return out


def fmt_row(res: dict, metric: str = "mrr@10") -> str:
    return " ".join(f"{d}={res['directions'][d][metric]:.3f}" for d in DIRECTIONS if d in res["directions"])


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t
