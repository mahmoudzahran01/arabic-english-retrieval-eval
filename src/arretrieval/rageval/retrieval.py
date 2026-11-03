"""Retrieval for the RAG system under test: embed -> exact cosine search -> optional cross-encoder rerank.

Both models run in-process (CPU is enough for a 220-passage corpus):
  EMBED_MODEL   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (384-d, truncates at 128 word-pieces)
  RERANK_MODEL  cross-encoder/mmarco-mMiniLMv2-L12-H384-v1                    (256-token limit)

The published runs queried a self-hosted HTTP embed/rerank service and a Qdrant collection that served exactly these two
models with exact cosine search; vectors computed in-process were checked against the stored ones (cosine 1.000). This
module keeps the same small API (`create_collection`, `upsert`, `search`, `retrieve`) on top of an on-disk exact-cosine
index (a numpy matrix plus a json payload list per collection), so no service is needed to reproduce the pipeline.

Why a deliberately plain dense retriever? The point of this part of the repository is to evaluate a RAG *system* and its
judges, and the weak spots of a small multilingual embedder (128 word-piece truncation, cross-lingual drift) are what
gives the evaluation something real to find.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading

import numpy as np

from .paths import CACHE

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
INDEX_DIR = CACHE / "index"

_lock = threading.Lock()
_db: sqlite3.Connection | None = None
_models: dict = {}
_index: dict[str, tuple[np.ndarray, list[dict]]] = {}


def _embedding_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(str(CACHE / "embeddings.sqlite"), check_same_thread=False, timeout=60)
        _db.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v TEXT)")
    return _db


def _local_models() -> dict:
    """Load the two models once per process (sentence-transformers; set HF_HOME to relocate the download cache)."""
    if not _models:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        torch.set_num_threads(int(os.getenv("ARR_CPU_THREADS", "6")))
        _models["emb"] = SentenceTransformer(EMBED_MODEL, device="cpu")
        _models["rr"] = CrossEncoder(RERANK_MODEL, device="cpu", max_length=256)
    return _models


def embed(texts: list[str], batch: int = 16) -> list[list[float]]:
    """Embed (L2-normalised); per-text disk cache because queries repeat across configurations."""
    db = _embedding_db()
    out: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        k = hashlib.sha1(t.encode()).hexdigest()
        with _lock:
            row = db.execute("SELECT v FROM e WHERE k=?", (k,)).fetchone()
        if row:
            out[i] = json.loads(row[0])
        else:
            todo.append((i, k, t))
    for s in range(0, len(todo), batch):
        part = todo[s : s + batch]
        vecs = _local_models()["emb"].encode([t for _, _, t in part], normalize_embeddings=True, batch_size=32).tolist()
        with _lock:
            for (i, k, _), v in zip(part, vecs):
                out[i] = v
                db.execute("INSERT OR REPLACE INTO e VALUES (?,?)", (k, json.dumps(v)))
            db.commit()
    return out


def rerank(query: str, docs: list[str]) -> tuple[list[float], list[int]]:
    """Cross-encoder scores for (query, doc) pairs and the document order, best first."""
    scores = [float(x) for x in _local_models()["rr"].predict([(query, d) for d in docs], batch_size=32)]
    return scores, sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


# ------------------------------------------------------------------------------------- exact-cosine collections
def _paths(name: str):
    return INDEX_DIR / f"{name}.npy", INDEX_DIR / f"{name}.json"


def create_collection(name: str, dim: int = 384, recreate: bool = False) -> None:
    """Create an empty collection (no-op if it exists unless `recreate`). `dim` is kept for API compatibility."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if recreate:
        for p in _paths(name):
            p.unlink(missing_ok=True)
        _index.pop(name, None)


def _load(name: str) -> tuple[np.ndarray, list[dict]]:
    if name not in _index:
        vp, pp = _paths(name)
        if not vp.exists():
            raise FileNotFoundError(f"collection {name!r} not found in {INDEX_DIR}; run `python -m arretrieval.rageval.ingest` first")
        _index[name] = (np.load(vp), json.loads(pp.read_text()))
    return _index[name]


def upsert(name: str, points: list[dict]) -> None:
    """points: [{'id': int, 'vector': [...], 'payload': {...}}]; an existing id is replaced."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vp, pp = _paths(name)
    if vp.exists():
        vecs, payloads = list(np.load(vp)), json.loads(pp.read_text())
    else:
        vecs, payloads = [], []
    ids = [p["_id"] for p in payloads]
    for pt in points:
        payload = {"_id": pt["id"], **pt["payload"]}
        vec = np.asarray(pt["vector"], dtype=np.float32)
        if pt["id"] in ids:
            j = ids.index(pt["id"])
            vecs[j], payloads[j] = vec, payload
        else:
            ids.append(pt["id"])
            vecs.append(vec)
            payloads.append(payload)
    np.save(vp, np.stack(vecs))
    pp.write_text(json.dumps(payloads, ensure_ascii=False))
    _index.pop(name, None)


def search(name: str, qvec: list[float], limit: int) -> list[dict]:
    """Exact cosine top-`limit`: [{'score': float, **payload}], best first (ties keep insertion order)."""
    mat, payloads = _load(name)
    q = np.asarray(qvec, dtype=np.float32)
    scores = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
    order = np.argsort(-scores, kind="stable")[:limit]
    return [{"score": float(scores[i]), **{k: v for k, v in payloads[i].items() if k != "_id"}} for i in order]


def drop_index() -> None:
    """Delete every stored collection (they are rebuilt by `ingest`)."""
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    _index.clear()


def retrieve(collection: str, question: str, k: int = 3, use_rerank: bool = False, fetch: int = 10) -> list[dict]:
    """Return top-k passages [{id, lang, text, score, rerank_score?, rank}], dense only or dense->rerank."""
    qv = embed([question])[0]
    hits = search(collection, qv, fetch if use_rerank else k)
    if use_rerank:
        scores, order = rerank(question, [h["text"] for h in hits])
        hits = [{**hits[i], "rerank_score": scores[i]} for i in order]
    out = []
    for r, h in enumerate(hits[:k], 1):
        out.append({**h, "rank": r})
    return out
