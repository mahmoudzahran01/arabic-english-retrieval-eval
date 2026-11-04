"""Minimal embedding server over an exported ONNX variant (CPU), with a TEI-style `/embed` route.

    pip install fastapi uvicorn onnxruntime tokenizers numpy
    ONNX_ROOT=artifacts/v2-base VARIANT=int8 python -m uvicorn deploy.serve:app --port 8080

    curl -s localhost:8080/embed -H 'content-type: application/json' \
         -d '{"inputs": ["When was the university founded?"], "kind": "query"}'

Request  : {"inputs": "text" | ["text", ...], "kind": "query" | "passage" | null}
           `kind` prepends the e5 prefix ("query: " / "passage: "); leave it out if your inputs are already prefixed.
Response : [[float, ...], ...]  (L2-normalised, same shape as text-embeddings-inference's /embed)

For a production setup, text-embeddings-inference (TEI) can serve the sentence-transformers folder directly
(`--model-id <dir> --pooling mean` for the e5 models, `--pooling cls` for v2-m3); that path was not tested here.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MAX_BATCH = 64
MAX_CHARS = 20_000

app = FastAPI(title="Arabic-English retrieval embeddings (ONNX)")


@lru_cache(maxsize=1)
def encoder():
    from .common import OnnxEncoder

    root = os.environ.get("ONNX_ROOT", "artifacts/v2-base")
    variant = os.environ.get("VARIANT", "int8")
    prefix = os.environ.get("PREFIX", "e5")  # "e5" | "none" (bge-m3 family)
    threads = int(os.environ.get("THREADS", "2"))
    return OnnxEncoder(f"{root}/{variant}/model.onnx", root, prefix=prefix, threads=threads)


class EmbedRequest(BaseModel):
    inputs: str | list[str]
    kind: str | None = Field(default=None, pattern="^(query|passage)$")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest) -> list[list[float]]:
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    if not texts or len(texts) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"send between 1 and {MAX_BATCH} inputs")
    if any(len(t) > MAX_CHARS for t in texts):
        raise HTTPException(status_code=413, detail=f"each input must be at most {MAX_CHARS} characters")
    enc = encoder()
    if req.kind is None:  # inputs already carry their prefix
        enc_kind, texts = "p", texts
        saved = enc.p_prefix
        enc.p_prefix = ""
        try:
            return enc.encode(texts, enc_kind).tolist()
        finally:
            enc.p_prefix = saved
    return enc.encode(texts, "q" if req.kind == "query" else "p").tolist()
