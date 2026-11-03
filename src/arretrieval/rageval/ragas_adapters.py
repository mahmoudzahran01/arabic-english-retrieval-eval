"""Glue between Ragas 0.4.x "collections" metrics and our own cached, slot-limited LLM client.

Why a custom adapter instead of ragas.llms.llm_factory(...)?
  * llm_factory goes through the `instructor` library + AsyncOpenAI. That works with OpenAI-style servers, but we need
    (a) the hard concurrency cap (cross-process flock, so a shared server is never flooded), (b) an on-disk cache so a crashed
    or repeated run costs nothing, (c) `chat_template_kwargs.enable_thinking=false` on every call.
  * Ragas only needs `generate()/agenerate(prompt, ResponseModel) -> ResponseModel` (InstructorBaseRagasLLM), so we
    implement exactly that on top of llama.cpp's grammar-constrained `response_format: json_schema`.
Ragas' own prompts, claim-extraction and NLI logic are used unchanged.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib

import numpy as np
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

from .llm import LLM

SCORING_KEY: contextvars.ContextVar[str] = contextvars.ContextVar("scoring_key", default="")


class CachedRagasLLM(InstructorBaseRagasLLM):
    def __init__(self, llm: LLM, temperature: float = 0.0, max_tokens: int = 2000, sampling: bool = False):
        self.llm, self.temperature, self.max_tokens, self.sampling = llm, temperature, max_tokens, sampling
        self._counts: dict[tuple, int] = {}
        self.parse_failures = 0
        self.calls = 0

    def _seed(self, prompt: str) -> int:
        if not self.sampling:
            return 0
        k = (SCORING_KEY.get(), hashlib.md5(prompt.encode()).hexdigest())
        n = self._counts.get(k, 0)
        self._counts[k] = n + 1
        return n

    def generate(self, prompt: str, response_model):
        schema = response_model.model_json_schema()
        rf = {"type": "json_schema", "json_schema": {"name": response_model.__name__, "strict": True, "schema": schema}}
        seed = self._seed(prompt)
        last = None
        self.calls += 1
        for attempt in range(3):
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature if attempt == 0 else 0.3,
                seed=seed + 1000 * attempt,
                response_format=rf,
            )
            try:
                return response_model.model_validate_json(raw)
            except Exception as e:  # noqa: BLE001 - retried, counted, finally raised
                last = e
                self.parse_failures += 1
        raise ValueError(f"structured output failed 3x: {last!r}")

    async def agenerate(self, prompt: str, response_model):
        return await asyncio.to_thread(self.generate, prompt, response_model)


class LocalSTEmbeddings(BaseRagasEmbedding):
    """Local sentence-transformers model (default BAAI/bge-m3: strong multilingual/Arabic)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512

    def embed_text(self, text: str, **kw):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    async def aembed_text(self, text: str, **kw):
        return await asyncio.to_thread(self.embed_text, text)

    def embed_texts(self, texts, **kw):
        return self.model.encode(list(texts), normalize_embeddings=True, batch_size=16).tolist()

    async def aembed_texts(self, texts, **kw):
        return await asyncio.to_thread(self.embed_texts, texts)


class MiniLMEmbeddings(BaseRagasEmbedding):
    """The RAG system's own embedder, paraphrase-multilingual-MiniLM-L12-v2 (384-d, 128 word-piece limit), via retrieval.embed."""

    def __init__(self):
        super().__init__()
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"

    def embed_text(self, text: str, **kw):
        from . import retrieval as R

        return R.embed([text])[0]

    async def aembed_text(self, text: str, **kw):
        return await asyncio.to_thread(self.embed_text, text)

    def embed_texts(self, texts, **kw):
        from . import retrieval as R

        return R.embed(list(texts))

    async def aembed_texts(self, texts, **kw):
        return await asyncio.to_thread(self.embed_texts, texts)


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
