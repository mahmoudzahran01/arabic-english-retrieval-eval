"""OpenAI-compatible chat client used by every script in `arretrieval.rageval`.

Three properties matter because the LLM server may be shared with other jobs:
  1. HARD concurrency cap, enforced across *processes* with flock slot files (default 2 for the `gemma` role).
  2. On-disk SQLite cache keyed by the full request, so re-runs are free and partial progress survives crashes.
  3. enable_thinking=false on every call (otherwise reasoning models burn max_tokens on hidden reasoning); this is passed
     as `chat_template_kwargs`, a vLLM / llama.cpp extension of the OpenAI schema.

The cache key includes `seed`, which is how we draw several *different* samples of the same prompt
(e.g. Ragas answer-relevancy asks for n=3 generated questions).

Endpoints come from the environment (`GEMMA_BASE_URL`, `GEMMA_API_KEY`, `GEMMA_MODEL`, `GEMMA_SLOTS` and the `QWEN_*`
equivalents); see `arretrieval/config.py` and `.env.example`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..config import endpoint
from .paths import CACHE

SLOT_DIR = CACHE / "slots"


class LLM:
    def __init__(self, name: str = "gemma"):
        ep = endpoint(name)
        self.name, self.base, self.model, self.slots = name, ep.base_url, ep.model, ep.slots
        self._key = ep.api_key
        self._client = httpx.Client(timeout=httpx.Timeout(240.0, connect=10.0))
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(CACHE / f"llm_{name}.sqlite"), check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    # -- cache ---------------------------------------------------------------------------------
    def _k(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _get(self, k: str):
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, k: str, v: dict):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
            self._db.commit()

    # -- cross-process slot limiter --------------------------------------------------------------
    @contextlib.contextmanager
    def _slot(self):
        SLOT_DIR.mkdir(parents=True, exist_ok=True)
        fds = [os.open(str(SLOT_DIR / f"{self.name}.{i}.lock"), os.O_CREAT | os.O_RDWR) for i in range(self.slots)]
        got = None
        try:
            while got is None:
                for fd in fds:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        got = fd
                        break
                    except BlockingIOError:
                        continue
                else:
                    time.sleep(0.03 + random.random() * 0.03)
            yield
        finally:
            if got is not None:
                fcntl.flock(got, fcntl.LOCK_UN)
            for fd in fds:
                os.close(fd)

    # -- main call -------------------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict] | str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 0,
        response_format: dict | None = None,
        retries: int = 4,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format:
            payload["response_format"] = response_format
        k = self._k(payload)
        hit = self._get(k)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit["text"]
        last = None
        for attempt in range(retries):
            try:
                t0 = time.time()
                with self._slot():
                    r = self._client.post(f"{self.base}/chat/completions", json=payload, headers={"Authorization": f"Bearer {self._key}"})
                r.raise_for_status()
                d = r.json()
                text = d["choices"][0]["message"]["content"] or ""
                u = d.get("usage", {})
                self.stats["calls"] += 1
                self.stats["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.stats["completion_tokens"] += u.get("completion_tokens", 0)
                self.stats["seconds"] += time.time() - t0
                self._put(k, {"text": text, "usage": u})
                return text
            except Exception as e:  # noqa: BLE001 - retried, then raised with context
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} tries: {last!r}")

    def pmap(self, fn, items, workers: int | None = None, desc: str = "", every: int = 20):
        """Apply fn to items with `workers` threads (default = slots). Order preserved. Progress to stderr."""
        import sys

        items = list(items)
        out = [None] * len(items)
        done = [0]
        t0 = time.time()

        def run(i):
            out[i] = fn(items[i])
            done[0] += 1
            if desc and (done[0] % every == 0 or done[0] == len(items)):
                print(f"[{desc}] {done[0]}/{len(items)}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

        with ThreadPoolExecutor(max_workers=workers or self.slots) as ex:
            list(ex.map(run, range(len(items))))
        return out


_instances: dict[str, LLM] = {}


def get_llm(name: str = "gemma") -> LLM:
    if name not in _instances:
        _instances[name] = LLM(name)
    return _instances[name]


def parse_json(text: str):
    """Pull the first JSON object/array out of a model reply (tolerates ```json fences and prose)."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts[1::2]:
            p = p[4:] if p.lstrip().lower().startswith("json") else p
            try:
                return json.loads(p.strip())
            except Exception:  # noqa: BLE001
                continue
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda oc: t.find(oc[0]) if t.find(oc[0]) != -1 else 10**9)  # outermost structure first
    for opener, closer in pairs:
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except Exception:  # noqa: BLE001
                continue
    raise ValueError(f"no JSON in reply: {text[:120]!r}")
