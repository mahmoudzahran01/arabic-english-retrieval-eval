"""Environment-driven configuration shared by the `retrieval` and `rageval` packages.

Nothing machine-specific lives in the code: output directories and LLM endpoints come from environment
variables (see `.env.example`), with the defaults documented below.

Directories
  ARR_ROOT          repository root (default: two levels above this package, i.e. the checkout)
  ARR_RUNS_DIR      where pipelines write large, git-ignored outputs (default: <ARR_ROOT>/runs)

LLM endpoints (any OpenAI-compatible `/chat/completions` server, e.g. vLLM or llama.cpp's llama-server)
  The names are the *roles* used in the experiments. The published numbers were produced with
  Gemma-4-26B-A4B behind `GEMMA_*` and Qwen3-4B-Instruct-2507 behind `QWEN_*`.
    GEMMA_BASE_URL    default http://localhost:8000/v1
    GEMMA_API_KEY     default EMPTY (many local servers ignore it)
    GEMMA_MODEL       default gemma4-26b-a4b  (the served model name)
    GEMMA_SLOTS       max concurrent requests, enforced across processes (default 2)
    QWEN_BASE_URL     default http://localhost:8001/v1
    QWEN_API_KEY      default EMPTY
    QWEN_MODEL        default qwen-judge
    QWEN_SLOTS        default 6
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.getenv("ARR_ROOT", Path(__file__).resolve().parents[2]))
RUNS_DIR = Path(os.getenv("ARR_RUNS_DIR", ROOT / "runs"))


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible chat endpoint (role name, base URL ending in /v1, model name, API key, concurrency cap)."""

    name: str
    base_url: str
    model: str
    api_key: str
    slots: int


def endpoint(name: str) -> Endpoint:
    """Resolve the endpoint for a role (`gemma` or `qwen`) from the environment."""
    defaults = {
        "gemma": ("http://localhost:8000/v1", "gemma4-26b-a4b", 2),
        "qwen": ("http://localhost:8001/v1", "qwen-judge", 6),
    }
    if name not in defaults:
        raise KeyError(f"unknown LLM role {name!r}; expected one of {sorted(defaults)}")
    base, model, slots = defaults[name]
    up = name.upper()
    return Endpoint(
        name=name,
        base_url=os.getenv(f"{up}_BASE_URL", base).rstrip("/"),
        model=os.getenv(f"{up}_MODEL", model),
        api_key=os.getenv(f"{up}_API_KEY", "EMPTY"),
        slots=int(os.getenv(f"{up}_SLOTS", str(slots))),
    )
