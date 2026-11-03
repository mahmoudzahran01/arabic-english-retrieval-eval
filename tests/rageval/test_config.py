"""LLM endpoints come from the environment; nothing machine-specific is baked in."""

import pytest

from arretrieval.config import endpoint


def test_defaults_are_generic_local_endpoints(monkeypatch):
    for k in ("GEMMA_BASE_URL", "GEMMA_API_KEY", "GEMMA_MODEL", "GEMMA_SLOTS"):
        monkeypatch.delenv(k, raising=False)
    ep = endpoint("gemma")
    assert ep.base_url == "http://localhost:8000/v1" and ep.api_key == "EMPTY" and ep.slots == 2 and ep.model == "gemma4-26b-a4b"


def test_environment_overrides_every_field(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "https://llm.example.org/v1/")
    monkeypatch.setenv("QWEN_API_KEY", "k")
    monkeypatch.setenv("QWEN_MODEL", "my-judge")
    monkeypatch.setenv("QWEN_SLOTS", "12")
    ep = endpoint("qwen")
    assert (ep.base_url, ep.api_key, ep.model, ep.slots) == ("https://llm.example.org/v1", "k", "my-judge", 12)  # trailing slash stripped


def test_unknown_role_is_rejected():
    with pytest.raises(KeyError):
        endpoint("gpt")
