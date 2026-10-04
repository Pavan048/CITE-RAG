"""Smoke test with litellm.completion monkeypatched — there is no real BYOK key available in this
environment, so this verifies the client builds correct request shapes and passes `api_key`
through per-call (never via a global/env var), not that a real provider responds well.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import llm_client as llm_client_module


def _fake_completion(**kwargs):
    assert kwargs["api_key"] == "sk-test-key"
    content = "CAPTION" if any(isinstance(m.get("content"), list) for m in kwargs["messages"]) else "TEXT RESPONSE"
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_complete_text_passes_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)
    client = llm_client_module.LLMClient()
    result = client.complete_text(prompt="hi", api_key="sk-test-key", model="openai/gpt-4o-mini")
    assert result == "TEXT RESPONSE"


def test_caption_image_builds_multimodal_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client_module.litellm, "completion", _fake_completion)
    client = llm_client_module.LLMClient()
    result = client.caption_image(prompt="describe", image_bytes=b"fakepng", api_key="sk-test-key", model="openai/gpt-4o-mini")
    assert result == "CAPTION"
