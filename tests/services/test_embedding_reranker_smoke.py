"""Smoke tests against tests/mock_inference_server.py (run on :8001 / :8002) — verifies the HTTP
client code paths, not embedding/reranking quality (see that file's docstring for why)."""

from __future__ import annotations

from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService


def test_embedding_service_smoke() -> None:
    svc = EmbeddingService(base_url="http://localhost:8001")
    result = svc.embed(["hello world", "goodbye"])
    assert len(result.dense) == 2
    assert len(result.dense[0]) == 1024
    assert len(result.sparse) == 2
    assert result.sparse[0].indices

    vec = svc.embed_one_dense("hello world")
    assert len(vec) == 1024


def test_reranker_service_smoke() -> None:
    svc = RerankerService(base_url="http://localhost:8002")
    results = svc.rerank("hello world", ["hello world exactly", "totally unrelated text"])
    assert results[0].index == 0
    assert results[0].score >= results[1].score
