"""Thin HTTP client for the self-hosted BGE-reranker-v2-m3 service (PRD Section 1, Section 7:
"swappable interface" — swapping means pointing `reranker_service_url` at a different deployment,
never touching a call site).

Targets HuggingFace Text-Embeddings-Inference's real `/rerank` contract (a plausible, commonly used
way to self-host a BGE reranker model):

    POST {base_url}/rerank   {"query": str, "texts": [str, ...]}
      -> [{"index": int, "score": float}, ...]
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from app.config import settings


class RerankResult(BaseModel):
    index: int  # index into the `texts` list this client sent
    score: float


class RerankerService:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self._base_url = (base_url or settings.reranker_service_url).rstrip("/")
        self._timeout_s = timeout_s or settings.inference_service_timeout_s

    def rerank(self, query: str, texts: list[str]) -> list[RerankResult]:
        """Returns results sorted best-first (matches TEI's own ordering)."""
        if not texts:
            return []
        response = httpx.post(
            f"{self._base_url}/rerank",
            json={"query": query, "texts": texts},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return [RerankResult(**r) for r in response.json()]


reranker_service = RerankerService()
