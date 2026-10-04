"""Thin HTTP client for the self-hosted BGE-M3 embedding service (PRD Section 1, Section 7).

BGE-M3 is decoupled from the user's BYOK key on purpose (Section 1: "a user switching LLM
providers never orphans the existing vector index") and produces dense + sparse output from one
forward pass. We can't run the actual GPU model in this codebase's process, so this client targets
an explicit HTTP contract that whatever serves BGE-M3 (e.g. Infinity, or a small custom FastAPI
wrapper around `FlagEmbedding.BGEM3FlagModel`) must implement:

    POST {base_url}/embed   {"texts": [str, ...]}
      -> {"dense": [[float, ...], ...], "sparse": [{"indices": [int,...], "values": [float,...]}, ...]}

Adjust this contract to match the real inference server at deployment time — nothing outside this
file should know or care what shape that server actually speaks.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from app.config import settings


class SparseVectorData(BaseModel):
    indices: list[int]
    values: list[float]


class EmbeddingResult(BaseModel):
    dense: list[list[float]]
    sparse: list[SparseVectorData]


class EmbeddingService:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self._base_url = (base_url or settings.embedding_service_url).rstrip("/")
        self._timeout_s = timeout_s or settings.inference_service_timeout_s

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Batched dense+sparse embedding for `texts` (Section 2 Stage 4: "batch calls")."""
        if not texts:
            return EmbeddingResult(dense=[], sparse=[])
        response = httpx.post(
            f"{self._base_url}/embed",
            json={"texts": texts},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        return EmbeddingResult(
            dense=body["dense"],
            sparse=[SparseVectorData(**s) for s in body["sparse"]],
        )

    def embed_one_dense(self, text: str) -> list[float]:
        """Convenience for the many call sites that only need a single dense query vector
        (sub-question embedding for doc routing, chunk search query vector)."""
        return self.embed([text]).dense[0]


embedding_service = EmbeddingService()
