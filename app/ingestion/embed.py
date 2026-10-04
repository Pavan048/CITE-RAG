"""PRD Section 2 Stage 4: dual embed. A thin batching layer over `embedding_service` — batches keep
a single HTTP payload bounded regardless of document size (a 2000-page document could produce
thousands of chunks).
"""

from __future__ import annotations

from app.config import settings
from app.services.embedding_service import EmbeddingResult, embedding_service


def embed_chunks(chunk_texts: list[str]) -> EmbeddingResult:
    if not chunk_texts:
        return EmbeddingResult(dense=[], sparse=[])
    batch_size = settings.embedding_batch_size
    dense: list[list[float]] = []
    sparse = []
    for i in range(0, len(chunk_texts), batch_size):
        batch_result = embedding_service.embed(chunk_texts[i : i + batch_size])
        dense.extend(batch_result.dense)
        sparse.extend(batch_result.sparse)
    return EmbeddingResult(dense=dense, sparse=sparse)


def embed_document_summary(summary_text: str) -> list[float]:
    return embedding_service.embed_one_dense(summary_text)
