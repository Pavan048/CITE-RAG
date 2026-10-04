"""PRD Section 2 Stage 5: index. Writes both Qdrant collections for one document. Job/document
status transitions (including the "mark ready only after both Qdrant writes and the Mongo document
update succeed" rule) are owned by pipeline.py, which calls this after both Qdrant writes return
successfully — this module has no Mongo dependency at all, keeping the Stage 5 files each
responsible for exactly one storage system.
"""

from __future__ import annotations

from datetime import datetime

from app.models.mongo_models import ChunkPayload, DocumentSummaryPayload
from app.services.embedding_service import EmbeddingResult, SparseVectorData
from app.services.qdrant_client import qdrant_service


def index_document(
    doc_id: str,
    chunks: list[tuple[str, ChunkPayload]],
    chunk_embeddings: EmbeddingResult,
    summary_dense: list[float],
    filename: str,
    page_count: int,
    uploaded_at: datetime,
    content_hash: str,
) -> None:
    items: list[tuple[str, list[float], SparseVectorData, ChunkPayload]] = [
        (chunk_id, chunk_embeddings.dense[i], chunk_embeddings.sparse[i], payload)
        for i, (chunk_id, payload) in enumerate(chunks)
    ]
    qdrant_service.upsert_chunks_batch(items)
    qdrant_service.upsert_document_summary(
        doc_id,
        summary_dense,
        DocumentSummaryPayload(doc_id=doc_id, filename=filename, page_count=page_count, uploaded_at=uploaded_at, content_hash=content_hash),
    )
