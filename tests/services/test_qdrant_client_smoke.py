"""Smoke test against a real local Qdrant (docker-compose.dev.yml) — not mocked, since Qdrant is
cheap to run locally and the whole point is to verify our filter/fusion/index logic against the
real server semantics, not our own assumptions about them.
"""

from __future__ import annotations

import uuid

from app.models.mongo_models import ChunkPayload, ContentType, DocumentSummaryPayload
from app.services.embedding_service import SparseVectorData
from app.services.qdrant_client import QdrantService


def test_qdrant_roundtrip() -> None:
    svc = QdrantService()
    svc.ensure_collections()

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    svc.upsert_document_summary(
        doc_id,
        dense=[0.1] * 1024,
        payload=DocumentSummaryPayload(
            doc_id=doc_id, filename="test.pdf", page_count=2, uploaded_at="2026-01-01T00:00:00Z", content_hash="abc"
        ),
    )
    svc.upsert_chunks_batch(
        [
            (
                chunk_id,
                [0.1] * 1024,
                SparseVectorData(indices=[1, 2, 3], values=[1.0, 0.5, 0.25]),
                ChunkPayload(
                    doc_id=doc_id,
                    page_number=1,
                    section_title="Intro",
                    content_type=ContentType.TEXT,
                    chunk_index=0,
                    parent_block_id="pb1",
                    content_hash="hash1",
                    token_count=10,
                    chunk_text="hello world this is a test chunk",
                ),
            )
        ]
    )

    doc_results = svc.search_documents(query_dense=[0.1] * 1024, top_k=5, score_threshold=0.0)
    assert any(r.doc_id == doc_id for r in doc_results)

    chunk_results = svc.hybrid_search_chunks(
        query_dense=[0.1] * 1024,
        query_sparse=SparseVectorData(indices=[1, 2], values=[1.0, 0.5]),
        doc_ids=[doc_id],
        limit_per_list=10,
        rrf_k=60,
        fused_top_k=10,
    )
    assert any(r.chunk_id == chunk_id for r in chunk_results)
    found = next(r for r in chunk_results if r.chunk_id == chunk_id)
    assert found.payload.chunk_text == "hello world this is a test chunk"

    svc.delete_document(doc_id)
    doc_results_after = svc.search_documents(query_dense=[0.1] * 1024, top_k=5, score_threshold=0.0)
    assert not any(r.doc_id == doc_id for r in doc_results_after)
    chunk_results_after = svc.hybrid_search_chunks(
        query_dense=[0.1] * 1024,
        query_sparse=SparseVectorData(indices=[1, 2], values=[1.0, 0.5]),
        doc_ids=[doc_id],
        limit_per_list=10,
        rrf_k=60,
        fused_top_k=10,
    )
    assert not any(r.chunk_id == chunk_id for r in chunk_results_after)
