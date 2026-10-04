"""A stand-in server implementing the HTTP contracts `embedding_service.py` and
`reranker_service.py` expect (see their docstrings) — built so this sandbox (no GPU) can still run
something real instead of BGE-M3 / BGE-reranker-v2-m3.

Dense embeddings use `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-friendly, ~90MB) —
genuine semantic embeddings, not a hash trick: they understand meaning, typos, and paraphrasing
(e.g. "get infomration about the pavan" still matches a resume about Pavan) in a way a bag-of-words
hash never could, which is what motivated swapping this in. Padded to 1024 dims with zeros to match
the Qdrant collection schema (`chunks_collection`/`documents_collection` are sized for BGE-M3's
1024-dim output) — zero-padding both sides of a cosine similarity leaves the result unchanged,
so nothing downstream needs to know the real vectors are only 384-dim underneath.

Reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` — a real, small, CPU-friendly cross-encoder,
not a word-overlap heuristic.

Sparse vectors still use a feature-hashed bag-of-words (the "hashing trick", à la scikit-learn's
HashingVectorizer) — this is a reasonable stand-in for BM25-style sparse retrieval on its own
terms, and wasn't the part causing trouble, so it's unchanged.

None of this is BGE-M3/BGE-reranker-v2-m3 quality (Section 1) — it is real, learned semantics
instead of a hashing trick, which is what actually matters for this sandbox to be usable
interactively, but it is still a smaller, more general-purpose model than the PRD's specified ones.
"""

from __future__ import annotations

import hashlib
import re

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

app = FastAPI()

DIM = 1024
_MODEL_DIM = 384

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _word_hash(word: str) -> int:
    return int(hashlib.sha256(word.encode()).hexdigest(), 16)


def _pseudo_dense(text: str) -> list[float]:
    vector = _embedding_model.encode(text, normalize_embeddings=True).tolist()
    return vector + [0.0] * (DIM - _MODEL_DIM)


def _pseudo_sparse(text: str) -> dict[str, list]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    indices = sorted({_word_hash(w) % 30000 for w in words}) or [0]
    values = [1.0 / (i + 1) for i in range(len(indices))]
    return {"indices": indices, "values": values}


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    return {
        "dense": [_pseudo_dense(t) for t in req.texts],
        "sparse": [_pseudo_sparse(t) for t in req.texts],
    }


class RerankRequest(BaseModel):
    query: str
    texts: list[str]


@app.post("/rerank")
def rerank(req: RerankRequest) -> list[dict]:
    if not req.texts:
        return []
    scores = _cross_encoder.predict([(req.query, t) for t in req.texts])
    scored = [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
    return sorted(scored, key=lambda r: r["score"], reverse=True)
