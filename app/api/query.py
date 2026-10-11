"""PRD Section 4: POST /v1/query. Also POST /v1/query/stream — not part of the PRD's documented
contract, added so the UI can render the answer arriving token-by-token instead of only a loading
spinner. Both endpoints share the exact same citation-validation logic
(retrieval.nodes.generate_cite.resolve_citations) and debug-trace assembly
(retrieval.debug_trace.build_debug_info) — the only thing that differs is whether the final LLM
call is made via `llm_client.complete_text` (blocking, inside the graph) or `llm_client.stream_text`
(streamed, outside the graph — see generate_cite.py and graph.py's module docstrings for why).
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.schemas import QueryRequest, QueryResponse, TokenSummary
from app.retrieval.debug_trace import build_debug_info
from app.retrieval.graph import retrieval_graph, retrieval_graph_pre_generation, run_with_trace
from app.retrieval.nodes.generate_cite import (
    NOT_FOUND_ANSWER,
    RETRIEVAL_UNAVAILABLE_ANSWER,
    build_grounded_answer_prompt,
    resolve_citations,
)
from app.retrieval.state import LLMCallRecord, RetrievalState
from app.services.embedding_service import embedding_service
from app.services.llm_client import llm_client
from app.services.semantic_cache import semantic_cache

router = APIRouter()


def _build_initial_state(body: QueryRequest, llm_key: str, llm_model: str) -> RetrievalState:
    return RetrievalState(
        question=body.question,
        llm_api_key=llm_key,
        llm_model=llm_model,
        file_ids=body.file_ids,
        top_k=body.top_k or settings.default_top_k,
    )


def _routed_docs(state: RetrievalState) -> list[str] | None:
    # `routed_docs` is null whenever the caller pinned `file_ids` (routing never ran, Section 4) —
    # the graph state itself only ever holds a plain accumulable list (see state.py), so that
    # mapping happens here, at the API boundary.
    return None if state.file_ids else (state.routed_docs or None)


@router.post("/v1/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    x_llm_key: str = Header(..., alias="X-LLM-Key"),
    x_llm_model: str | None = Header(None, alias="X-LLM-Model"),
) -> QueryResponse:
    llm_model = x_llm_model or settings.default_llm_model
    initial_state = _build_initial_state(body, x_llm_key, llm_model)

    # --- Semantic cache lookup ---
    # Embed the query for cache lookup. Failure is non-fatal: an unreachable embedding service
    # here means a cache miss, not a request failure — the full pipeline below handles the same
    # outage with its own graceful degradation (retrieval_unavailable path).
    query_embedding: list[float] | None = None
    try:
        query_embedding = await asyncio.to_thread(embedding_service.embed_one_dense, body.question)
        cached = semantic_cache.get(query_embedding, body.file_ids)
        if cached is not None:
            return QueryResponse(
                answer=cached.answer,
                citations=cached.citations,
                routed_docs=cached.routed_docs,
                partial_answer=cached.partial_answer,
                debug=None,  # cache hits don't have a debug trace
                token_summary=TokenSummary(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    llm_calls_count=0,
                    estimated_cost_usd=0.0,
                ),
            )
    except Exception:
        pass  # embedding failure → cache miss, proceed normally

    # The graph's node functions call synchronous service clients (httpx, pymongo, qdrant-client);
    # running the whole invocation in a thread keeps this route non-blocking for the event loop,
    # same pattern as the ingestion pipeline.
    final_state, graph_path = await asyncio.to_thread(run_with_trace, retrieval_graph, initial_state)

    debug_info = build_debug_info(final_state, graph_path)
    response = QueryResponse(
        answer=final_state.answer,
        citations=final_state.citations,
        routed_docs=_routed_docs(final_state),
        partial_answer=final_state.partial_answer,
        debug=debug_info,
        token_summary=debug_info.token_summary if debug_info else None,
    )

    # --- Semantic cache store ---
    # Only cache successful (non-empty, non-error) answers. partial_answer results are cached
    # too — a repeated query won't do better on subsequent hops and would just burn tokens.
    if query_embedding is not None and final_state.answer and not final_state.retrieval_unavailable:
        try:
            semantic_cache.put(
                query_embedding=query_embedding,
                file_ids=body.file_ids,
                answer=final_state.answer,
                citations=final_state.citations,
                partial_answer=final_state.partial_answer,
                routed_docs=_routed_docs(final_state),
            )
        except Exception:
            pass  # cache write failure is never fatal

    return response


@router.post("/v1/query/stream")
async def query_stream(
    body: QueryRequest,
    x_llm_key: str = Header(..., alias="X-LLM-Key"),
    x_llm_model: str | None = Header(None, alias="X-LLM-Model"),
) -> StreamingResponse:
    llm_model = x_llm_model or settings.default_llm_model
    initial_state = _build_initial_state(body, x_llm_key, llm_model)

    async def event_stream():
        # The whole body is one try/except, not just the streaming call below: once a
        # StreamingResponse has sent its 200 and started the body, there is no way to turn a
        # later exception into a normal HTTP error response — the ASGI server just aborts the
        # connection, which a browser reports as a bare, unhelpful network error (confirmed live:
        # an invalid API key raised inside the *pre-generation* graph phase — classify_plan's LLM
        # call — did exactly this before this fix, because only the final streaming call had a
        # try/except). Every failure from here on must become a clean `event: error` frame instead.
        try:
            # --- Semantic cache lookup (streaming path) ---
            # A cache hit emits a single `done` frame immediately — no tokens stream, but the
            # client receives the same SSE event shape it would after a real stream, so it can
            # handle both paths identically. Cache miss falls through to normal generation.
            query_embedding: list[float] | None = None
            try:
                query_embedding = await asyncio.to_thread(embedding_service.embed_one_dense, body.question)
                cached = semantic_cache.get(query_embedding, body.file_ids)
                if cached is not None:
                    payload = QueryResponse(
                        answer=cached.answer,
                        citations=cached.citations,
                        routed_docs=cached.routed_docs,
                        partial_answer=cached.partial_answer,
                        debug=None,
                        token_summary=TokenSummary(
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            llm_calls_count=0,
                            estimated_cost_usd=0.0,
                        ),
                    )
                    yield f"event: done\ndata: {payload.model_dump_json()}\n\n"
                    return
            except Exception:
                pass  # embedding failure → cache miss, proceed normally

            # Everything up to generation runs exactly like the non-streaming path, just against
            # the pre-generation graph variant (graph.py) — no node's own behavior differs here.
            pre_state, graph_path = await asyncio.to_thread(run_with_trace, retrieval_graph_pre_generation, initial_state)

            if pre_state.retrieval_unavailable or not pre_state.context_blocks:
                # Mirrors generate_cite_node's own fallbacks exactly (no LLM call to stream from
                # when there's nothing to hallucinate from either way) — retrieval_unavailable gets
                # its own message, never NOT_FOUND_ANSWER, since an outage isn't "nothing relevant"
                # (see generate_cite.py's module docstring).
                answer = RETRIEVAL_UNAVAILABLE_ANSWER if pre_state.retrieval_unavailable else NOT_FOUND_ANSWER
                debug = build_debug_info(pre_state, [*graph_path, "generate_cite"])
                payload = QueryResponse(
                    answer=answer,
                    citations=[],
                    routed_docs=_routed_docs(pre_state),
                    partial_answer=pre_state.partial_answer,
                    debug=debug,
                    token_summary=debug.token_summary if debug else None,
                )
                yield f"event: done\ndata: {payload.model_dump_json()}\n\n"
                return

            prompt = build_grounded_answer_prompt(pre_state)
            accumulated = ""
            async for chunk in llm_client.stream_text(prompt=prompt, api_key=x_llm_key, model=llm_model):
                accumulated += chunk
                yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"

            # Citation markers can't be safely resolved from partial text mid-stream (a bracket
            # like "[doc-id:1:abc" with no closing "]" yet isn't a match) — resolved once, here,
            # against the full accumulated text, using the same trusted-lookup logic the blocking
            # node uses.
            cleaned_answer, citations = resolve_citations(accumulated, pre_state.context_blocks)
            gen_prompt_tokens = max(1, len(prompt) // 4)
            gen_completion_tokens = max(1, len(accumulated) // 4)
            try:
                import tiktoken
                enc = tiktoken.get_encoding(settings.token_count_encoding)
                gen_prompt_tokens = len(enc.encode(prompt))
                gen_completion_tokens = len(enc.encode(accumulated))
            except Exception:
                pass

            final_state = pre_state.model_copy(
                update={
                    "answer": cleaned_answer,
                    "citations": citations,
                    "llm_calls": [
                        *pre_state.llm_calls,
                        LLMCallRecord(
                            node="generate_cite",
                            prompt=prompt,
                            response=accumulated,
                            prompt_tokens=gen_prompt_tokens,
                            completion_tokens=gen_completion_tokens,
                        ),
                    ],
                    "token_usage_prompt": pre_state.token_usage_prompt + gen_prompt_tokens,
                    "token_usage_completion": pre_state.token_usage_completion + gen_completion_tokens,
                }
            )
            debug = build_debug_info(final_state, [*graph_path, "generate_cite"])
            payload = QueryResponse(
                answer=cleaned_answer,
                citations=citations,
                routed_docs=_routed_docs(final_state),
                partial_answer=final_state.partial_answer,
                debug=debug,
                token_summary=debug.token_summary if debug else None,
            )
            yield f"event: done\ndata: {payload.model_dump_json()}\n\n"

            # --- Semantic cache store (streaming path) ---
            if query_embedding is not None and cleaned_answer and not pre_state.retrieval_unavailable:
                try:
                    semantic_cache.put(
                        query_embedding=query_embedding,
                        file_ids=body.file_ids,
                        answer=cleaned_answer,
                        citations=citations,
                        partial_answer=final_state.partial_answer,
                        routed_docs=_routed_docs(final_state),
                    )
                except Exception:
                    pass  # cache write failure is never fatal

        except Exception as e:
            # Same redaction guarantee as llm_client's own errors (the key is already stripped
            # before it ever reaches here) — nothing further to sanitize before showing this to
            # the caller.
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

