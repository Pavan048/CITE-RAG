"""Typed state for the retrieval graph (PRD Section 3).

A pydantic model, not a dict, per the code-quality bar — every hop reads and writes a fully typed
object instead of stringly-keyed state. LangGraph still requires node functions to *return* plain
dicts of partial updates (that's how its reducer/merge step works), but the object that flows
through the graph and that every node receives as input is this model.

Fields marked with an `Annotated[..., operator.add]` reducer accumulate across hops instead of
being overwritten — this is what makes `evidence_pool` a running pool ("never reset mid-query",
Section 3) rather than a per-hop scratch buffer, and is exactly the property that distinguishes the
multi-hop loop from a simple retry.
"""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from app.models.mongo_models import ContentType
from app.models.schemas import Citation


class EvidenceItem(BaseModel):
    """One retrieved-and-fused chunk candidate, carried in `evidence_pool`.

    Kept independent of `ChunkPayload` (models/mongo_models.py) on purpose — this is the retrieval
    loop's working shape, not a storage shape, and the two are free to diverge as either evolves.
    """

    chunk_id: str
    doc_id: str
    page_number: int
    section_title: str | None
    content_type: ContentType
    parent_block_id: str
    text: str
    fused_score: float
    source_sub_question: str
    rerank_score: float | None = None


class CitationOption(BaseModel):
    """One valid (chunk_id, page_number) the generator may cite for a given context block."""

    chunk_id: str
    page_number: int


class ContextBlock(BaseModel):
    """One block of the context handed to generate_cite_node: a parent block's full text, tagged
    with every surviving chunk that maps to it. Built by context_budget_node; see its module
    docstring for why blocks are deduped by parent_block_id but still carry per-chunk citation
    identity."""

    doc_id: str
    text: str
    citation_options: list[CitationOption]


class LLMCallRecord(BaseModel):
    """One LLM call made anywhere in the loop, for the debug trace exposed via api/query.py's
    `debug` field — not part of the PRD's core contract, added because there is otherwise no way
    to see *why* the loop made the decision it did (which prompt, which raw response) without
    reading server logs. `node` names the node that made the call (e.g. "classify_plan")."""

    node: str
    prompt: str
    response: str


class RetrievalState(BaseModel):
    # --- Query input, fixed for the lifetime of the request ---
    question: str
    llm_api_key: str
    llm_model: str
    # Defaults to None (rather than being a bare required field) because LangGraph's internal
    # state re-validation between steps can drop keys whose value is None from the dict it
    # re-validates against this schema — without a default, that turns into a spurious "field
    # required" error on every step, not just the first.
    file_ids: list[str] | None = None
    top_k: int

    # --- Routing/planning ---
    is_multi_hop: bool = False
    # The sub-question the node currently executing should act on. Distinct from
    # `sub_question_history` (an append-only log) because route_retrieve_node and
    # sufficiency_check_node both need "the one live question right now", not the whole history.
    current_sub_question: str | None = None
    sub_question_history: Annotated[list[str], operator.add] = Field(default_factory=list)

    # --- Evidence ---
    # Written only by route_retrieve_node, read only by accumulate_node — a transient handoff
    # field, overwritten each hop, so route_retrieve_node never has to know about dedup and
    # accumulate_node never has to know about retrieval.
    latest_hop_candidates: list[EvidenceItem] = Field(default_factory=list)
    # Accumulated across hops (Section 3: accumulate_node), "never reset mid-query". accumulate_node
    # is the sole writer — it dedupes `latest_hop_candidates` against what's already here before
    # returning the (usually smaller) delta that actually gets appended.
    evidence_pool: Annotated[list[EvidenceItem], operator.add] = Field(default_factory=list)
    # Populated whenever route_retrieve_node's document router actually ran, on any hop (Section
    # 4: "required for debugging a wrong routing decision, not optional polish"). Stays empty when
    # the caller pinned `file_ids` (routing is skipped, so there is nothing to report) — the API
    # layer (api/query.py) maps "empty because routing was skipped" to the `routed_docs: null` the
    # PRD wants; the graph state itself just needs a plain accumulable list, not an Optional, since
    # LangGraph's reducer can't add a delta onto `None`.
    routed_docs: Annotated[list[str], operator.add] = Field(default_factory=list)

    # --- Loop control ---
    # Accumulated, not overwritten — route_retrieve_node returns a delta of 1 each time it runs
    # within the multi-hop loop, so this reflects the count of completed hops, not the last hop's
    # own index.
    hop_count: Annotated[int, operator.add] = 0
    # Set by sufficiency_check_node; read by the graph's conditional edge to decide whether to
    # loop back for another hop or move on to reranking. Not one of the PRD's literally-listed
    # required fields, but required for the graph to route at all — sufficiency is a judgment the
    # loop must act on, not just record.
    is_sufficient: bool | None = None
    missing_evidence: str | None = None
    partial_answer: bool = False
    not_found: bool = False
    # Set by route_retrieve_node when the embedding service is unreachable on the *first* hop (no
    # evidence exists yet to fall back on). Deliberately distinct from `not_found`: "nothing
    # relevant in the knowledge base" and "the retrieval service is down" are different failure
    # modes, and generate_cite_node must not tell the user the former when the truth is the
    # latter. On a later hop, an embedding failure instead just contributes no evidence for that
    # hop (mirrors the empty-routing-result branch) rather than aborting a query that already has
    # evidence from earlier hops.
    retrieval_unavailable: bool = False

    # --- Post-loop: reranking and context budgeting ---
    # A separate, plain-overwrite field rather than another accumulated list: rerank_node needs to
    # set `rerank_score` on items already in `evidence_pool`, and `evidence_pool`'s reducer only
    # ever appends — it can't express "replace these entries with a scored version" without
    # duplicating them. `evidence_pool` stays the untouched accumulated record; this is the
    # sorted/trimmed derived view built from it once, after the loop ends.
    reranked_evidence: list[EvidenceItem] = Field(default_factory=list)
    context_blocks: list[ContextBlock] = Field(default_factory=list)

    # --- Debug trace (not part of the PRD's core state contract — see LLMCallRecord) ---
    llm_calls: Annotated[list[LLMCallRecord], operator.add] = Field(default_factory=list)

    # --- Output ---
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
