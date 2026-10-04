"""PRD Section 3: the LangGraph DAG and the multi-hop loop's wiring.

accumulate_node runs on every path, including single-hop — the PRD's diagram draws the single-hop
line straight from route_retrieve_node to rerank_node, but that is a diagram simplification for the
common/fast path, not a literal instruction to skip a real node: rerank_node reads `evidence_pool`,
and accumulate_node is evidence_pool's only writer, so something has to move route_retrieve_node's
output into the pool regardless of hop count. For a single hop that merge is trivial (nothing to
dedupe against yet), which is exactly why the diagram doesn't bother drawing it.

The three conditional edges below are where the actual branching lives — everything else is a
straight line:
  - after route_retrieve: `not_found` short-circuits straight to generation, skipping
    accumulate/sufficiency/rerank/budget entirely (Section 3).
  - after accumulate: single-hop skips the loop machinery (sufficiency_check, the loop-back edge,
    the hop cap) entirely and goes straight to reranking (Section 3).
  - after sufficiency_check: sufficient, or the hop cap forcing a stop, both lead to reranking;
    otherwise the loop goes back to route_retrieve_node with the new sub-question.

`build_graph(include_generation=False)` produces a second variant that ends at context_budget_node
instead of generate_cite_node — used only by api/query.py's streaming endpoint, which needs to run
everything *up to* generation with LangGraph's normal blocking `.invoke()`, then hand off to
`llm_client.stream_text` for the one call that actually needs to stream token-by-token (see
generate_cite.py's module docstring for why that call's prompt-building/citation-validation logic
is shared rather than duplicated). No node's own logic changes between the two variants — only
which edges exist after route_retrieve's not_found check and after context_budget.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.retrieval.nodes.accumulate import accumulate_node
from app.retrieval.nodes.classify_plan import classify_plan_node
from app.retrieval.nodes.context_budget import context_budget_node
from app.retrieval.nodes.generate_cite import generate_cite_node
from app.retrieval.nodes.rerank import rerank_node
from app.retrieval.nodes.route_retrieve import route_retrieve_node
from app.retrieval.nodes.sufficiency_check import sufficiency_check_node
from app.retrieval.state import RetrievalState


def _after_accumulate(state: RetrievalState) -> str:
    return "sufficiency_check" if state.is_multi_hop else "rerank"


def _after_sufficiency_check(state: RetrievalState) -> str:
    return "rerank" if (state.is_sufficient or state.partial_answer) else "route_retrieve"


def build_graph(include_generation: bool = True) -> CompiledStateGraph:
    graph = StateGraph(RetrievalState)

    graph.add_node("classify_plan", classify_plan_node)
    graph.add_node("route_retrieve", route_retrieve_node)
    graph.add_node("accumulate", accumulate_node)
    graph.add_node("sufficiency_check", sufficiency_check_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("context_budget", context_budget_node)

    not_found_target = "generate_cite" if include_generation else END
    after_budget_target = "generate_cite" if include_generation else END

    def _after_route_retrieve(state: RetrievalState) -> str:
        # Both short-circuits skip straight to generation (or END, in the pre-generation graph
        # variant) — neither has anything for accumulate/sufficiency/rerank/budget to work with.
        return not_found_target if (state.not_found or state.retrieval_unavailable) else "accumulate"

    graph.add_edge(START, "classify_plan")
    graph.add_edge("classify_plan", "route_retrieve")
    graph.add_conditional_edges("route_retrieve", _after_route_retrieve, {not_found_target: not_found_target, "accumulate": "accumulate"})
    graph.add_conditional_edges("accumulate", _after_accumulate, {"sufficiency_check": "sufficiency_check", "rerank": "rerank"})
    graph.add_conditional_edges(
        "sufficiency_check", _after_sufficiency_check, {"rerank": "rerank", "route_retrieve": "route_retrieve"}
    )
    graph.add_edge("rerank", "context_budget")

    if include_generation:
        graph.add_node("generate_cite", generate_cite_node)
        graph.add_edge("context_budget", "generate_cite")
        graph.add_edge("generate_cite", END)
    else:
        graph.add_edge("context_budget", after_budget_target)

    return graph.compile()


retrieval_graph = build_graph()
# Ends at context_budget_node — see module docstring. Used only by the streaming query endpoint.
retrieval_graph_pre_generation = build_graph(include_generation=False)


def run_with_trace(graph: CompiledStateGraph, initial_state: RetrievalState) -> tuple[RetrievalState, list[str]]:
    """Runs `graph` to completion exactly like `.invoke()` would, but also returns the literal,
    ordered sequence of node names that executed — for the `debug.graph_path` field in
    api/query.py's response (PRD Section 4 has no such field; this is additive, for the UI's debug
    panel). Built from LangGraph's own `stream_mode=["updates", "values"]`, which yields a
    `(node_name -> partial_update)` dict after every node and the full merged state after every
    step, rather than reconstructed after the fact from state fields — this is authoritative
    because it's what LangGraph itself actually executed, not a guess at what the fixed edges
    "should" imply for a given hop_count.
    """
    graph_path: list[str] = []
    final_values: dict = {}
    for mode, chunk in graph.stream(initial_state, stream_mode=["updates", "values"]):
        if mode == "updates":
            graph_path.extend(chunk.keys())
        else:
            final_values = chunk
    return RetrievalState.model_validate(final_values), graph_path
