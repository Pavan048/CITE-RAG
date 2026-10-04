"""PRD Section 3: classify_plan_node (Reason). Runs once per query, before any retrieval — decides
single-hop vs multi-hop and, for multi-hop, produces the first sub-question. Single-hop queries
skip the loop machinery entirely (Section 3: "to keep latency low for the majority case"), so this
node's only job is the routing decision itself, not any retrieval.
"""

from __future__ import annotations

from app.extensibility.prompt_provider import prompt_provider
from app.retrieval.json_utils import parse_json_response
from app.retrieval.state import LLMCallRecord, RetrievalState
from app.services.llm_client import llm_client


def classify_plan_node(state: RetrievalState) -> dict:
    prompt = prompt_provider.get("retrieval.classify_plan", question=state.question)
    raw = llm_client.complete_text(prompt=prompt, api_key=state.llm_api_key, model=state.llm_model, json_mode=True)
    parsed = parse_json_response(raw)
    trace = [LLMCallRecord(node="classify_plan", prompt=prompt, response=raw)]

    is_multi_hop = bool(parsed.get("is_multi_hop", False))
    if not is_multi_hop:
        return {"is_multi_hop": False, "llm_calls": trace}

    first_sub_question = parsed.get("first_sub_question") or state.question
    return {
        "is_multi_hop": True,
        "current_sub_question": first_sub_question,
        "sub_question_history": [first_sub_question],
        "llm_calls": trace,
    }
