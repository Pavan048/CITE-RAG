"""PRD Section 3: sufficiency_check_node (Reason).

Checks the *accumulated* evidence_pool against the *original* question — never the current
sub-question, since by the time this runs the pool may contain evidence gathered for several
different sub-questions and the thing that actually needs answering has not changed.

Also owns the hop-cap decision (Section 3: "on reaching the cap without sufficiency, generate from
whatever's in evidence_pool and set partial_answer: true"). That check is deliberately colocated
here rather than in a separate graph-routing function: whether to force a stop is a judgment about
*this* sufficiency result (insufficient, but out of hops), not an independent concern, and this
node is the only place that already has both `hop_count` and the sufficiency verdict in hand.

Only calls the sub-question rewrite prompt when the loop is actually going to continue — producing
a next sub-question that will never be used (because the cap was just hit) would be a wasted LLM
call.
"""

from __future__ import annotations

from app.config import settings
from app.extensibility.prompt_provider import prompt_provider
from app.retrieval.json_utils import parse_json_response
from app.retrieval.state import LLMCallRecord, RetrievalState
from app.services.llm_client import llm_client


def sufficiency_check_node(state: RetrievalState) -> dict:
    prompt = prompt_provider.get(
        "retrieval.sufficiency_check",
        question=state.question,
        sub_question_history=state.sub_question_history,
        evidence_blocks=state.evidence_pool,
    )
    raw = llm_client.complete_text(prompt=prompt, api_key=state.llm_api_key, model=state.llm_model, json_mode=True)
    parsed = parse_json_response(raw)
    is_sufficient = bool(parsed.get("sufficient", False))
    missing = parsed.get("missing")
    trace = [LLMCallRecord(node="sufficiency_check", prompt=prompt, response=raw)]

    if is_sufficient:
        return {"is_sufficient": True, "missing_evidence": None, "llm_calls": trace}

    if state.hop_count >= settings.hop_cap:
        return {"is_sufficient": False, "partial_answer": True, "missing_evidence": missing, "llm_calls": trace}

    rewrite_prompt = prompt_provider.get(
        "retrieval.rewrite_subquestion", question=state.question, sub_question_history=state.sub_question_history, missing=missing
    )
    raw_rewrite = llm_client.complete_text(prompt=rewrite_prompt, api_key=state.llm_api_key, model=state.llm_model, json_mode=True)
    next_sub_question = parse_json_response(raw_rewrite)["next_sub_question"]
    trace.append(LLMCallRecord(node="rewrite_subquestion", prompt=rewrite_prompt, response=raw_rewrite))

    return {
        "is_sufficient": False,
        "missing_evidence": missing,
        "current_sub_question": next_sub_question,
        "sub_question_history": [next_sub_question],
        "llm_calls": trace,
    }
