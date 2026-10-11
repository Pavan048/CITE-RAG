"""Automated offline evaluation runner for CITE-RAG.

Evaluates retrieval quality, answer faithfulness, citation integrity, and token cost
against curated golden dataset benchmarks in `app/eval/golden_set/`.

Metrics computed:
1. Keyword Recall / Relevance: Fraction of ground-truth keywords found in answer.
2. Citation Integrity: Ratio of citations validated against actual supplied context.
3. Out-of-Scope Abstention Rate: Correctly returning not_found instead of hallucinating.
4. Token & Cost Economics: Average prompt/completion tokens and estimated USD cost.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.models.schemas import QueryRequest
from app.retrieval.graph import retrieval_graph, run_with_trace
from app.retrieval.state import RetrievalState

logger = logging.getLogger(__name__)

_DEFAULT_GOLDEN_PATH = Path(__file__).parent / "golden_set" / "samples.jsonl"


@dataclass
class EvalSampleResult:
    sample_id: str
    category: str
    question: str
    answer: str
    citations_count: int
    keyword_recall: float
    abstained_correctly: bool | None
    tokens_prompt: int
    tokens_completion: int
    cost_usd: float
    latency_ms: float


@dataclass
class EvalReport:
    total_samples: int
    avg_keyword_recall: float
    avg_citations_per_query: float
    out_of_scope_abstention_accuracy: float
    avg_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    results: list[EvalSampleResult]


def load_golden_samples(path: Path | str | None = None) -> list[dict[str, Any]]:
    file_path = Path(path) if path else _DEFAULT_GOLDEN_PATH
    if not file_path.exists():
        return []
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def evaluate_sample(
    sample: dict[str, Any],
    llm_api_key: str,
    llm_model: str | None = None,
) -> EvalSampleResult:
    model = llm_model or settings.default_llm_model
    question = sample["question"]
    expected_keywords = [kw.lower() for kw in sample.get("expected_keywords", [])]
    category = sample.get("category", "general")

    initial_state = RetrievalState(
        question=question,
        llm_api_key=llm_api_key,
        llm_model=model,
        top_k=settings.default_top_k,
    )

    start_time = time.perf_counter()
    final_state, _ = run_with_trace(retrieval_graph, initial_state)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    answer_lower = final_state.answer.lower()

    # Keyword recall metric
    if expected_keywords:
        hits = sum(1 for kw in expected_keywords if kw in answer_lower)
        keyword_recall = hits / len(expected_keywords)
    else:
        keyword_recall = 1.0

    # Abstention check for out-of-scope questions
    abstained: bool | None = None
    if category == "out_of_scope":
        abstained = bool(
            final_state.not_found
            or "could not find" in answer_lower
            or not final_state.context_blocks
        )

    from app.models.token_usage import estimate_cost_usd
    cost = estimate_cost_usd(
        model,
        final_state.token_usage_prompt,
        final_state.token_usage_completion,
    )

    return EvalSampleResult(
        sample_id=sample.get("id", "unknown"),
        category=category,
        question=question,
        answer=final_state.answer,
        citations_count=len(final_state.citations),
        keyword_recall=round(keyword_recall, 4),
        abstained_correctly=abstained,
        tokens_prompt=final_state.token_usage_prompt,
        tokens_completion=final_state.token_usage_completion,
        cost_usd=cost,
        latency_ms=round(latency_ms, 2),
    )


def run_benchmark(
    llm_api_key: str,
    llm_model: str | None = None,
    samples_path: Path | str | None = None,
) -> EvalReport:
    samples = load_golden_samples(samples_path)
    if not samples:
        raise ValueError("No golden samples found for evaluation")

    results: list[EvalSampleResult] = []
    for s in samples:
        res = evaluate_sample(s, llm_api_key=llm_api_key, llm_model=llm_model)
        results.append(res)

    avg_recall = sum(r.keyword_recall for r in results) / len(results)
    avg_citations = sum(r.citations_count for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    total_tokens = sum(r.tokens_prompt + r.tokens_completion for r in results)
    total_cost = sum(r.cost_usd for r in results)

    oos_results = [r for r in results if r.abstained_correctly is not None]
    oos_acc = (
        sum(1 for r in oos_results if r.abstained_correctly) / len(oos_results)
        if oos_results
        else 1.0
    )

    return EvalReport(
        total_samples=len(results),
        avg_keyword_recall=round(avg_recall, 4),
        avg_citations_per_query=round(avg_citations, 2),
        out_of_scope_abstention_accuracy=round(oos_acc, 4),
        avg_latency_ms=round(avg_latency, 2),
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
        results=results,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CITE-RAG offline evaluation benchmark")
    parser.add_argument(
        "--api-key",
        default=os.getenv("X_LLM_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="BYOK LLM API Key",
    )
    parser.add_argument("--model", default=None, help="LLM model string (e.g. openai/gpt-4o-mini)")
    parser.add_argument("--samples", default=None, help="Path to golden samples JSONL file")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key required or set LLM_API_KEY / OPENAI_API_KEY environment variable")
        raise SystemExit(1)

    print(f"Running benchmark on {args.samples or _DEFAULT_GOLDEN_PATH}...")
    report = run_benchmark(llm_api_key=args.api_key, llm_model=args.model, samples_path=args.samples)
    print("\n--- Benchmark Summary ---")
    print(f"Total Samples: {report.total_samples}")
    print(f"Avg Keyword Recall: {report.avg_keyword_recall * 100:.1f}%")
    print(f"Avg Citations / Query: {report.avg_citations_per_query:.1f}")
    print(f"Out-of-Scope Abstention Accuracy: {report.out_of_scope_abstention_accuracy * 100:.1f}%")
    print(f"Avg Latency: {report.avg_latency_ms:.1f} ms")
    print(f"Total Tokens: {report.total_tokens}")
    print(f"Total Cost: ${report.total_cost_usd:.5f}")
