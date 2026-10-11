from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimates cost in USD based on model identifier and token counts."""
    m = model.lower()
    if "gpt-4o-mini" in m:
        rate_in, rate_out = 0.15 / 1_000_000, 0.60 / 1_000_000
    elif "gpt-4o" in m:
        rate_in, rate_out = 2.50 / 1_000_000, 10.00 / 1_000_000
    elif "claude-3-5-sonnet" in m or "claude-sonnet" in m:
        rate_in, rate_out = 3.00 / 1_000_000, 15.00 / 1_000_000
    elif "haiku" in m:
        rate_in, rate_out = 0.25 / 1_000_000, 1.25 / 1_000_000
    elif "gpt-3.5" in m:
        rate_in, rate_out = 0.50 / 1_000_000, 1.50 / 1_000_000
    else:
        # Generic default estimate
        rate_in, rate_out = 0.50 / 1_000_000, 1.50 / 1_000_000

    cost = (prompt_tokens * rate_in) + (completion_tokens * rate_out)
    return round(cost, 6)
