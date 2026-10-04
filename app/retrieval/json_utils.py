"""Shared, tiny helper for parsing structured LLM output. Not a node itself (so it lives beside
`nodes/`, not inside it) — several nodes ask their prompt for a JSON object back and need the same
tolerant extraction, since providers sometimes wrap JSON in markdown fences despite instructions
asking for a bare object.
"""

from __future__ import annotations

import json
import re

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMJsonParseError(ValueError):
    pass


def parse_json_response(text: str) -> dict:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise LLMJsonParseError(f"No JSON object found in LLM response: {text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise LLMJsonParseError(f"Malformed JSON in LLM response: {text!r}") from e
