"""The only sanctioned way to get prompt text into an LLM call (PRD Section 7).

This exists solely so a future versioned prompt library can replace file-backed lookup with a
DB-backed one without touching any call site — every node and ingestion stage calls
`PromptProvider.get(key, **vars)` and never sees a raw string. Do not build that future backend now
(PRD Section 0); this is a thin file-based implementation on purpose.

Template rendering uses Jinja2 rather than `str.format`: several prompts instruct the model to
emit JSON, and JSON's literal `{}` would collide with `str.format`'s placeholder syntax. Jinja2's
`{{ }}` delimiter doesn't have that collision.

Deliberately uncached — every `get()` call re-reads the file from disk. This is unrelated to the
"semantic caching" the PRD excludes (Section 0), which is about caching query results, but a
template cache is one more thing that could silently mask an edit during iteration, and prompt
files are small enough that re-reading costs nothing worth trading that away for.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptNotFoundError(FileNotFoundError):
    pass


class PromptProvider:
    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        self._dir = Path(prompts_dir) if prompts_dir else _DEFAULT_PROMPTS_DIR

    def get(self, key: str, **vars: object) -> str:
        """Render the prompt registered under `key` (e.g. "generation.grounded_answer") with the
        given template variables. `key` maps directly to `prompts/<key with '.' -> '/'>.md`.
        """
        path = self._dir / f"{key.replace('.', '/')}.md"
        if not path.is_file():
            raise PromptNotFoundError(f"No prompt file registered for key '{key}' (looked at {path})")
        template_text = path.read_text(encoding="utf-8")
        return Template(template_text).render(**vars)


prompt_provider = PromptProvider()
