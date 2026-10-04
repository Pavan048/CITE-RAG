"""Thin client for the user's BYOK LLM (PRD Section 1, Section 7), via LiteLLM so this system isn't
locked to one vendor's SDK/API shape — a `model` string like "openai/gpt-4o-mini" or
"anthropic/claude-sonnet-5-20260101" selects the provider, and `api_key` is passed per-call rather
than via a global/env var, since it is per-request BYOK data, not process-wide configuration.

Section 8: `X-LLM-Key` (surfaced here as `api_key`) must never be persisted, logged, or included in
an error trace. This function receives it, uses it for exactly one call, and lets it go out of
scope — but a provider's own exception text can echo the key back verbatim: confirmed live against
a real OpenAI key during development, an invalid-key `AuthenticationError`'s message reads
"Incorrect API key provided: <the actual key>...". Every caller downstream (the ingestion pipeline,
the retrieval nodes) only ever does `str(e)` on failures for storage/logging, so if that string
still contains the key, it ends up in a Mongo `JobError` or a server log despite every intention
not to. Both methods below therefore catch every exception, strip the key out of its message, and
re-raise a fresh exception type with `from None` — `from None` is not stylistic here: without it,
Python's implicit exception chaining (`__context__`) keeps the *original* exception attached, and
anything that later logs this exception "with traceback" (uvicorn's default error logging does)
would print the unredacted original right back.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from typing import AsyncIterator

import litellm

from app.config import settings


class LLMCallError(Exception):
    pass


_STREAM_DONE = object()


def _redact(message: str, secret: str) -> str:
    return message.replace(secret, "[REDACTED]") if secret else message


class LLMClient:
    def complete_text(
        self,
        *,
        prompt: str,
        api_key: str,
        model: str,
        json_mode: bool = False,
    ) -> str:
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                timeout=settings.llm_request_timeout_s,
                response_format={"type": "json_object"} if json_mode else None,
            )
        except Exception as e:
            raise LLMCallError(_redact(str(e), api_key)) from None
        return response.choices[0].message.content or ""

    async def stream_text(
        self,
        *,
        prompt: str,
        api_key: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Streaming counterpart to `complete_text`, for api/query.py's SSE endpoint. litellm's
        `stream=True` response is a *synchronous* iterator that blocks on each `next()` while
        waiting for network data — iterating it directly inside this async function would block
        the whole event loop, stalling every other in-flight request. Instead a background thread
        drives that blocking iterator and pushes each chunk into an `asyncio.Queue`; this
        coroutine only ever awaits the queue, never the network call itself.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                stream = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=api_key,
                    timeout=settings.llm_request_timeout_s,
                    stream=True,
                )
                for part in stream:
                    delta = part.choices[0].delta.content
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as e:
                error = LLMCallError(_redact(str(e), api_key))
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

        threading.Thread(target=produce, daemon=True).start()

        while True:
            item = await queue.get()
            if item is _STREAM_DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def caption_image(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        api_key: str,
        model: str,
        mime_type: str = "image/png",
    ) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                        ],
                    }
                ],
                api_key=api_key,
                timeout=settings.llm_request_timeout_s,
            )
        except Exception as e:
            raise LLMCallError(_redact(str(e), api_key)) from None
        return response.choices[0].message.content or ""


llm_client = LLMClient()
