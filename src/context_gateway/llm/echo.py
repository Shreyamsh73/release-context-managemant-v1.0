"""Echo backend: a deterministic fake provider for demos and end-to-end checks."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from context_gateway.core.content import flatten_content
from context_gateway.core.modes import SYSTEM_ROLES
from context_gateway.core.summary import SUMMARY_SYSTEM_PROMPT
from context_gateway.core.tokens import HeuristicTokenCounter

_PREVIOUS_COUNT = re.compile(r"summary of (\d+) turns")


class EchoLLMClient:
    """Replies with a description of what it received. Never touches the network."""

    def __init__(self) -> None:
        self._counter = HeuristicTokenCounter()

    async def complete(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Return an OpenAI chat completion."""
        reply = self._reply(messages)
        prompt_tokens = self._counter.count(messages, model)
        completion_tokens = self._counter.count([{"role": "assistant", "content": reply}], model)
        return {
            "id": f"chatcmpl-echo-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def stream(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Return an async iterator of OpenAI chunks."""
        return self._chunks(model, self._reply(messages))

    async def _chunks(self, model: str, reply: str) -> AsyncIterator[dict[str, Any]]:
        base = {
            "id": f"chatcmpl-echo-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }
        for position, piece in enumerate(_split(reply)):
            delta: dict[str, Any] = {"content": piece}
            if position == 0:
                delta["role"] = "assistant"
            yield {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    async def close(self) -> None:
        """Nothing to release."""

    def _reply(self, messages: list[dict[str, Any]]) -> str:
        if _is_summary_request(messages):
            return self._summary_reply(messages)
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        text = flatten_content(last_user.get("content")) if last_user else ""
        non_system = [m for m in messages if m.get("role") not in SYSTEM_ROLES]
        has_summary = any(
            m.get("role") in SYSTEM_ROLES and "<summary>" in flatten_content(m.get("content"))
            for m in messages
        )
        return (
            f"echo: {text} | history_messages={max(len(non_system) - 1, 0)} "
            f"| summary={'yes' if has_summary else 'no'}"
        )

    @staticmethod
    def _summary_reply(messages: list[dict[str, Any]]) -> str:
        user_text = flatten_content(messages[1].get("content")) if len(messages) > 1 else ""
        existing, _, transcript = user_text.partition("New turns to merge:")
        previous = _PREVIOUS_COUNT.search(existing)
        turns = transcript.count("### Turn ") + (int(previous.group(1)) if previous else 0)
        return f"- summary of {turns} turns"


def _is_summary_request(messages: list[dict[str, Any]]) -> bool:
    if not messages or messages[0].get("role") != "system":
        return False
    return flatten_content(messages[0].get("content")).startswith(SUMMARY_SYSTEM_PROMPT[:60])


def _split(text: str) -> list[str]:
    """Split text into 3 to 5 pieces, longer text giving more pieces."""
    count = min(5, max(3, len(text) // 25))
    size = -(-len(text) // count)
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]
