"""Test doubles. `FakeLLM` is a scripted `LLMClient` that records every call."""

from __future__ import annotations

import copy
import json
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from context_gateway.core.summary import SUMMARY_SYSTEM_PROMPT
from context_gateway.llm.base import LLMError


@dataclass
class Call:
    model: str
    messages: list[dict[str, Any]]
    params: dict[str, Any]
    stream: bool

    @property
    def is_summary(self) -> bool:
        first = self.messages[0] if self.messages else {}
        return str(first.get("content", "")).startswith(SUMMARY_SYSTEM_PROMPT[:60])


@dataclass
class _Reply:
    text: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: LLMError | None = None
    error_when: str = "call"  # "call" or "first_chunk"
    fail_after_chunks: int | None = None


def _chunks_of(text: str, size: int = 4) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


class FakeLLM:
    """Scripted replies, falling back to a default. Summary prompts get their own reply."""

    def __init__(
        self, default_reply: str = "fake reply", summary_text: str = "Fake summary."
    ) -> None:
        self.default_reply = default_reply
        self.summary_text = summary_text
        self.calls: list[Call] = []
        self._script: deque[_Reply] = deque()
        self._summary_script: deque[str] = deque()
        self.closed = False

    # scripting ------------------------------------------------------------------------
    def queue_text(self, text: str, *, fail_after_chunks: int | None = None) -> None:
        self._script.append(_Reply(text=text, fail_after_chunks=fail_after_chunks))

    def queue_tool_call(
        self, name: str, args: dict[str, Any], *, call_id: str = "call_1", text: str | None = None
    ) -> None:
        call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        self._script.append(_Reply(text=text, tool_calls=[call]))

    def queue_error(self, status: int | None, message: str, *, when: str = "call") -> None:
        self._script.append(_Reply(error=LLMError(status, message), error_when=when))

    def queue_summary(self, text: str) -> None:
        self._summary_script.append(text)

    # inspection -----------------------------------------------------------------------
    @property
    def chat_calls(self) -> list[Call]:
        return [c for c in self.calls if not c.is_summary]

    @property
    def summary_calls(self) -> list[Call]:
        return [c for c in self.calls if c.is_summary]

    # LLMClient ------------------------------------------------------------------------
    async def complete(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        call = self._record(model, messages, params, stream=False)
        if call.is_summary:
            text = self._summary_script.popleft() if self._summary_script else self.summary_text
            return self._completion(model, _Reply(text=text))
        reply = self._next_reply()
        if reply.error:
            raise reply.error
        return self._completion(model, reply)

    async def stream(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        call = self._record(model, messages, params, stream=True)
        if call.is_summary:
            text = self._summary_script.popleft() if self._summary_script else self.summary_text
            reply = _Reply(text=text)
        else:
            reply = self._next_reply()
        if reply.error and reply.error_when == "call":
            raise reply.error
        return self._chunk_iter(model, reply)

    async def close(self) -> None:
        self.closed = True

    # internals ------------------------------------------------------------------------
    def _record(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any], *, stream: bool
    ) -> Call:
        call = Call(model, copy.deepcopy(messages), copy.deepcopy(params), stream)
        self.calls.append(call)
        return call

    def _next_reply(self) -> _Reply:
        return self._script.popleft() if self._script else _Reply(text=self.default_reply)

    @staticmethod
    def _completion(model: str, reply: _Reply) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            message["tool_calls"] = reply.tool_calls
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if reply.tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def _chunk_iter(self, model: str, reply: _Reply) -> AsyncIterator[dict[str, Any]]:
        base = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        emitted = 0

        def check_failure() -> None:
            if reply.fail_after_chunks is not None and emitted >= reply.fail_after_chunks:
                raise LLMError(502, "fake mid-stream failure")

        if reply.error and reply.error_when == "first_chunk":
            raise reply.error

        yield chunk({"role": "assistant", "content": ""})
        emitted += 1
        for piece in _chunks_of(reply.text) if reply.text else []:
            check_failure()
            yield chunk({"content": piece})
            emitted += 1
        for index, call in enumerate(reply.tool_calls):
            check_failure()
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["function"]["name"], "arguments": ""},
                        }
                    ]
                }
            )
            emitted += 1
            for piece in _chunks_of(call["function"]["arguments"], 5):
                check_failure()
                yield chunk({"tool_calls": [{"index": index, "function": {"arguments": piece}}]})
                emitted += 1
        check_failure()
        yield chunk({}, "tool_calls" if reply.tool_calls else "stop")
