"""Streaming helpers: assemble a message from chunks and encode server-sent events."""

from __future__ import annotations

import json
from typing import Any

from context_gateway.core.types import Message


class StreamAccumulator:
    """Collects OpenAI-style streaming chunks into one assistant message."""

    def __init__(self) -> None:
        self._role: str | None = None
        self._text: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._function_call: dict[str, Any] | None = None
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.finished = False

    def add(self, chunk: dict[str, Any]) -> None:
        """Feed one chunk."""
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if (choice.get("index") or 0) != 0:
                continue
            self._add_choice(choice)

    def _add_choice(self, choice: dict[str, Any]) -> None:
        delta = choice.get("delta") or {}
        if delta.get("role"):
            self._role = delta["role"]
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._text.append(content)
        for position, call in enumerate(delta.get("tool_calls") or []):
            self._merge_tool_call(position, call)
        if delta.get("function_call"):
            self._merge_function_call(delta["function_call"])
        if choice.get("finish_reason") is not None:
            self.finish_reason = choice["finish_reason"]
            self.finished = True

    def _merge_tool_call(self, position: int, call: dict[str, Any]) -> None:
        index = call.get("index")
        index = position if index is None else index
        merged = self._tool_calls.setdefault(
            index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if call.get("id"):
            merged["id"] = call["id"]
        if call.get("type"):
            merged["type"] = call["type"]
        function = call.get("function") or {}
        if function.get("name"):
            merged["function"]["name"] = function["name"]
        if function.get("arguments"):
            merged["function"]["arguments"] += function["arguments"]

    def _merge_function_call(self, call: dict[str, Any]) -> None:
        merged = self._function_call or {"name": "", "arguments": ""}
        if call.get("name"):
            merged["name"] = call["name"]
        if call.get("arguments"):
            merged["arguments"] += call["arguments"]
        self._function_call = merged

    def mark_done(self) -> None:
        """Mark the stream complete when the upstream ended without a finish reason."""
        self.finished = True

    @property
    def completed(self) -> bool:
        """True once a finish reason was seen or the upstream ended normally."""
        return self.finished

    @property
    def text(self) -> str:
        """All content text received so far."""
        return "".join(self._text)

    def message(self) -> Message:
        """Return the assembled assistant message. Empty keys are omitted."""
        text = self.text
        calls = [self._tool_calls[i] for i in sorted(self._tool_calls)]
        message: Message = {"role": "assistant"}
        if text:
            message["content"] = text
        elif calls or self._function_call:
            message["content"] = None
        if calls:
            message["tool_calls"] = calls
        if self._function_call:
            message["function_call"] = self._function_call
        return message


def sse_event(data: dict[str, Any] | str) -> bytes:
    """Encode one server-sent event."""
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return b"data: " + payload.encode("utf-8") + b"\n\n"


def sse_done() -> bytes:
    """The terminating event of an OpenAI stream."""
    return b"data: [DONE]\n\n"
