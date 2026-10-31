"""Pieces shared by the chat and memory routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from context_gateway.core.types import ContextReport, Scope
from context_gateway.errors import GatewayError

MEMORY_HEADERS = (
    "X-Memory-Mode",
    "X-Memory-App",
    "X-Memory-User",
    "X-Memory-Session",
    "X-Memory-History-Turns",
    "X-Memory-Dropped-Turns",
    "X-Memory-Summary",
    "X-Memory-Prompt-Tokens",
)

_NON_PASSTHROUGH = frozenset({"model", "messages", "stream", "user"})


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completions body. Unknown fields are kept and passed to the provider."""

    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    user: str | None = None

    def passthrough_params(self, *, streaming: bool) -> dict[str, Any]:
        """Fields forwarded to the provider unchanged."""
        params = {k: v for k, v in (self.model_extra or {}).items() if k not in _NON_PASSTHROUGH}
        if not streaming:
            params.pop("stream_options", None)
        return params


def format_validation_errors(errors: list[Any]) -> tuple[str, str | None]:
    """Turn pydantic or FastAPI error dicts into one short message and a param name."""
    pieces = []
    param: str | None = None
    for error in errors[:3]:
        location = [str(part) for part in error.get("loc", ()) if part != "body"]
        if param is None and location:
            param = location[0]
        where = ".".join(location)
        pieces.append(f"{where}: {error.get('msg', 'invalid value')}" if where else error["msg"])
    return "; ".join(pieces) or "Invalid request", param


async def read_json_object(request: Request) -> dict[str, Any]:
    """Read the request body as a JSON object, or raise a 400."""
    raw = await request.body()
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise GatewayError(400, "invalid_request", "Request body must be valid JSON") from None
    if not isinstance(data, dict):
        raise GatewayError(400, "invalid_request", "Request body must be a JSON object")
    return data


def parse_chat_request(data: dict[str, Any]) -> ChatCompletionRequest:
    """Validate the chat body, raising a 400 in the OpenAI error shape."""
    try:
        return ChatCompletionRequest.model_validate(data)
    except ValidationError as exc:
        message, param = format_validation_errors(exc.errors())
        raise GatewayError(400, "invalid_request", message, param=param) from None


def memory_headers(report: ContextReport, scope: Scope | None) -> dict[str, str]:
    """Headers describing what memory did for this request."""
    headers = {
        "X-Memory-Mode": report.mode,
        "X-Memory-History-Turns": str(report.history_turns),
        "X-Memory-Dropped-Turns": str(report.dropped_turns),
        "X-Memory-Summary": "used" if report.summary_used else "none",
        "X-Memory-Prompt-Tokens": str(report.prompt_tokens),
    }
    if scope is not None:
        headers["X-Memory-App"] = scope.app
        headers["X-Memory-User"] = scope.user
        headers["X-Memory-Session"] = scope.session
    return headers
