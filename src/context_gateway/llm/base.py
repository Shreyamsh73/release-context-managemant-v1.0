"""LLM client protocol and error type."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol


class LLMError(Exception):
    """A provider call failed. `status_code` is the upstream HTTP status when known."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class LLMClient(Protocol):
    """Sends OpenAI-format chat requests to a provider."""

    async def complete(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Return an OpenAI chat completion as a dict."""
        ...

    async def stream(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Return an async iterator of OpenAI chunk dicts."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
