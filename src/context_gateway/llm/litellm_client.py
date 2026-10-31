"""LiteLLM-backed client. LiteLLM is imported when the client is built, never at module load."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from context_gateway.llm.base import LLMError

logger = logging.getLogger(__name__)


def to_llm_error(exc: Exception) -> LLMError:
    """Wrap a LiteLLM exception, keeping the upstream HTTP status when it has one."""
    status = getattr(exc, "status_code", None)
    return LLMError(status if isinstance(status, int) else None, str(exc))


class LiteLLMClient:
    """Sends OpenAI-format requests to any LiteLLM provider."""

    def __init__(self) -> None:
        import litellm  # lazy: importing litellm takes about 8 seconds

        litellm.suppress_debug_info = True  # no "Provider List" banners on errors
        self._litellm = litellm

    async def complete(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Return an OpenAI chat completion as a dict."""
        try:
            response = await self._litellm.acompletion(
                model=model, messages=messages, drop_params=True, **params
            )
            return response.model_dump()
        except Exception as exc:
            raise to_llm_error(exc) from exc

    async def stream(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Open a streaming call and return an iterator of chunk dicts."""
        try:
            response = await self._litellm.acompletion(
                model=model, messages=messages, drop_params=True, stream=True, **params
            )
        except Exception as exc:
            raise to_llm_error(exc) from exc
        return self._chunks(response)

    @staticmethod
    async def _chunks(response: Any) -> AsyncIterator[dict[str, Any]]:
        try:
            async for chunk in response:
                yield chunk.model_dump(exclude_none=True)
        except Exception as exc:
            raise to_llm_error(exc) from exc

    async def close(self) -> None:
        """Nothing to release; LiteLLM manages its own HTTP clients."""
