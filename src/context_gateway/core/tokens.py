"""Token counting. Counts are estimates; the budget leaves headroom."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from typing import Any, Protocol

from context_gateway.core.content import flatten_content

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    """Estimates the prompt tokens of a list of chat messages."""

    def count(self, messages: Sequence[dict[str, Any]], model: str | None) -> int:
        """Return the estimated token count."""
        ...


class HeuristicTokenCounter:
    """Deterministic estimate: 4 tokens per message plus one token per 4 characters."""

    def count(self, messages: Sequence[dict[str, Any]], model: str | None = None) -> int:
        """Return the estimated token count."""
        total = 0
        for message in messages:
            text = flatten_content(message.get("content"))
            tool_calls = message.get("tool_calls")
            if tool_calls:
                text += json.dumps(tool_calls, sort_keys=True, ensure_ascii=False, default=str)
            total += 4 + math.ceil(len(text) / 4)
        return total


class LiteLLMTokenCounter:
    """Uses LiteLLM's bundled tokenizers, and falls back to the heuristic on any error."""

    def __init__(self) -> None:
        import litellm  # lazy: importing litellm takes about 8 seconds

        self._litellm = litellm
        self._fallback = HeuristicTokenCounter()

    def count(self, messages: Sequence[dict[str, Any]], model: str | None = None) -> int:
        """Return the estimated token count."""
        try:
            return int(self._litellm.token_counter(model=model or "", messages=list(messages)))
        except Exception:
            logger.debug("litellm token_counter failed, using heuristic", exc_info=True)
            return self._fallback.count(messages, model)


def make_token_counter(kind: str) -> TokenCounter:
    """Build the counter named by `CTX_TOKEN_COUNTER`."""
    if kind == "heuristic":
        return HeuristicTokenCounter()
    return LiteLLMTokenCounter()
