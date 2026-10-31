"""Small builders used across tests."""

from __future__ import annotations

from typing import Any

from context_gateway.config import Settings
from context_gateway.core.types import Turn


def make_settings(**overrides: Any) -> Settings:
    """Settings that ignore the environment and any .env file."""
    base: dict[str, Any] = {"store": "memory", "token_counter": "heuristic", "llm_backend": "echo"}
    base.update(overrides)
    return Settings(_env_file=None, **base)


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def system(text: str) -> dict[str, Any]:
    return {"role": "system", "content": text}


def make_turn(index: int, tokens: int, text: str | None = None) -> Turn:
    """A two-message turn with a fixed token count."""
    return Turn(
        index=index,
        messages=[user(text or f"q{index}"), assistant(f"a{index}")],
        tokens=tokens,
    )
