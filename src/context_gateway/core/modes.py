"""Client mode detection and request message validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from context_gateway.core.types import ClientMode, Message
from context_gateway.errors import GatewayError

SYSTEM_ROLES = frozenset({"system", "developer"})
_STATEFUL_ROLES = frozenset({"assistant", "tool", "function"})
_OVERRIDES = {"auto", "stateless", "stateful", "off"}


def split_leading_system(messages: Sequence[Message]) -> tuple[list[Message], list[Message]]:
    """Split off the longest prefix of system or developer messages."""
    end = 0
    for message in messages:
        if message.get("role") in SYSTEM_ROLES:
            end += 1
        else:
            break
    return list(messages[:end]), list(messages[end:])


def validate_messages(messages: Any) -> None:
    """Raise a 400 unless messages is a non-empty list of dicts with a string role."""
    if not isinstance(messages, list) or not messages:
        raise GatewayError(
            400, "invalid_request", "messages must be a non-empty list", param="messages"
        )
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise GatewayError(
                400,
                "invalid_request",
                f"messages[{index}] must be an object with a string role",
                param="messages",
            )


def detect_mode(messages: Sequence[Message], override: str | None = None) -> ClientMode:
    """Pick the client mode from the override value, or from the messages when it is auto."""
    value = (override or "").strip().lower()
    if value and value not in _OVERRIDES:
        raise GatewayError(
            400,
            "invalid_memory_mode",
            f"X-Memory-Mode must be one of auto, stateless, stateful, off (got {override!r})",
        )
    if value and value != "auto":
        return ClientMode(value)
    if any(message.get("role") in _STATEFUL_ROLES for message in messages):
        return ClientMode.STATEFUL
    return ClientMode.STATELESS


def validate_for_mode(messages: Sequence[Message], mode: ClientMode) -> None:
    """Stateless requests need at least one user message after the leading system block."""
    if mode is not ClientMode.STATELESS:
        return
    _, rest = split_leading_system(messages)
    if not any(message.get("role") == "user" for message in rest):
        raise GatewayError(
            400,
            "invalid_request",
            "stateless mode needs at least one user message",
            param="messages",
        )
