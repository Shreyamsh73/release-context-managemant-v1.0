"""Decides which messages get stored after a successful provider call."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from context_gateway.core.content import clean_assistant_message, is_empty_assistant_message
from context_gateway.core.tokens import TokenCounter
from context_gateway.core.types import Message
from context_gateway.store.base import NewMessage


def extract_response_message(response: dict[str, Any]) -> Message | None:
    """Return the cleaned assistant message of `choices[0]`, or None if nothing is storable."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return None
    cleaned = clean_assistant_message(message)
    return None if is_empty_assistant_message(cleaned) else cleaned


def finalize_assistant_message(message: Message | None) -> Message | None:
    """Clean an assembled stream message and drop it when it is empty."""
    if message is None:
        return None
    cleaned = clean_assistant_message(message)
    return None if is_empty_assistant_message(cleaned) else cleaned


def to_new_messages(
    to_record: Sequence[Message],
    assistant_message: Message | None,
    counter: TokenCounter,
    model: str | None,
) -> list[NewMessage]:
    """Wrap client messages and the assistant reply for the store, with token counts."""
    messages = list(to_record)
    if assistant_message is not None:
        messages.append(assistant_message)
    return [
        NewMessage(
            message=message,
            role=str(message.get("role", "user")),
            token_count=counter.count([message], model),
            model=model,
        )
        for message in messages
    ]
