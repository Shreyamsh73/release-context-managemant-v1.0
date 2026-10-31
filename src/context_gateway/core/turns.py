"""Turn grouping. A turn is the unit of trimming and summarizing."""

from __future__ import annotations

from collections.abc import Sequence

from context_gateway.core.types import Message


def group_turns(messages: Sequence[Message]) -> list[list[Message]]:
    """Group non-system messages into turns. Each user message starts a new turn.

    A leading non-user message (for example an assistant greeting) forms its own turn.
    Tool calls stay with their tool results because only user messages start a turn.
    """
    turns: list[list[Message]] = []
    for message in messages:
        if message.get("role") == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def flatten_turns(turns: Sequence[Sequence[Message]]) -> list[Message]:
    """Concatenate turns back into a flat message list."""
    return [message for turn in turns for message in turn]
