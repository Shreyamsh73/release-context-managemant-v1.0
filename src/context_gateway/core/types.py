"""Plain data types shared across the gateway. No I/O here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

Message = dict[str, Any]

# Reported in `X-Memory-Mode` when the memory layer failed and the request was forwarded as-is.
DEGRADED = "degraded"


class ClientMode(StrEnum):
    """Whether the gateway injects stored history into a request."""

    STATELESS = "stateless"
    STATEFUL = "stateful"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class Scope:
    """The (app, user, session) triple that identifies one conversation."""

    app: str
    user: str
    session: str


@dataclass(frozen=True, slots=True)
class Turn:
    """A user message plus the assistant and tool messages that follow it."""

    index: int
    messages: list[Message]
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict."""
        return {"index": self.index, "messages": self.messages, "tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        """Rebuild a turn from `to_dict` output."""
        return cls(index=data["index"], messages=data["messages"], tokens=data["tokens"])


@dataclass(frozen=True, slots=True)
class Window:
    """What the store returns for context building."""

    session_id: UUID
    summary: str | None
    summarized_through_turn: int
    turns: list[Turn]
    unsummarized_tokens: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (used by the Redis cache)."""
        return {
            "session_id": str(self.session_id),
            "summary": self.summary,
            "summarized_through_turn": self.summarized_through_turn,
            "turns": [turn.to_dict() for turn in self.turns],
            "unsummarized_tokens": self.unsummarized_tokens,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Window:
        """Rebuild a window from `to_dict` output."""
        return cls(
            session_id=UUID(data["session_id"]),
            summary=data["summary"],
            summarized_through_turn=data["summarized_through_turn"],
            turns=[Turn.from_dict(t) for t in data["turns"]],
            unsummarized_tokens=data["unsummarized_tokens"],
            truncated=data.get("truncated", False),
        )


@dataclass(frozen=True, slots=True)
class ContextReport:
    """What the context builder did. Reported to the client in response headers."""

    mode: str
    session_key: str = ""
    history_turns: int = 0
    dropped_turns: int = 0
    summary_used: bool = False
    prompt_tokens: int = 0
    over_budget: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChatPlan:
    """The result of context assembly for one request."""

    messages: list[Message]
    to_record: list[Message]
    report: ContextReport
    extras: dict[str, Any] = field(default_factory=dict)
