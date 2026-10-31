"""Session store protocol and its record types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from context_gateway.core.types import Scope, Window


class SessionNotFoundError(LookupError):
    """Raised when a write targets a session id that does not exist."""


@dataclass(frozen=True)
class SessionRecord:
    """One row of `cg_sessions`."""

    id: UUID
    app: str
    user: str
    session_key: str
    summary: str | None
    summarized_through_turn: int
    last_seq: int
    last_turn: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class NewMessage:
    """A message about to be stored. Turn and seq are assigned by the store."""

    message: dict[str, Any]
    role: str
    token_count: int
    model: str | None


@dataclass(frozen=True)
class StoredMessage:
    """One row of `cg_messages`."""

    seq: int
    turn_index: int
    role: str
    message: dict[str, Any]
    token_count: int
    model: str | None
    created_at: datetime


@dataclass(frozen=True)
class AppendResult:
    """State of the session after an append."""

    last_seq: int
    last_turn: int
    unsummarized_tokens: int


class SessionStore(Protocol):
    """Everything the gateway needs from session storage."""

    async def get_or_create_session(self, scope: Scope) -> SessionRecord:
        """Return the session for the scope, creating it when missing."""
        ...

    async def find_session(self, scope: Scope) -> SessionRecord | None:
        """Return the session for the scope, or None. Never creates."""
        ...

    async def get_session(self, session_id: UUID) -> SessionRecord | None:
        """Return the session with this id, or None."""
        ...

    async def load_window(self, session_id: UUID, *, max_turns: int | None) -> Window | None:
        """Return the summary and unsummarized turns, or None if the session is missing."""
        ...

    async def append_messages(
        self, session_id: UUID, messages: Sequence[NewMessage]
    ) -> AppendResult:
        """Append messages, assigning seq and turn index atomically."""
        ...

    async def apply_summary(
        self, session_id: UUID, *, summary: str, through_turn: int, expected_through_turn: int
    ) -> bool:
        """Compare-and-set the summary. Returns True when the row changed."""
        ...

    async def list_messages(self, session_id: UUID, *, limit: int) -> list[StoredMessage]:
        """Return the most recent stored messages in ascending order."""
        ...

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete the session and its messages. Returns True if it existed."""
        ...

    async def ping(self) -> None:
        """Raise if the store is not reachable."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
