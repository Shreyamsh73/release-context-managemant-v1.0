"""Job queue protocol."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class JobQueue(Protocol):
    """Runs summary jobs outside the request path."""

    async def enqueue_summarize(self, session_id: UUID) -> None:
        """Schedule a summary pass for the session."""
        ...

    async def start(self) -> None:
        """Open any connections the queue needs."""
        ...

    async def close(self) -> None:
        """Finish or cancel pending work and release resources."""
        ...
