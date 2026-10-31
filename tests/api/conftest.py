"""Fixtures for API tests: the real app with FakeLLM, in-memory store and inline jobs."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from context_gateway.app import create_app
from context_gateway.core.summary import Summarizer
from context_gateway.core.tokens import HeuristicTokenCounter
from context_gateway.jobs.inline import InlineJobQueue
from context_gateway.store.memory import InMemorySessionStore
from tests.fakes import FakeLLM
from tests.helpers import make_settings


@dataclass
class Harness:
    app: Any
    client: httpx.AsyncClient
    llm: FakeLLM
    store: Any
    queue: InlineJobQueue
    settings: Any

    async def chat(
        self,
        *messages: dict[str, Any],
        headers: dict[str, str] | None = None,
        path: str = "/v1/chat/completions",
        **body: Any,
    ) -> httpx.Response:
        payload = {"model": "m", "messages": list(messages), **body}
        return await self.client.post(path, json=payload, headers=headers)


@asynccontextmanager
async def running_app(
    *,
    llm: FakeLLM | None = None,
    store: Any = None,
    **settings_overrides: Any,
) -> AsyncIterator[Harness]:
    settings = make_settings(**settings_overrides)
    llm = llm or FakeLLM()
    store = store or InMemorySessionStore()
    counter = HeuristicTokenCounter()
    queue = InlineJobQueue(Summarizer(store, llm, counter, settings).run)
    app = create_app(settings, llm=llm, store=store, queue=queue, counter=counter)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(app, client, llm, store, queue, settings)


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    async with running_app() as h:
        yield h


def parse_sse(text: str) -> list[Any]:
    """Return the `data:` payloads of an SSE body. JSON where possible, raw string otherwise."""
    events: list[Any] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), block
        payload = block[len("data: ") :]
        try:
            events.append(json.loads(payload))
        except ValueError:
            events.append(payload)
    return events
