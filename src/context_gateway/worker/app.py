"""The procrastinate app shared by the gateway (to defer jobs) and the worker (to run them)."""

from __future__ import annotations

from procrastinate import App, PsycopgConnector

from context_gateway.config import get_settings

app = App(
    connector=PsycopgConnector(conninfo=get_settings().libpq_database_url),
    import_paths=["context_gateway.worker.tasks"],
)
