"""Command line entry point: `context-gateway serve | migrate | worker`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from context_gateway.config import Settings, get_settings

WORKER_QUEUES = ["summaries"]


def build_alembic_config(sync_url: str):
    """Build an Alembic config in code, pointing at the migrations shipped in the package."""
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    # ConfigParser treats "%" as interpolation, so escape it for percent-encoded passwords.
    config.set_main_option("sqlalchemy.url", sync_url.replace("%", "%%"))
    return config


def upgrade_database(sync_url: str, revision: str = "head") -> None:
    """Apply migrations. Running it again when up to date does nothing."""
    from alembic import command

    command.upgrade(build_alembic_config(sync_url), revision)


def _load_settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError as exc:
        print("Invalid configuration:", file=sys.stderr)
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"])
            message = str(error["msg"]).removeprefix("Value error, ")
            label = f"CTX_{where.upper()}: " if where else ""
            print(f"  - {label}{message}", file=sys.stderr)
        return None


def _serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "context_gateway.app:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
        access_log=False,  # the gateway writes its own access log line
    )
    return 0


def _migrate(settings: Settings) -> int:
    upgrade_database(settings.sync_database_url)
    print("database is up to date")
    return 0


async def _run_worker() -> None:
    from context_gateway.worker.app import app

    async with app.open_async():
        await app.run_worker_async(queues=WORKER_QUEUES)


def _worker(settings: Settings) -> int:
    from context_gateway.logging import configure_logging

    configure_logging(settings.log_level)
    asyncio.run(_run_worker())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, split out so tests can use it."""
    parser = argparse.ArgumentParser(prog="context-gateway", description="Context Gateway")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the HTTP gateway")
    serve.add_argument("--host", help="bind address (default: CTX_HOST)")
    serve.add_argument("--port", type=int, help="port (default: CTX_PORT)")
    serve.add_argument("--reload", action="store_true", help="reload on code changes")
    commands.add_parser("migrate", help="apply database migrations")
    commands.add_parser("worker", help="run the summary job worker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    args = build_parser().parse_args(argv)
    settings = _load_settings()
    if settings is None:
        return 2
    if args.command == "serve":
        return _serve(settings, args)
    if args.command == "migrate":
        return _migrate(settings)
    return _worker(settings)


if __name__ == "__main__":
    sys.exit(main())
