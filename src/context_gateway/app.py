"""FastAPI app factory, lifespan, middleware and error handlers."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from context_gateway import __version__
from context_gateway.api import chat, health, memory, models
from context_gateway.api.common import MEMORY_HEADERS, format_validation_errors
from context_gateway.config import Settings, get_settings
from context_gateway.core.context import ContextBuilder
from context_gateway.core.summary import Summarizer
from context_gateway.core.tokens import TokenCounter
from context_gateway.deps import (
    Deps,
    build_llm,
    build_queue,
    build_redis,
    build_store,
    make_counter,
)
from context_gateway.errors import GatewayError, error_body, truncate_message
from context_gateway.jobs.base import JobQueue
from context_gateway.llm.base import LLMClient, LLMError
from context_gateway.logging import configure_logging, kv
from context_gateway.store.base import SessionStore

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("context_gateway.access")

_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_PHRASE_CODES = {
    400: ("invalid_request_error", "invalid_request"),
    401: ("authentication_error", "invalid_api_key"),
    404: ("not_found_error", "not_found"),
    405: ("invalid_request_error", "method_not_allowed"),
}


def _internal_error_body() -> dict[str, Any]:
    return error_body("Internal server error", "server_error", "internal_error")


class RequestContextMiddleware:
    """Assigns a request id, adds it to the response, and writes the access log line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = ""
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                incoming = value.decode("latin-1").strip()
                break
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        access: dict[str, Any] = {}
        state["access"] = access

        started = time.perf_counter()
        status = {"code": 500, "sent": False}

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                status["sent"] = True
                headers = list(message.get("headers", []))
                if not any(name.lower() == b"x-request-id" for name, _ in headers):
                    headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            logger.error("unhandled error %s", kv(request_id=request_id), exc_info=True)
            if status["sent"]:
                raise
            response = JSONResponse(_internal_error_body(), status_code=500)
            await response(scope, receive, send_with_id)
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            self._log(scope, request_id, status["code"], latency_ms, access)

    @staticmethod
    def _log(
        scope: Scope, request_id: str, status: int, latency_ms: float, access: dict[str, Any]
    ) -> None:
        path = scope.get("path", "")
        level = logging.DEBUG if path in ("/healthz", "/readyz") else logging.INFO
        if not access_logger.isEnabledFor(level):
            return
        fields = {"request_id": request_id, "method": scope.get("method"), "path": path}
        fields.update(access)
        fields.update(status=status, latency_ms=latency_ms)
        access_logger.log(level, "request %s", kv(**fields))


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def on_gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(exc.body(), status_code=exc.status_code, headers=headers)

    @app.exception_handler(LLMError)
    async def on_llm_error(request: Request, exc: LLMError) -> JSONResponse:
        status = exc.status_code
        mapped = status if status is not None and 400 <= status <= 499 else 502
        body = error_body(truncate_message(exc.message), "upstream_error", "upstream_error")
        return JSONResponse(body, status_code=mapped)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        message, param = format_validation_errors(list(exc.errors()))
        body = error_body(message, "invalid_request_error", "invalid_request", param)
        return JSONResponse(body, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_type, code = _PHRASE_CODES.get(exc.status_code, ("server_error", "http_error"))
        body = error_body(str(exc.detail), error_type, code)
        return JSONResponse(
            body, status_code=exc.status_code, headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse(_internal_error_body(), status_code=500)


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    store: SessionStore | None = None,
    queue: JobQueue | None = None,
    counter: TokenCounter | None = None,
) -> FastAPI:
    """Build the FastAPI app. Anything not injected is built from settings in the lifespan."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis = None
        if store is None:
            redis = build_redis(settings)
        llm_ = llm or build_llm(settings)
        counter_ = counter or make_counter(settings)
        store_ = store or build_store(settings, redis)
        summarizer = Summarizer(store_, llm_, counter_, settings)
        queue_ = queue or build_queue(settings, summarizer)
        deps = Deps(
            settings=settings,
            llm=llm_,
            store=store_,
            queue=queue_,
            counter=counter_,
            builder=ContextBuilder(counter_, settings),
            summarizer=summarizer,
            redis=redis,
        )
        await queue_.start()
        app.state.deps = deps
        logger.info(
            "started %s",
            kv(
                version=__version__,
                store=settings.store,
                jobs=settings.job_backend,
                llm=settings.llm_backend,
                redis=bool(redis),
                auth=bool(settings.api_keys),
            ),
        )
        try:
            yield
        finally:
            await deps.close()

    app = FastAPI(
        title="Context Gateway",
        version=__version__,
        description="Memory for stateless LLM APIs.",
        lifespan=lifespan,
    )
    _install_error_handlers(app)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[*MEMORY_HEADERS, "X-Request-Id"],
        )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(memory.router)
    return app
