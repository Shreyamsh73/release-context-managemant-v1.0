"""POST /v1/chat/completions, plain and scoped."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from context_gateway.api.auth import require_auth
from context_gateway.api.common import (
    ChatCompletionRequest,
    memory_headers,
    parse_chat_request,
    read_json_object,
)
from context_gateway.core.context import ContextBuilder
from context_gateway.core.modes import detect_mode, validate_for_mode, validate_messages
from context_gateway.core.recorder import (
    extract_response_message,
    finalize_assistant_message,
    to_new_messages,
)
from context_gateway.core.routing import resolve_model
from context_gateway.core.scope import resolve_scope
from context_gateway.core.stream import StreamAccumulator, sse_done, sse_event
from context_gateway.core.types import ChatPlan, ClientMode, Message, Scope
from context_gateway.deps import Deps, get_deps
from context_gateway.errors import error_body, truncate_message
from context_gateway.llm.base import LLMError
from context_gateway.logging import kv
from context_gateway.store.base import SessionRecord

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_auth)])


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, deps: Deps = Depends(get_deps)):
    """Chat with memory. Scope comes from headers and the body `user` field."""
    return await handle_chat(request, deps, path_scope=None)


@router.post("/scope/{app}/{user}/{session}/v1/chat/completions")
async def scoped_chat_completions(
    app: str, user: str, session: str, request: Request, deps: Deps = Depends(get_deps)
):
    """Chat with memory. Scope comes from the path; headers still override it."""
    scope = {"app": app, "user": user, "session": session}
    return await handle_chat(request, deps, path_scope=scope)


async def handle_chat(request: Request, deps: Deps, path_scope: Mapping[str, str] | None):
    """Run one chat request through the memory lifecycle."""
    settings = deps.settings
    chat = parse_chat_request(await read_json_object(request))
    validate_messages(chat.messages)
    mode = detect_mode(chat.messages, request.headers.get("x-memory-mode"))
    validate_for_mode(chat.messages, mode)
    provider_model = resolve_model(chat.model, settings)
    params = chat.passthrough_params(streaming=chat.stream)
    client_model = chat.model or None

    scope: Scope | None = None
    session: SessionRecord | None = None
    if mode is ClientMode.OFF:
        plan = ContextBuilder.passthrough(chat.messages, mode=ClientMode.OFF.value)
    else:
        scope = resolve_scope(
            headers=request.headers,
            path=path_scope,
            body_user=chat.user,
            default_app=settings.default_app,
            mode=mode,
            messages=chat.messages,
        )
        plan, session = await _memory_phase(deps, mode, scope, chat, provider_model)

    _note_access(request, scope, plan, provider_model)
    headers = memory_headers(plan.report, scope)

    if chat.stream:
        return await _stream_response(
            deps, session, mode, plan, scope, provider_model, client_model, params, headers
        )

    response = await deps.llm.complete(model=provider_model, messages=plan.messages, params=params)
    if client_model:
        response["model"] = client_model
    assistant = extract_response_message(response)
    await _record_turn(deps, session, mode, plan, assistant, provider_model)
    return JSONResponse(jsonable_encoder(response), headers=headers)


async def _memory_phase(
    deps: Deps,
    mode: ClientMode,
    scope: Scope,
    chat: ChatCompletionRequest,
    provider_model: str,
) -> tuple[ChatPlan, SessionRecord | None]:
    """Load the session and build the context. Any failure degrades to passthrough."""
    try:
        session = await deps.store.get_or_create_session(scope)
        window = None
        if mode is ClientMode.STATELESS:
            window = await deps.store.load_window(
                session.id, max_turns=deps.settings.window_max_turns
            )
        plan = deps.builder.build(mode, chat.messages, window, session, provider_model)
        return plan, session
    except Exception:
        logger.warning("memory phase failed, forwarding without memory", exc_info=True)
        return ContextBuilder.passthrough(chat.messages, session_key=scope.session), None


async def _record_turn(
    deps: Deps,
    session: SessionRecord | None,
    mode: ClientMode,
    plan: ChatPlan,
    assistant: Message | None,
    provider_model: str,
) -> None:
    """Store the turn and enqueue a summary if due. Failures are logged, never raised."""
    if session is None or plan.report.mode not in (ClientMode.STATELESS, ClientMode.STATEFUL):
        return
    try:
        new_messages = to_new_messages(plan.to_record, assistant, deps.counter, provider_model)
        if not new_messages:
            return
        result = await deps.store.append_messages(session.id, new_messages)
    except Exception:
        logger.warning("recording failed %s", kv(session=session.id), exc_info=True)
        return
    if mode is ClientMode.STATELESS and (
        result.unsummarized_tokens > deps.settings.summary_trigger_tokens
    ):
        try:
            await deps.queue.enqueue_summarize(session.id)
        except Exception:
            logger.warning("summary enqueue failed %s", kv(session=session.id), exc_info=True)


async def _stream_response(
    deps: Deps,
    session: SessionRecord | None,
    mode: ClientMode,
    plan: ChatPlan,
    scope: Scope | None,
    provider_model: str,
    client_model: str | None,
    params: dict[str, Any],
    headers: dict[str, str],
) -> StreamingResponse:
    iterator = await deps.llm.stream(model=provider_model, messages=plan.messages, params=params)
    # Pull the first chunk now so connection and auth errors still become JSON errors.
    try:
        first: dict[str, Any] | None = await iterator.__anext__()
    except StopAsyncIteration:
        first = None
    except BaseException:
        await _aclose(iterator)
        raise

    accumulator = StreamAccumulator()
    failed = False

    def encode(chunk: dict[str, Any]) -> bytes:
        accumulator.add(chunk)
        if client_model:
            chunk = {**chunk, "model": client_model}
        return sse_event(chunk)

    async def events() -> AsyncIterator[bytes]:
        nonlocal failed
        try:
            if first is not None:
                yield encode(first)
            async for chunk in iterator:
                yield encode(chunk)
            accumulator.mark_done()
        except LLMError as exc:
            failed = True
            yield _error_event(exc.message)
            return
        except Exception:
            failed = True
            logger.warning("stream failed", exc_info=True)
            yield _error_event("Internal error while streaming")
            return
        finally:
            await _aclose(iterator)
        yield sse_done()

    async def record() -> None:
        if failed or not accumulator.completed:
            return
        assistant = finalize_assistant_message(accumulator.message())
        await _record_turn(deps, session, mode, plan, assistant, provider_model)

    stream_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **headers}
    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers=stream_headers,
        background=BackgroundTask(record),
    )


def _error_event(message: str) -> bytes:
    text = truncate_message(message)
    return sse_event(error_body(text, "upstream_error", "upstream_error"))


async def _aclose(iterator: Any) -> None:
    """Close an async generator even when the surrounding task is being cancelled."""
    closer = getattr(iterator, "aclose", None)
    if closer is None:
        return
    try:
        with anyio.CancelScope(shield=True):
            await closer()
    except Exception:
        logger.debug("error closing upstream stream", exc_info=True)


def _note_access(request: Request, scope: Scope | None, plan: ChatPlan, model: str) -> None:
    """Hand the request details to the access log line written by the middleware."""
    access = getattr(request.state, "access", None)
    if access is None:
        return
    report = plan.report
    access.update(
        app=scope.app if scope else None,
        user=scope.user if scope else None,
        session=scope.session if scope else None,
        mode=report.mode,
        history_turns=report.history_turns,
        dropped_turns=report.dropped_turns,
        prompt_tokens=report.prompt_tokens,
        model=model,
    )
