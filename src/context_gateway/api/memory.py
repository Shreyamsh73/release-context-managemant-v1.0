"""Memory admin API under /memory/v1."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from context_gateway.api.auth import require_auth
from context_gateway.api.common import parse_chat_request, read_json_object
from context_gateway.core.context import ContextBuilder
from context_gateway.core.modes import detect_mode, validate_for_mode, validate_messages
from context_gateway.core.routing import resolve_model
from context_gateway.core.scope import resolve_scope, validate_scope_value
from context_gateway.core.types import ClientMode, Scope
from context_gateway.deps import Deps, get_deps
from context_gateway.errors import GatewayError
from context_gateway.store.base import SessionRecord

router = APIRouter(prefix="/memory/v1", dependencies=[Depends(require_auth)])


def _path_scope(app: str, user: str, session: str) -> Scope:
    """Admin paths carry literal scope values. `auto` is not resolved here."""
    return Scope(
        validate_scope_value("app", app),
        validate_scope_value("user", user),
        validate_scope_value("session", session),
    )


async def _require_session(deps: Deps, scope: Scope) -> SessionRecord:
    session = await deps.store.find_session(scope)
    if session is None:
        raise GatewayError(
            404,
            "session_not_found",
            f"No session for app={scope.app} user={scope.user} session={scope.session}",
        )
    return session


@router.get("/sessions/{app}/{user}/{session}")
async def get_session(app: str, user: str, session: str, deps: Deps = Depends(get_deps)):
    """Session info, including the rolling summary."""
    record = await _require_session(deps, _path_scope(app, user, session))
    window = await deps.store.load_window(record.id, max_turns=1)
    return {
        "app": record.app,
        "user": record.user,
        "session": record.session_key,
        "id": str(record.id),
        "summary": record.summary,
        "summarized_through_turn": record.summarized_through_turn,
        "turn_count": record.last_turn,
        "message_count": record.last_seq,
        "unsummarized_tokens": window.unsummarized_tokens if window else 0,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


@router.get("/sessions/{app}/{user}/{session}/messages")
async def list_messages(
    app: str,
    user: str,
    session: str,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    deps: Deps = Depends(get_deps),
):
    """The most recent stored messages, oldest first."""
    record = await _require_session(deps, _path_scope(app, user, session))
    stored = await deps.store.list_messages(record.id, limit=limit)
    data = [
        {
            "seq": m.seq,
            "turn_index": m.turn_index,
            "role": m.role,
            "message": m.message,
            "token_count": m.token_count,
            "model": m.model,
            "created_at": m.created_at.isoformat(),
        }
        for m in stored
    ]
    return {"object": "list", "data": data}


@router.delete("/sessions/{app}/{user}/{session}", status_code=204)
async def delete_session(app: str, user: str, session: str, deps: Deps = Depends(get_deps)):
    """Delete a session and all its messages."""
    record = await _require_session(deps, _path_scope(app, user, session))
    await deps.store.delete_session(record.id)
    return Response(status_code=204)


@router.post("/sessions/{app}/{user}/{session}/summarize", status_code=202)
async def summarize_session(app: str, user: str, session: str, deps: Deps = Depends(get_deps)):
    """Enqueue a summary pass now."""
    record = await _require_session(deps, _path_scope(app, user, session))
    await deps.queue.enqueue_summarize(record.id)
    return {"enqueued": True}


@router.post("/context/preview")
async def preview_context(request: Request, deps: Deps = Depends(get_deps)):
    """Dry run of context assembly. Makes no LLM call and writes nothing."""
    settings = deps.settings
    chat = parse_chat_request(await read_json_object(request))
    extras: dict[str, Any] = dict(chat.model_extra or {})
    scope_field = extras.get("scope")
    if scope_field is not None and not isinstance(scope_field, dict):
        raise GatewayError(400, "invalid_request", "scope must be an object", param="scope")
    memory_mode = extras.get("memory_mode")
    if memory_mode is not None and not isinstance(memory_mode, str):
        raise GatewayError(
            400, "invalid_request", "memory_mode must be a string", param="memory_mode"
        )

    validate_messages(chat.messages)
    mode = detect_mode(chat.messages, request.headers.get("x-memory-mode") or memory_mode)
    validate_for_mode(chat.messages, mode)
    provider_model = resolve_model(chat.model, settings)

    if mode is ClientMode.OFF:
        plan = ContextBuilder.passthrough(chat.messages, mode=ClientMode.OFF.value)
        return JSONResponse(
            {
                "mode": mode.value,
                "session": None,
                "model": provider_model,
                "messages": plan.messages,
                "report": plan.report.to_dict(),
            }
        )

    scope = resolve_scope(
        headers=request.headers,
        path=scope_field,
        body_user=chat.user,
        default_app=settings.default_app,
        mode=mode,
        messages=chat.messages,
    )
    session = await deps.store.find_session(scope)
    window = None
    if session is not None and mode is ClientMode.STATELESS:
        window = await deps.store.load_window(session.id, max_turns=settings.window_max_turns)
    plan = deps.builder.build(
        mode, chat.messages, window, session, provider_model, session_key=scope.session
    )
    return JSONResponse(
        jsonable_encoder(
            {
                "mode": mode.value,
                "session": scope.session,
                "model": provider_model,
                "messages": plan.messages,
                "report": plan.report.to_dict(),
            }
        )
    )
