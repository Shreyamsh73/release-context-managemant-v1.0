"""Scope resolution: which (app, user, session) a request belongs to."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from context_gateway.core.modes import SYSTEM_ROLES
from context_gateway.core.types import ClientMode, Message, Scope
from context_gateway.errors import GatewayError

AUTO = "auto"
DEFAULT_USER = "anonymous"
DEFAULT_SESSION_KEY = "default"

_ID_PATTERN = re.compile(r"[A-Za-z0-9._:@-]{1,128}")

HEADER_APP = "x-memory-app"
HEADER_USER = "x-memory-user"
HEADER_SESSION = "x-memory-session"


def validate_scope_value(field: str, value: str) -> str:
    """Return the value if it is a safe scope id, else raise `invalid_scope` naming the field."""
    if not _ID_PATTERN.fullmatch(value):
        shown = value if len(value) <= 40 else value[:40] + "..."
        raise GatewayError(
            400,
            "invalid_scope",
            f"Invalid {field} {shown!r}: use 1 to 128 characters from A-Z a-z 0-9 . _ : @ -",
            param=field,
        )
    return value


def canonical_json(value: Any) -> str:
    """Stable JSON used to fingerprint message content."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def auto_session_key(messages: Sequence[Message], mode: ClientMode) -> str:
    """Derive the session key for `auto` sessions.

    Stateful clients get a fingerprint of their first user message, so each new chat in an IDE
    gets its own session and the same chat always maps to the same one. Everyone else shares
    one `default` session per app and user.
    """
    if mode is not ClientMode.STATEFUL:
        return DEFAULT_SESSION_KEY
    first = next((m for m in messages if m.get("role") == "user"), None)
    if first is None:
        first = next((m for m in messages if m.get("role") not in SYSTEM_ROLES), None)
    content = first.get("content") if first else None
    digest = hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()
    return "fp-" + digest[:16]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            value = value.strip()
            return value or None
    return None


def _clean(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def resolve_scope(
    *,
    headers: Mapping[str, str],
    path: Mapping[str, Any] | None = None,
    body_user: str | None = None,
    default_app: str = "default",
    mode: ClientMode = ClientMode.STATELESS,
    messages: Sequence[Message] = (),
) -> Scope:
    """Resolve and validate the scope. Header beats path beats body beats default."""
    path = path or {}
    app = _header(headers, HEADER_APP) or _clean(path.get("app")) or default_app
    user = _header(headers, HEADER_USER) or _clean(path.get("user")) or _clean(body_user)
    session = _header(headers, HEADER_SESSION) or _clean(path.get("session")) or AUTO
    user = user or DEFAULT_USER

    validate_scope_value("app", app)
    validate_scope_value("user", user)
    validate_scope_value("session", session)

    if session == AUTO:
        session = auto_session_key(messages, mode)
    return Scope(app=app, user=user, session=session)
