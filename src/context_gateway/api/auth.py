"""Optional API key check for the gateway itself."""

from __future__ import annotations

import hmac

from fastapi import Depends, Request

from context_gateway.deps import Deps, get_deps
from context_gateway.errors import GatewayError


def _presented_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    api_key = request.headers.get("x-api-key", "").strip()
    return api_key or None


def key_matches(presented: str, configured: list[str]) -> bool:
    """Constant-time comparison against every configured key."""
    matched = False
    for key in configured:
        if hmac.compare_digest(presented.encode("utf-8"), key.encode("utf-8")):
            matched = True
    return matched


async def require_auth(request: Request, deps: Deps = Depends(get_deps)) -> None:
    """Reject the request unless auth is disabled or a configured key is presented."""
    keys = deps.settings.api_keys
    if not keys:
        return
    presented = _presented_key(request)
    if presented is None or not key_matches(presented, keys):
        raise GatewayError(401, "invalid_api_key", "Missing or invalid API key")
