"""Liveness and readiness probes. Never require auth."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from context_gateway.deps import Deps, get_deps

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """The process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(deps: Deps = Depends(get_deps)):
    """The store and, when configured, Redis answer."""
    checks: dict[str, str] = {}
    try:
        await deps.store.ping()
        checks["store"] = "ok"
    except Exception as exc:
        logger.warning("readiness: store check failed", exc_info=True)
        checks["store"] = f"error: {type(exc).__name__}"
    if deps.redis is None:
        checks["redis"] = "disabled"
    else:
        try:
            await deps.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            logger.warning("readiness: redis check failed", exc_info=True)
            checks["redis"] = f"error: {type(exc).__name__}"
    healthy = all(value in ("ok", "disabled") for value in checks.values())
    body = {"status": "ok" if healthy else "unavailable", "checks": checks}
    return JSONResponse(body, status_code=200 if healthy else 503)
