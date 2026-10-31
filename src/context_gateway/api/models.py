"""GET /v1/models, plain and scoped."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from context_gateway.api.auth import require_auth
from context_gateway.core.routing import list_model_ids
from context_gateway.deps import Deps, get_deps

router = APIRouter(dependencies=[Depends(require_auth)])


def _model_list(deps: Deps) -> dict[str, Any]:
    data = [
        {"id": model_id, "object": "model", "created": 0, "owned_by": "context-gateway"}
        for model_id in list_model_ids(deps.settings)
    ]
    return {"object": "list", "data": data}


@router.get("/v1/models")
async def list_models(deps: Deps = Depends(get_deps)):
    """List the configured model aliases."""
    return _model_list(deps)


@router.get("/scope/{app}/{user}/{session}/v1/models")
async def list_models_scoped(app: str, user: str, session: str, deps: Deps = Depends(get_deps)):
    """Same list. IDE plugins call `{base}/models` on the scoped base URL."""
    return _model_list(deps)
