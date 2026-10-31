"""Model alias resolution."""

from __future__ import annotations

from context_gateway.config import Settings
from context_gateway.errors import GatewayError

DEFAULT_MODEL_ID = "default"


def resolve_model(requested: str, settings: Settings) -> str:
    """Map the model name a client sent to the provider-facing LiteLLM model string."""
    aliases = settings.model_aliases
    if requested in aliases:
        return aliases[requested]
    if requested in ("", DEFAULT_MODEL_ID) and settings.default_model:
        return aliases.get(settings.default_model, settings.default_model)
    if settings.allow_unlisted_models:
        return requested
    raise GatewayError(
        400,
        "model_not_found",
        f"The model {requested!r} is not available. Use GET /v1/models to list models.",
        param="model",
    )


def resolve_summary_model(settings: Settings) -> str | None:
    """The model used for summaries, or None when neither it nor a default model is set."""
    name = settings.summary_model or settings.default_model
    if not name:
        return None
    return settings.model_aliases.get(name, name)


def list_model_ids(settings: Settings) -> list[str]:
    """Names shown by `/v1/models`: the aliases, plus `default` when a default model is set."""
    ids = list(settings.model_aliases)
    if settings.default_model and DEFAULT_MODEL_ID not in ids:
        ids.append(DEFAULT_MODEL_ID)
    return ids
