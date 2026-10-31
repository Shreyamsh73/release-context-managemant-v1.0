"""Gateway settings. All values come from `CTX_*` environment variables or a `.env` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def normalize_database_url(url: str) -> str:
    """Return the plain libpq form `postgresql://...` for any accepted input form."""
    for prefix in ("postgresql+psycopg://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


class Settings(BaseSettings):
    """Runtime configuration. See section 10 of the spec for the meaning of each field."""

    model_config = SettingsConfigDict(
        env_prefix="CTX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    store: Literal["postgres", "memory"] = "postgres"
    database_url: str = "postgresql://cg:cg@localhost:5432/cg"
    redis_url: str = ""
    job_backend: Literal["inline", "procrastinate"] = "inline"
    llm_backend: Literal["litellm", "echo"] = "litellm"
    token_counter: Literal["litellm", "heuristic"] = "litellm"

    default_model: str = ""
    model_aliases: dict[str, str] = Field(default_factory=dict)
    allow_unlisted_models: bool = True
    summary_model: str = ""

    context_budget_tokens: int = 24000
    summary_trigger_tokens: int = 8000
    summary_keep_recent_tokens: int = 3000
    summary_max_fold_tokens: int = 12000
    summary_max_output_tokens: int = 800
    window_max_turns: int = 200
    cache_ttl_seconds: int = 600

    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    default_app: str = "default"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    log_level: str = "INFO"
    log_content: bool = False
    host: str = "0.0.0.0"
    port: int = 8080

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("database_url")
    @classmethod
    def _normalize_url(cls, value: str) -> str:
        return normalize_database_url(value.strip())

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _check_limits(self) -> Settings:
        if self.context_budget_tokens <= 0:
            raise ValueError("CTX_CONTEXT_BUDGET_TOKENS must be greater than 0")
        if self.summary_trigger_tokens <= 0:
            raise ValueError("CTX_SUMMARY_TRIGGER_TOKENS must be greater than 0")
        if self.summary_keep_recent_tokens < 0:
            raise ValueError("CTX_SUMMARY_KEEP_RECENT_TOKENS must not be negative")
        if self.summary_keep_recent_tokens >= self.summary_trigger_tokens:
            raise ValueError(
                "CTX_SUMMARY_KEEP_RECENT_TOKENS must be less than CTX_SUMMARY_TRIGGER_TOKENS"
            )
        if self.summary_max_fold_tokens <= 0:
            raise ValueError("CTX_SUMMARY_MAX_FOLD_TOKENS must be greater than 0")
        if self.summary_max_output_tokens <= 0:
            raise ValueError("CTX_SUMMARY_MAX_OUTPUT_TOKENS must be greater than 0")
        if self.window_max_turns <= 0:
            raise ValueError("CTX_WINDOW_MAX_TURNS must be greater than 0")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("CTX_CACHE_TTL_SECONDS must be greater than 0")
        if not self.default_app.strip():
            raise ValueError("CTX_DEFAULT_APP must not be empty")
        return self

    @property
    def libpq_database_url(self) -> str:
        """`postgresql://...` form, used by procrastinate."""
        return normalize_database_url(self.database_url)

    @property
    def async_database_url(self) -> str:
        """`postgresql+psycopg://...` form for the async SQLAlchemy engine."""
        return "postgresql+psycopg://" + self.libpq_database_url.split("://", 1)[1]

    @property
    def sync_database_url(self) -> str:
        """`postgresql+psycopg://...` form for Alembic's sync engine."""
        return self.async_database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once."""
    return Settings()
