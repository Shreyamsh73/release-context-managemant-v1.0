import pytest
from pydantic import ValidationError

from context_gateway.config import Settings, normalize_database_url
from tests.helpers import make_settings


# spec 35
def test_aliases_parse_from_json_env(monkeypatch):
    monkeypatch.setenv("CTX_MODEL_ALIASES", '{"claude": "bedrock/converse/x"}')
    settings = Settings(_env_file=None)
    assert settings.model_aliases == {"claude": "bedrock/converse/x"}


def test_aliases_must_map_strings_to_strings():
    with pytest.raises(ValidationError):
        make_settings(model_aliases={"a": 1})
    with pytest.raises(ValidationError):
        make_settings(model_aliases=["a"])


def test_defaults_match_the_spec():
    settings = Settings(_env_file=None)
    assert settings.store == "postgres"
    assert settings.database_url == "postgresql://cg:cg@localhost:5432/cg"
    assert settings.redis_url == ""
    assert settings.job_backend == "inline"
    assert settings.llm_backend == "litellm"
    assert settings.token_counter == "litellm"
    assert settings.allow_unlisted_models is True
    assert settings.context_budget_tokens == 24000
    assert settings.summary_trigger_tokens == 8000
    assert settings.summary_keep_recent_tokens == 3000
    assert settings.summary_max_fold_tokens == 12000
    assert settings.summary_max_output_tokens == 800
    assert settings.window_max_turns == 200
    assert settings.cache_ttl_seconds == 600
    assert settings.api_keys == []
    assert settings.default_app == "default"
    assert settings.log_content is False
    assert (settings.host, settings.port) == ("0.0.0.0", 8080)


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_budget_tokens": 0},
        {"context_budget_tokens": -5},
        {"summary_trigger_tokens": 0},
        {"summary_keep_recent_tokens": 8000},
        {"summary_keep_recent_tokens": 9000},
        {"summary_keep_recent_tokens": -1},
        {"summary_trigger_tokens": 100, "summary_keep_recent_tokens": 100},
        {"summary_max_fold_tokens": 0},
        {"window_max_turns": 0},
    ],
)
def test_invalid_limits_fail_validation(overrides):
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_valid_limits_pass():
    settings = make_settings(summary_trigger_tokens=60, summary_keep_recent_tokens=20)
    assert settings.summary_trigger_tokens == 60


def test_comma_lists_are_split(monkeypatch):
    monkeypatch.setenv("CTX_API_KEYS", "k1, k2 ,,k3")
    monkeypatch.setenv("CTX_CORS_ORIGINS", "http://a.example, http://b.example")
    settings = Settings(_env_file=None)
    assert settings.api_keys == ["k1", "k2", "k3"]
    assert settings.cors_origins == ["http://a.example", "http://b.example"]


def test_empty_lists_from_empty_env(monkeypatch):
    monkeypatch.setenv("CTX_API_KEYS", "")
    assert Settings(_env_file=None).api_keys == []


@pytest.mark.parametrize(
    "raw",
    [
        "postgresql://u:p@h:5432/d",
        "postgres://u:p@h:5432/d",
        "postgresql+psycopg://u:p@h:5432/d",
    ],
)
def test_database_url_forms_are_normalized(raw):
    settings = make_settings(database_url=raw)
    assert settings.libpq_database_url == "postgresql://u:p@h:5432/d"
    assert settings.async_database_url == "postgresql+psycopg://u:p@h:5432/d"
    assert settings.sync_database_url == "postgresql+psycopg://u:p@h:5432/d"
    assert normalize_database_url(raw) == "postgresql://u:p@h:5432/d"


def test_invalid_store_value_fails():
    with pytest.raises(ValidationError):
        make_settings(store="sqlite")


def test_env_example_parses_and_documents_every_setting():
    from pathlib import Path

    example = Path(__file__).parents[2] / ".env.example"
    settings = Settings(_env_file=example)
    assert settings.model_aliases == {"claude": "bedrock/converse/us.anthropic.claude-sonnet-4-6"}
    assert settings.api_keys == []
    assert settings.cors_origins == []
    text = example.read_text()
    assert "replace" in text.lower() and "bedrock console" in text.lower()
    for name in Settings.model_fields:
        assert f"CTX_{name.upper()}=" in text, f"{name} is missing from .env.example"
