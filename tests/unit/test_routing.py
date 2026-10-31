import pytest

from context_gateway.core.routing import list_model_ids, resolve_model, resolve_summary_model
from context_gateway.errors import GatewayError
from tests.helpers import make_settings

ALIASES = {"claude": "bedrock/converse/model-a", "fast": "bedrock/converse/model-b"}


# spec 33
def test_alias_maps_to_provider_string():
    settings = make_settings(model_aliases=ALIASES)
    assert resolve_model("claude", settings) == "bedrock/converse/model-a"


def test_default_resolves_default_model_raw():
    settings = make_settings(default_model="openai/gpt-4o-mini")
    assert resolve_model("default", settings) == "openai/gpt-4o-mini"
    assert resolve_model("", settings) == "openai/gpt-4o-mini"


def test_default_model_may_itself_be_an_alias():
    settings = make_settings(default_model="fast", model_aliases=ALIASES)
    assert resolve_model("default", settings) == "bedrock/converse/model-b"
    assert resolve_model("", settings) == "bedrock/converse/model-b"


def test_unlisted_model_passes_through_when_allowed():
    settings = make_settings(model_aliases=ALIASES, allow_unlisted_models=True)
    assert resolve_model("openai/gpt-4o", settings) == "openai/gpt-4o"


def test_unlisted_model_is_rejected_when_not_allowed():
    settings = make_settings(model_aliases=ALIASES, allow_unlisted_models=False)
    with pytest.raises(GatewayError) as exc:
        resolve_model("openai/gpt-4o", settings)
    assert exc.value.status_code == 400
    assert exc.value.code == "model_not_found"
    assert exc.value.error_type == "invalid_request_error"


def test_default_without_default_model_follows_unlisted_rule():
    assert resolve_model("default", make_settings()) == "default"
    with pytest.raises(GatewayError):
        resolve_model("default", make_settings(allow_unlisted_models=False))


def test_alias_beats_default_keyword():
    settings = make_settings(model_aliases={"default": "x/y"}, default_model="fast")
    assert resolve_model("default", settings) == "x/y"


def test_list_model_ids_includes_default_when_set():
    assert list_model_ids(make_settings(model_aliases=ALIASES)) == ["claude", "fast"]
    settings = make_settings(model_aliases=ALIASES, default_model="claude")
    assert list_model_ids(settings) == ["claude", "fast", "default"]


def test_summary_model_falls_back_to_default_and_resolves_aliases():
    assert resolve_summary_model(make_settings()) is None
    assert resolve_summary_model(make_settings(default_model="a/b")) == "a/b"
    settings = make_settings(model_aliases=ALIASES, default_model="claude", summary_model="fast")
    assert resolve_summary_model(settings) == "bedrock/converse/model-b"
    settings = make_settings(model_aliases=ALIASES, default_model="claude")
    assert resolve_summary_model(settings) == "bedrock/converse/model-a"
