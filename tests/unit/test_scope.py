import re

import pytest

from context_gateway.core.scope import resolve_scope
from context_gateway.core.types import ClientMode, Scope
from context_gateway.errors import GatewayError


def resolve(**kwargs):
    kwargs.setdefault("headers", {})
    return resolve_scope(**kwargs)


# spec 1
def test_header_beats_path_beats_body_beats_default():
    path = {"app": "p-app", "user": "p-user", "session": "p-sess"}
    headers = {"X-Memory-App": "h-app", "X-Memory-User": "h-user", "X-Memory-Session": "h-sess"}

    assert resolve(headers=headers, path=path, body_user="b-user") == Scope(
        "h-app", "h-user", "h-sess"
    )
    assert resolve(path=path, body_user="b-user") == Scope("p-app", "p-user", "p-sess")
    assert resolve(body_user="b-user") == Scope("default", "b-user", "default")
    assert resolve() == Scope("default", "anonymous", "default")


def test_header_names_are_case_insensitive():
    scope = resolve(headers={"x-memory-user": "alice"})
    assert scope.user == "alice"


def test_default_app_comes_from_settings_value():
    assert resolve(default_app="my-app").app == "my-app"


# spec 2
def test_empty_header_is_treated_as_absent():
    path = {"app": "p-app", "user": "p-user", "session": "p-sess"}
    headers = {"X-Memory-App": "", "X-Memory-User": "  ", "X-Memory-Session": ""}
    assert resolve(headers=headers, path=path) == Scope("p-app", "p-user", "p-sess")


# spec 3
@pytest.mark.parametrize(
    ("header", "field"),
    [("X-Memory-App", "app"), ("X-Memory-User", "user"), ("X-Memory-Session", "session")],
)
@pytest.mark.parametrize("bad", ["has space", "semi;colon", "slash/inside", "x" * 129, "ünï"])
def test_invalid_values_raise_invalid_scope_naming_the_field(header, field, bad):
    with pytest.raises(GatewayError) as exc:
        resolve(headers={header: bad})
    assert exc.value.status_code == 400
    assert exc.value.code == "invalid_scope"
    assert field in exc.value.message
    assert exc.value.param == field


def test_invalid_body_user_is_rejected():
    with pytest.raises(GatewayError) as exc:
        resolve(body_user="bad user")
    assert exc.value.code == "invalid_scope"


def test_trailing_newline_is_rejected():
    with pytest.raises(GatewayError):
        resolve(headers={"X-Memory-User": "ok\nsneaky"})


def test_max_length_value_is_accepted():
    assert resolve(headers={"X-Memory-User": "x" * 128}).user == "x" * 128


# spec 4
def test_auto_session_stateful_is_stable_fingerprint():
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second question"},
    ]
    one = resolve(mode=ClientMode.STATEFUL, messages=messages)
    two = resolve(
        mode=ClientMode.STATEFUL, messages=[*messages, {"role": "assistant", "content": "x"}]
    )
    assert re.fullmatch(r"fp-[0-9a-f]{16}", one.session)
    assert one.session == two.session

    other = [{"role": "user", "content": "another chat"}, {"role": "assistant", "content": "a"}]
    assert resolve(mode=ClientMode.STATEFUL, messages=other).session != one.session


def test_auto_session_stateful_without_user_uses_first_non_system():
    messages = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "hello"}]
    scope = resolve(mode=ClientMode.STATEFUL, messages=messages)
    assert scope.session.startswith("fp-")


def test_auto_session_handles_list_content():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": "a"},
    ]
    first = resolve(mode=ClientMode.STATEFUL, messages=messages)
    second = resolve(mode=ClientMode.STATEFUL, messages=messages)
    assert first.session == second.session


def test_explicit_auto_value_in_path_is_resolved():
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    scope = resolve(path={"session": "auto"}, mode=ClientMode.STATEFUL, messages=messages)
    assert scope.session.startswith("fp-")


# spec 5
def test_auto_session_stateless_is_default():
    messages = [{"role": "user", "content": "q"}]
    assert resolve(mode=ClientMode.STATELESS, messages=messages).session == "default"


def test_explicit_session_is_not_replaced_in_stateful_mode():
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    scope = resolve(
        headers={"X-Memory-Session": "mine"}, mode=ClientMode.STATEFUL, messages=messages
    )
    assert scope.session == "mine"
