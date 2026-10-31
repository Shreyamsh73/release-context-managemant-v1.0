import pytest

from context_gateway.core.modes import (
    detect_mode,
    split_leading_system,
    validate_for_mode,
    validate_messages,
)
from context_gateway.core.types import ClientMode
from context_gateway.errors import GatewayError
from tests.helpers import assistant, system, user


# spec 6
def test_system_and_user_only_is_stateless():
    assert detect_mode([system("s"), user("q")]) is ClientMode.STATELESS


@pytest.mark.parametrize(
    "extra",
    [
        assistant("a"),
        {"role": "tool", "tool_call_id": "1", "content": "r"},
        {"role": "function", "name": "f", "content": "r"},
    ],
)
def test_assistant_tool_or_function_message_is_stateful(extra):
    assert detect_mode([user("q"), extra, user("q2")]) is ClientMode.STATEFUL


# spec 7
@pytest.mark.parametrize("value", ["stateless", "stateful", "off", "STATEFUL", " off "])
def test_override_wins_over_detection(value):
    messages = [user("q"), assistant("a"), user("q2")]
    assert detect_mode(messages, value) is ClientMode(value.strip().lower())


def test_auto_and_empty_override_detect_from_messages():
    assert detect_mode([user("q")], "auto") is ClientMode.STATELESS
    assert detect_mode([user("q")], "") is ClientMode.STATELESS
    assert detect_mode([user("q")], None) is ClientMode.STATELESS


def test_invalid_override_raises():
    with pytest.raises(GatewayError) as exc:
        detect_mode([user("q")], "sometimes")
    assert exc.value.status_code == 400
    assert exc.value.code == "invalid_memory_mode"
    assert exc.value.error_type == "invalid_request_error"


# spec 8
def test_split_leading_system_handles_developer_and_none():
    leading, rest = split_leading_system(
        [{"role": "developer", "content": "d"}, system("s"), user("q")]
    )
    assert [m["role"] for m in leading] == ["developer", "system"]
    assert [m["role"] for m in rest] == ["user"]

    leading, rest = split_leading_system([user("q")])
    assert leading == []
    assert rest == [user("q")]


def test_system_message_in_the_middle_stays_in_rest():
    leading, rest = split_leading_system([system("a"), user("q"), system("mid"), user("q2")])
    assert leading == [system("a")]
    assert [m["role"] for m in rest] == ["user", "system", "user"]


def test_split_leading_system_with_only_system_messages():
    leading, rest = split_leading_system([system("a"), system("b")])
    assert len(leading) == 2
    assert rest == []


def test_validate_messages():
    validate_messages([user("q")])
    for bad in (None, [], "text", [{"content": "no role"}], [{"role": 3}], ["str"]):
        with pytest.raises(GatewayError) as exc:
            validate_messages(bad)
        assert exc.value.status_code == 400
        assert exc.value.code == "invalid_request"


def test_stateless_needs_a_user_message():
    validate_for_mode([system("s"), user("q")], ClientMode.STATELESS)
    with pytest.raises(GatewayError) as exc:
        validate_for_mode([system("s")], ClientMode.STATELESS)
    assert exc.value.code == "invalid_request"
    # Forced stateless with an assistant-only tail is rejected too.
    with pytest.raises(GatewayError):
        validate_for_mode([assistant("a")], ClientMode.STATELESS)
    # Other modes do not need a user message.
    validate_for_mode([assistant("a")], ClientMode.STATEFUL)
    validate_for_mode([system("s")], ClientMode.OFF)
