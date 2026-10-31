from context_gateway.core.content import (
    clean_assistant_message,
    flatten_content,
    is_empty_assistant_message,
)


# spec 22
def test_flatten_content_str_and_none():
    assert flatten_content("hello") == "hello"
    assert flatten_content(None) == ""


def test_flatten_content_text_parts_join_with_newline():
    parts = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    assert flatten_content(parts) == "one\ntwo"


def test_flatten_content_image_and_unknown_parts():
    parts = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "input_audio", "input_audio": {}},
    ]
    assert flatten_content(parts) == "look\n[image]\n[input_audio]"


def test_flatten_content_anything_else_is_json():
    assert flatten_content({"a": 1}) == '{"a": 1}'
    assert flatten_content(5) == "5"


# spec 21
def test_clean_assistant_message_keeps_only_allowed_keys():
    raw = {
        "role": "model",
        "content": "hi",
        "name": "bot",
        "refusal": None,
        "reasoning_content": "thinking",
        "provider_specific_fields": {"x": 1},
        "audio": None,
        "function_call": None,
        "tool_calls": None,
    }
    assert clean_assistant_message(raw) == {"role": "assistant", "content": "hi", "name": "bot"}


def test_clean_assistant_message_keeps_none_content_with_tool_calls():
    calls = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    cleaned = clean_assistant_message({"role": "assistant", "content": None, "tool_calls": calls})
    assert cleaned == {"role": "assistant", "content": None, "tool_calls": calls}


def test_clean_assistant_message_adds_content_none_when_missing_with_tool_calls():
    calls = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    cleaned = clean_assistant_message({"tool_calls": calls})
    assert cleaned["content"] is None
    assert cleaned["role"] == "assistant"


def test_clean_assistant_message_drops_none_content_without_tool_calls():
    assert clean_assistant_message({"role": "assistant", "content": None}) == {"role": "assistant"}


def test_clean_assistant_message_keeps_legacy_function_call_and_refusal():
    raw = {"content": None, "function_call": {"name": "f", "arguments": "{}"}, "refusal": "no"}
    cleaned = clean_assistant_message(raw)
    assert cleaned["function_call"] == {"name": "f", "arguments": "{}"}
    assert cleaned["refusal"] == "no"


def test_empty_assistant_message_detection():
    assert is_empty_assistant_message({"role": "assistant"})
    assert is_empty_assistant_message({"role": "assistant", "content": ""})
    assert not is_empty_assistant_message({"role": "assistant", "content": "x"})
    assert not is_empty_assistant_message({"role": "assistant", "tool_calls": [{"id": "1"}]})
