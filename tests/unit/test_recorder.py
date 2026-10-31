from context_gateway.core.recorder import (
    extract_response_message,
    finalize_assistant_message,
    to_new_messages,
)
from context_gateway.core.tokens import HeuristicTokenCounter
from tests.helpers import user


def response(message):
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


def test_extract_response_message_cleans_the_first_choice():
    resp = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi", "audio": None}},
            {"index": 1, "message": {"role": "assistant", "content": "second"}},
        ]
    }
    assert extract_response_message(resp) == {"role": "assistant", "content": "hi"}


def test_extract_response_message_returns_none_without_choices_or_message():
    assert extract_response_message({}) is None
    assert extract_response_message({"choices": []}) is None
    assert extract_response_message({"choices": [{"index": 0}]}) is None
    assert extract_response_message({"choices": [{"message": "not a dict"}]}) is None


def test_extract_response_message_skips_empty_replies():
    assert extract_response_message(response({"role": "assistant", "content": ""})) is None
    assert extract_response_message(response({"role": "assistant", "content": None})) is None


def test_finalize_assistant_message():
    assert finalize_assistant_message(None) is None
    assert finalize_assistant_message({"role": "assistant"}) is None
    assert finalize_assistant_message({"role": "assistant", "content": "x"}) == {
        "role": "assistant",
        "content": "x",
    }


def test_to_new_messages_counts_tokens_and_keeps_order():
    counter = HeuristicTokenCounter()
    reply = {"role": "assistant", "content": "answer"}
    stored = to_new_messages([user("question")], reply, counter, "provider/model")
    assert [m.role for m in stored] == ["user", "assistant"]
    assert [m.message for m in stored] == [user("question"), reply]
    assert stored[0].token_count == counter.count([user("question")], None)
    assert all(m.model == "provider/model" for m in stored)


def test_to_new_messages_without_reply():
    stored = to_new_messages([user("q")], None, HeuristicTokenCounter(), None)
    assert len(stored) == 1
    assert stored[0].model is None
