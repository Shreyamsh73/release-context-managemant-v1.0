import sys

from context_gateway.core.tokens import HeuristicTokenCounter, make_token_counter


# spec 34
def test_heuristic_counter_is_deterministic():
    counter = HeuristicTokenCounter()
    messages = [{"role": "user", "content": "x" * 40}, {"role": "assistant", "content": "ok"}]
    first = counter.count(messages, "any")
    assert first == counter.count(messages, "other-model")
    # (4 + 40 / 4) + (4 + ceil(2 / 4))
    assert first == 14 + 5


def test_heuristic_counter_counts_tool_calls():
    counter = HeuristicTokenCounter()
    plain = {"role": "assistant", "content": None}
    with_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
        ],
    }
    assert counter.count([with_call], None) > counter.count([plain], None)
    assert counter.count([plain], None) == 4


def test_heuristic_counter_handles_list_content_and_empty_input():
    counter = HeuristicTokenCounter()
    assert counter.count([], None) == 0
    parts = {"role": "user", "content": [{"type": "text", "text": "abcd"}]}
    assert counter.count([parts], None) == 5


def test_heuristic_factory_does_not_import_litellm():
    assert isinstance(make_token_counter("heuristic"), HeuristicTokenCounter)
    assert "litellm" not in sys.modules
