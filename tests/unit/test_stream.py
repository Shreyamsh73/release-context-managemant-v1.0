from context_gateway.core.stream import StreamAccumulator, sse_done, sse_event


def chunk(delta=None, finish=None, index=0, **extra):
    return {
        "choices": [{"index": index, "delta": delta or {}, "finish_reason": finish}],
        **extra,
    }


# spec 23
def test_text_deltas_concatenate_and_finish_reason_completes():
    acc = StreamAccumulator()
    acc.add(chunk({"role": "assistant", "content": "Hel"}))
    assert acc.completed is False
    acc.add(chunk({"content": "lo "}))
    acc.add(chunk({"content": "world"}))
    acc.add(chunk({}, finish="stop"))
    assert acc.completed is True
    assert acc.finish_reason == "stop"
    assert acc.message() == {"role": "assistant", "content": "Hello world"}


def test_mark_done_completes_a_stream_without_finish_reason():
    acc = StreamAccumulator()
    acc.add(chunk({"content": "x"}))
    assert acc.completed is False
    acc.mark_done()
    assert acc.completed is True


def test_other_choice_indexes_and_empty_choices_are_ignored():
    acc = StreamAccumulator()
    acc.add(chunk({"content": "keep"}))
    acc.add(chunk({"content": "ignore"}, index=1))
    acc.add({"choices": []})
    acc.add({"id": "no choices at all"})
    assert acc.text == "keep"


def test_usage_chunk_is_stored():
    acc = StreamAccumulator()
    acc.add({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    assert acc.usage == {"prompt_tokens": 3, "completion_tokens": 4}


def test_empty_stream_gives_message_without_content():
    assert StreamAccumulator().message() == {"role": "assistant"}


# spec 24
def test_tool_call_deltas_merge_by_index_with_argument_concatenation():
    acc = StreamAccumulator()
    acc.add(
        chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            }
        )
    )
    acc.add(chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}))
    acc.add(chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "Paris"}'}}]}))
    acc.add(chunk({}, finish="tool_calls"))

    assert acc.message() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }


# spec 25
def test_two_parallel_tool_calls_interleaved_merge_correctly():
    acc = StreamAccumulator()
    acc.add(
        chunk(
            {
                "tool_calls": [
                    {"index": 1, "id": "b", "function": {"name": "second", "arguments": '{"y"'}},
                    {"index": 0, "id": "a", "function": {"name": "first", "arguments": '{"x"'}},
                ]
            }
        )
    )
    acc.add(chunk({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}))
    acc.add(chunk({"tool_calls": [{"index": 1, "function": {"arguments": ": 2}"}}]}))
    acc.add(chunk({}, finish="tool_calls"))

    calls = acc.message()["tool_calls"]
    assert [c["id"] for c in calls] == ["a", "b"]
    assert calls[0]["function"] == {"name": "first", "arguments": '{"x": 1}'}
    assert calls[1]["function"] == {"name": "second", "arguments": '{"y": 2}'}


def test_repeated_function_name_is_not_concatenated():
    acc = StreamAccumulator()
    acc.add(chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f"}}]}))
    acc.add(chunk({"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{}"}}]}))
    assert acc.message()["tool_calls"][0]["function"] == {"name": "f", "arguments": "{}"}


def test_tool_calls_without_index_use_their_position():
    acc = StreamAccumulator()
    acc.add(
        chunk(
            {
                "tool_calls": [
                    {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "g", "arguments": "{}"}},
                ]
            }
        )
    )
    assert [c["id"] for c in acc.message()["tool_calls"]] == ["a", "b"]


def test_text_and_tool_calls_together_keep_both():
    acc = StreamAccumulator()
    acc.add(chunk({"content": "calling"}))
    acc.add(chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f"}}]}))
    message = acc.message()
    assert message["content"] == "calling"
    assert len(message["tool_calls"]) == 1


def test_legacy_function_call_merges():
    acc = StreamAccumulator()
    acc.add(chunk({"function_call": {"name": "f", "arguments": '{"a"'}}))
    acc.add(chunk({"function_call": {"arguments": ": 1}"}}))
    message = acc.message()
    assert message["function_call"] == {"name": "f", "arguments": '{"a": 1}'}
    assert message["content"] is None


# spec 26
def test_sse_event_and_done_byte_format():
    assert sse_event({"a": 1, "b": "é"}) == 'data: {"a":1,"b":"é"}\n\n'.encode()
    assert sse_event("[DONE]") == b"data: [DONE]\n\n"
    assert sse_done() == b"data: [DONE]\n\n"
