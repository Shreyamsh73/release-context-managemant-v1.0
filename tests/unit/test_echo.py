from context_gateway.core.stream import StreamAccumulator
from context_gateway.core.summary import build_summary_prompt
from context_gateway.core.types import Turn
from context_gateway.llm.echo import EchoLLMClient


async def test_echo_reply_reports_history_and_summary_flag():
    llm = EchoLLMClient()
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    response = await llm.complete(model="m", messages=messages, params={})
    assert response["choices"][0]["message"]["content"] == (
        "echo: second | history_messages=2 | summary=no"
    )


async def test_echo_detects_summary_in_system_message_only():
    llm = EchoLLMClient()
    with_block = [
        {"role": "system", "content": "memory\n<summary>\nfacts\n</summary>"},
        {"role": "user", "content": "hi"},
    ]
    response = await llm.complete(model="m", messages=with_block, params={})
    assert response["choices"][0]["message"]["content"].endswith("history_messages=0 | summary=yes")

    in_user = [{"role": "user", "content": "<summary> not memory"}]
    response = await llm.complete(model="m", messages=in_user, params={})
    assert response["choices"][0]["message"]["content"].endswith("summary=no")


async def test_echo_response_has_openai_shape():
    llm = EchoLLMClient()
    response = await llm.complete(
        model="alias", messages=[{"role": "user", "content": "hi"}], params={}
    )
    assert response["object"] == "chat.completion"
    assert response["model"] == "alias"
    assert isinstance(response["id"], str)
    assert isinstance(response["created"], int)
    choice = response["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    usage = response["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_echo_stream_has_three_to_five_content_chunks_and_a_stop_chunk():
    llm = EchoLLMClient()
    messages = [{"role": "user", "content": "tell me something"}]
    expected = (await llm.complete(model="m", messages=messages, params={}))["choices"][0][
        "message"
    ]["content"]

    iterator = await llm.stream(model="m", messages=messages, params={})
    chunks = [chunk async for chunk in iterator]

    assert 4 <= len(chunks) <= 6
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    acc = StreamAccumulator()
    for chunk in chunks:
        acc.add(chunk)
    assert acc.completed
    assert acc.text == expected


async def test_echo_summary_request_gets_a_deterministic_summary():
    llm = EchoLLMClient()
    turns = [
        Turn(1, [{"role": "user", "content": "a"}], 1),
        Turn(2, [{"role": "user", "content": "b"}], 1),
    ]
    prompt = build_summary_prompt(None, turns)
    response = await llm.complete(model="m", messages=prompt, params={})
    assert response["choices"][0]["message"]["content"] == "- summary of 2 turns"

    prompt = build_summary_prompt("- summary of 2 turns", [turns[0]])
    response = await llm.complete(model="m", messages=prompt, params={})
    assert response["choices"][0]["message"]["content"] == "- summary of 3 turns"
