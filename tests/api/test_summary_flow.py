from tests.api.conftest import running_app
from tests.helpers import user

BASE = "/memory/v1/sessions/default/anonymous/default"


# spec 50
async def test_summary_end_to_end_with_inline_queue_and_tiny_thresholds():
    async with running_app(
        default_model="test/model", summary_trigger_tokens=40, summary_keep_recent_tokens=10
    ) as h:
        h.llm.summary_text = "- The user is called Shreyamsh and likes tea."

        triggered = None
        for i in range(1, 6):
            response = await h.chat(user(f"message number {i} with some padding text"))
            await h.queue.drain()
            info = (await h.client.get(BASE)).json()
            if info["summary"] and triggered is None:
                triggered = i
        assert triggered is not None, "a summary should have been written"

        info = (await h.client.get(BASE)).json()
        assert info["summary"] == "- The user is called Shreyamsh and likes tea."
        assert info["summarized_through_turn"] >= 1
        assert len(h.llm.summary_calls) >= 1
        prompt = h.llm.summary_calls[0].messages[1]["content"]
        assert "Existing summary:\n(none)" in prompt
        assert "### Turn 1" in prompt
        if len(h.llm.summary_calls) > 1:
            later = h.llm.summary_calls[1].messages[1]["content"]
            assert "Existing summary:\n- The user is called Shreyamsh" in later

        response = await h.chat(user("do you remember me?"))
        sent = h.llm.chat_calls[-1].messages
        memory = [m for m in sent if m["role"] == "system"]
        assert len(memory) == 1
        assert "<summary>" in memory[0]["content"]
        assert "likes tea" in memory[0]["content"]
        assert sent[0] == memory[0]
        assert response.headers["X-Memory-Summary"] == "used"
        # Summarized turns are no longer sent as raw history.
        contents = [m["content"] for m in sent if m["role"] == "user"]
        assert "message number 1 with some padding text" not in contents
        assert contents[-1] == "do you remember me?"


async def test_summary_never_runs_for_stateful_sessions():
    async with running_app(
        default_model="test/model", summary_trigger_tokens=20, summary_keep_recent_tokens=5
    ) as h:
        headers = {"X-Memory-Mode": "stateful"}
        history = []
        for i in range(1, 7):
            history.append(user(f"stateful message {i} with some padding text"))
            await h.chat(*history, headers=headers)
            history.append({"role": "assistant", "content": "fake reply"})
            await h.queue.drain()
        assert h.llm.summary_calls == []


async def test_below_trigger_nothing_is_summarized():
    async with running_app() as h:
        for i in range(3):
            await h.chat(user(f"short {i}"))
            await h.queue.drain()
        assert h.llm.summary_calls == []
        assert (await h.client.get(BASE)).json()["summary"] is None


async def test_manual_summarize_endpoint_runs_the_job():
    async with running_app(
        default_model="test/model", summary_trigger_tokens=40, summary_keep_recent_tokens=10
    ) as h:
        h.llm.queue_summary("manual summary")
        for i in range(1, 4):
            await h.chat(user(f"message number {i} with some padding text"))
        info_before = (await h.client.get(BASE)).json()
        await h.queue.drain()
        response = await h.client.post(BASE + "/summarize")
        assert response.status_code == 202
        await h.queue.drain()
        info = (await h.client.get(BASE)).json()
        assert info["summary"] is not None
        assert info_before["turn_count"] == info["turn_count"]


async def test_summary_failure_does_not_break_chat():
    async with running_app(
        default_model="test/model", summary_trigger_tokens=40, summary_keep_recent_tokens=10
    ) as h:
        h.llm.queue_summary("")  # empty output makes the job fail
        for i in range(1, 6):
            response = await h.chat(user(f"message number {i} with some padding text"))
            assert response.status_code == 200
            await h.queue.drain()
