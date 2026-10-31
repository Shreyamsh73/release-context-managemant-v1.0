import logging
from uuid import uuid4

import pytest

from context_gateway.core.summary import (
    MAX_MESSAGE_CHARS,
    SUMMARY_SYSTEM_PROMPT,
    Summarizer,
    SummaryError,
    SummaryOutcome,
    build_summary_prompt,
    render_transcript,
)
from context_gateway.core.tokens import HeuristicTokenCounter
from context_gateway.core.types import Scope, Turn
from context_gateway.store.base import NewMessage
from context_gateway.store.memory import InMemorySessionStore
from tests.fakes import FakeLLM
from tests.helpers import make_settings
from tests.store_contract import conversation

SCOPE = Scope("app", "alice", "default")


async def setup(turns: int, tokens: int = 10, **settings):
    store = InMemorySessionStore()
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(turns, tokens))
    llm = FakeLLM()
    defaults = {
        "default_model": "test/summary-model",
        "summary_trigger_tokens": 50,
        "summary_keep_recent_tokens": 45,
    }
    config = make_settings(**{**defaults, **settings})
    summarizer = Summarizer(store, llm, HeuristicTokenCounter(), config)
    return store, session, llm, summarizer


# spec 27
async def test_below_trigger_is_skipped_without_llm_call():
    store, session, llm, summarizer = await setup(turns=2, tokens=10)  # 40 tokens <= 50
    assert await summarizer.run(session.id) is SummaryOutcome.SKIPPED
    assert llm.calls == []
    assert (await store.get_session(session.id)).summary is None


async def test_exactly_at_trigger_is_skipped():
    _, session, llm, summarizer = await setup(turns=5, tokens=5)  # 50 tokens == 50
    assert await summarizer.run(session.id) is SummaryOutcome.SKIPPED
    assert llm.calls == []


async def test_missing_session_is_skipped():
    _, _, llm, summarizer = await setup(turns=1)
    assert await summarizer.run(uuid4()) is SummaryOutcome.SKIPPED
    assert llm.calls == []


# spec 28
async def test_above_trigger_folds_oldest_turns_and_keeps_newest():
    store, session, llm, summarizer = await setup(turns=5, tokens=10)  # 100 tokens
    await store.apply_summary(
        session.id, summary="OLD SUMMARY", through_turn=1, expected_through_turn=0
    )
    llm.queue_summary("NEW SUMMARY")

    outcome = await summarizer.run(session.id)

    assert outcome is SummaryOutcome.APPLIED
    refreshed = await store.get_session(session.id)
    assert refreshed.summary == "NEW SUMMARY"
    # Unsummarized turns 2..5 are 80 tokens. Keep budget 45 keeps turns 4 and 5.
    assert refreshed.summarized_through_turn == 3
    window = await store.load_window(session.id, max_turns=None)
    assert [t.index for t in window.turns] == [4, 5]

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.model == "test/summary-model"
    assert call.params == {"temperature": 0, "max_tokens": 800}
    assert call.messages[0] == {"role": "system", "content": SUMMARY_SYSTEM_PROMPT}
    prompt = call.messages[1]["content"]
    assert "Existing summary:\nOLD SUMMARY" in prompt
    assert "### Turn 2" in prompt
    assert "### Turn 3" in prompt
    assert "### Turn 4" not in prompt
    assert "USER: q2" in prompt
    assert "ASSISTANT: a3" in prompt


async def test_first_summary_says_none_for_existing():
    _, session, llm, summarizer = await setup(turns=5)
    await summarizer.run(session.id)
    assert "Existing summary:\n(none)" in llm.calls[0].messages[1]["content"]


async def test_summary_model_resolves_through_aliases():
    _, session, llm, summarizer = await setup(
        turns=5,
        default_model="",
        summary_model="fast",
        model_aliases={"fast": "bedrock/converse/small"},
    )
    await summarizer.run(session.id)
    assert llm.calls[0].model == "bedrock/converse/small"


async def test_no_summary_model_logs_warning_and_skips(caplog):
    _, session, llm, summarizer = await setup(turns=5, default_model="")
    with caplog.at_level(logging.WARNING):
        assert await summarizer.run(session.id) is SummaryOutcome.SKIPPED
    assert llm.calls == []
    assert "CTX_SUMMARY_MODEL" in caplog.text


# spec 29
async def test_always_keeps_at_least_one_turn_even_if_it_exceeds_keep_budget():
    store, session, _llm, summarizer = await setup(
        turns=3, tokens=50, summary_keep_recent_tokens=10
    )  # each turn is 100 tokens
    outcome = await summarizer.run(session.id)
    assert outcome is SummaryOutcome.APPLIED
    refreshed = await store.get_session(session.id)
    assert refreshed.summarized_through_turn == 2
    window = await store.load_window(session.id, max_turns=None)
    assert [t.index for t in window.turns] == [3]


async def test_single_oversized_turn_is_skipped_because_nothing_can_fold():
    _, session, llm, summarizer = await setup(turns=1, tokens=100)
    assert await summarizer.run(session.id) is SummaryOutcome.SKIPPED
    assert llm.calls == []


# spec 30
async def test_max_fold_tokens_cap_gives_partial_and_leaves_later_turns():
    store, session, _llm, summarizer = await setup(
        turns=6, tokens=10, summary_keep_recent_tokens=20, summary_max_fold_tokens=45
    )
    outcome = await summarizer.run(session.id)
    assert outcome is SummaryOutcome.PARTIAL
    refreshed = await store.get_session(session.id)
    assert refreshed.summarized_through_turn == 2  # turns 1 and 2 (40 tokens) fit in 45
    window = await store.load_window(session.id, max_turns=None)
    assert [t.index for t in window.turns] == [3, 4, 5, 6]


async def test_first_turn_is_folded_even_if_it_exceeds_the_fold_cap():
    store, session, _, summarizer = await setup(
        turns=4, tokens=10, summary_keep_recent_tokens=20, summary_max_fold_tokens=5
    )
    assert await summarizer.run(session.id) is SummaryOutcome.PARTIAL
    assert (await store.get_session(session.id)).summarized_through_turn == 1


async def test_runs_after_partial_continue_until_below_trigger():
    store, session, llm, summarizer = await setup(
        turns=6, tokens=10, summary_keep_recent_tokens=20, summary_max_fold_tokens=45
    )
    assert await summarizer.run(session.id) is SummaryOutcome.PARTIAL
    llm.queue_summary("S2")
    # Turns 3..6 are 80 tokens, still above the trigger of 50. Turns 3 and 4 fold next.
    assert await summarizer.run(session.id) is SummaryOutcome.PARTIAL
    refreshed = await store.get_session(session.id)
    assert refreshed.summary == "S2"
    assert refreshed.summarized_through_turn == 4
    # Turns 5 and 6 are 40 tokens, below the trigger, so the chain ends.
    assert await summarizer.run(session.id) is SummaryOutcome.SKIPPED


# spec 31
async def test_cas_conflict_gives_stale_without_error():
    store, session, _llm, summarizer = await setup(turns=5)

    class RacingLLM(FakeLLM):
        async def complete(self, **kwargs):
            response = await super().complete(**kwargs)
            # Another job wins the race while this one waits for the model.
            await store.apply_summary(
                session.id, summary="WINNER", through_turn=1, expected_through_turn=0
            )
            return response

    racing = RacingLLM()
    summarizer = Summarizer(
        store,
        racing,
        HeuristicTokenCounter(),
        make_settings(default_model="m", summary_trigger_tokens=50, summary_keep_recent_tokens=45),
    )
    assert await summarizer.run(session.id) is SummaryOutcome.STALE
    refreshed = await store.get_session(session.id)
    assert refreshed.summary == "WINNER"
    assert refreshed.summarized_through_turn == 1


# spec 32
@pytest.mark.parametrize("text", ["", "   \n"])
async def test_empty_llm_output_raises_summary_error(text):
    store, session, llm, summarizer = await setup(turns=5)
    llm.queue_summary(text)
    with pytest.raises(SummaryError):
        await summarizer.run(session.id)
    assert (await store.get_session(session.id)).summary is None


async def test_unexpected_response_shape_raises_summary_error():
    _store, session, llm, summarizer = await setup(turns=5)

    async def broken(**kwargs):
        return {"choices": []}

    llm.complete = broken
    with pytest.raises(SummaryError):
        await summarizer.run(session.id)


async def test_summary_text_is_stripped():
    store, session, llm, summarizer = await setup(turns=5)
    llm.queue_summary("  - fact one\n")
    await summarizer.run(session.id)
    assert (await store.get_session(session.id)).summary == "- fact one"


# transcript rendering
def test_transcript_renders_tool_calls_and_results():
    call = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}}
    turn = Turn(
        index=7,
        messages=[
            {"role": "user", "content": "find it"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "found"},
            {"role": "tool", "name": "named", "tool_call_id": "c2", "content": "other"},
            {"role": "assistant", "content": "here"},
        ],
        tokens=10,
    )
    assert render_transcript([turn]).splitlines() == [
        "### Turn 7",
        "USER: find it",
        'ASSISTANT called lookup({"q": 1})',
        "TOOL(c1): found",
        "TOOL(named): other",
        "ASSISTANT: here",
    ]


def test_transcript_truncates_long_messages():
    turn = Turn(
        index=1,
        messages=[{"role": "user", "content": "x" * (MAX_MESSAGE_CHARS + 50)}],
        tokens=1,
    )
    line = render_transcript([turn]).splitlines()[1]
    assert line == "USER: " + "x" * MAX_MESSAGE_CHARS + "... [truncated]"


def test_transcript_separates_turns_and_handles_list_content():
    turns = [
        Turn(1, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], 1),
        Turn(2, [{"role": "user", "content": "two"}], 1),
    ]
    assert render_transcript(turns) == "### Turn 1\nUSER: hi\n\n### Turn 2\nUSER: two"


def test_build_summary_prompt_shape():
    prompt = build_summary_prompt(None, [Turn(1, [{"role": "user", "content": "q"}], 1)])
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert prompt[1]["content"] == (
        "Existing summary:\n(none)\n\nNew turns to merge:\n### Turn 1\nUSER: q"
    )


def test_new_message_import_is_used():
    # Keeps the NewMessage import honest for helpers that build custom token counts.
    assert NewMessage({"role": "user", "content": "x"}, "user", 1, None).token_count == 1
