import copy
from datetime import UTC, datetime
from uuid import uuid4

from context_gateway.core.context import MEMORY_BLOCK_TEMPLATE, ContextBuilder
from context_gateway.core.tokens import HeuristicTokenCounter
from context_gateway.core.types import ClientMode, Turn, Window
from context_gateway.store.base import SessionRecord
from tests.helpers import assistant, make_settings, make_turn, system, user

COUNTER = HeuristicTokenCounter()


def builder(budget: int = 10_000) -> ContextBuilder:
    return ContextBuilder(COUNTER, make_settings(context_budget_tokens=budget))


def window(turns, summary=None) -> Window:
    return Window(
        session_id=uuid4(),
        summary=summary,
        summarized_through_turn=0,
        turns=list(turns),
        unsummarized_tokens=sum(t.tokens for t in turns),
        truncated=False,
    )


def session(last_seq: int = 0) -> SessionRecord:
    now = datetime.now(UTC)
    return SessionRecord(
        id=uuid4(),
        app="a",
        user="u",
        session_key="default",
        summary=None,
        summarized_through_turn=0,
        last_seq=last_seq,
        last_turn=0,
        created_at=now,
        updated_at=now,
    )


def stateless(request, win=None, budget=10_000):
    return builder(budget).build(ClientMode.STATELESS, request, win, session(), "m")


def test_template_matches_spec_text():
    assert MEMORY_BLOCK_TEMPLATE.startswith("[Conversation memory] The following summarizes")
    assert "<summary>\n{summary}\n</summary>" in MEMORY_BLOCK_TEMPLATE


# spec 10
def test_stateless_with_empty_window_sends_leading_system_and_current_only():
    request = [system("be brief"), user("hello")]
    plan = stateless(request, window([]))
    assert plan.messages == request
    assert plan.to_record == [user("hello")]
    assert plan.report.mode == "stateless"
    assert plan.report.history_turns == 0
    assert plan.report.dropped_turns == 0
    assert plan.report.summary_used is False
    assert plan.report.over_budget is False
    assert plan.report.prompt_tokens == COUNTER.count(request, "m")


def test_stateless_with_missing_window_behaves_like_empty():
    plan = stateless([user("hello")], None)
    assert plan.messages == [user("hello")]


# spec 11
def test_stateless_with_history_places_it_between_system_and_current():
    turns = [make_turn(1, 10), make_turn(2, 10)]
    request = [system("sys"), user("now")]
    plan = stateless(request, window(turns))
    assert [m["content"] for m in plan.messages] == ["sys", "q1", "a1", "q2", "a2", "now"]
    assert plan.report.history_turns == 2
    assert plan.to_record == [user("now")]


# spec 12
def test_summary_becomes_last_message_of_leading_system_block():
    turns = [make_turn(1, 10)]
    request = [system("one"), {"role": "developer", "content": "two"}, user("now")]
    plan = stateless(request, window(turns, summary="User likes tea."))

    roles = [m["role"] for m in plan.messages]
    assert roles == ["system", "developer", "system", "user", "assistant", "user"]
    block = plan.messages[2]
    assert "<summary>\nUser likes tea.\n</summary>" in block["content"]
    assert plan.report.summary_used is True
    # The memory block is never stored.
    assert plan.to_record == [user("now")]


def test_summary_without_leading_system_goes_first():
    plan = stateless([user("now")], window([], summary="S"))
    assert plan.messages[0]["role"] == "system"
    assert plan.messages[1] == user("now")


def test_summary_text_with_braces_is_safe():
    plan = stateless([user("now")], window([], summary="use {x} and {0}"))
    assert "use {x} and {0}" in plan.messages[0]["content"]


# spec 13
def test_budget_trimming_drops_oldest_turns_first():
    turns = [make_turn(i, 10) for i in range(1, 5)]
    request = [system("sys"), user("now")]
    fixed = COUNTER.count(request, "m")
    plan = stateless(request, window(turns), budget=fixed + 25)
    assert [m["content"] for m in plan.messages if m["role"] == "user"] == ["q3", "q4", "now"]
    assert plan.report.history_turns == 2
    assert plan.report.dropped_turns == 2
    assert plan.report.over_budget is False


def test_exact_fit_keeps_the_turn():
    request = [user("now")]
    fixed = COUNTER.count(request, "m")
    plan = stateless(request, window([make_turn(1, 10)]), budget=fixed + 10)
    assert plan.report.history_turns == 1


# spec 14
def test_a_turn_that_does_not_fit_stops_inclusion():
    turns = [make_turn(1, 5), make_turn(2, 5), make_turn(3, 100), make_turn(4, 5)]
    request = [user("now")]
    fixed = COUNTER.count(request, "m")
    plan = stateless(request, window(turns), budget=fixed + 50)
    contents = [m["content"] for m in plan.messages]
    assert contents == ["q4", "a4", "now"]
    assert plan.report.history_turns == 1
    assert plan.report.dropped_turns == 3


# spec 15
def test_tool_call_turn_is_kept_or_dropped_as_a_whole():
    tool_turn_messages = [
        user("use tool"),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "42"},
        assistant("it is 42"),
    ]
    tool_turn = Turn(index=1, messages=tool_turn_messages, tokens=30)
    request = [user("now")]
    fixed = COUNTER.count(request, "m")

    kept = stateless(request, window([tool_turn]), budget=fixed + 30)
    assert kept.messages == [*tool_turn_messages, user("now")]

    dropped = stateless(request, window([tool_turn]), budget=fixed + 29)
    assert dropped.messages == [user("now")]
    assert dropped.report.dropped_turns == 1


# spec 16
def test_over_budget_fixed_part_includes_no_history():
    request = [system("s" * 400), user("now")]
    plan = stateless(request, window([make_turn(1, 1)]), budget=10)
    assert plan.report.over_budget is True
    assert plan.report.history_turns == 0
    assert plan.report.dropped_turns == 1
    assert plan.messages == request


# spec 17
def test_stateful_does_not_inject_stored_history_and_trims_oldest_prior_turns():
    request = [
        system("sys"),
        user("q1"),
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
    ]
    stored = window([make_turn(1, 10), make_turn(2, 10)], summary="stored summary")
    big = ContextBuilder(COUNTER, make_settings(context_budget_tokens=10_000))
    plan = big.build(ClientMode.STATEFUL, request, stored, session(last_seq=4), "m")
    assert plan.messages == request
    assert plan.report.summary_used is False
    assert plan.report.mode == "stateful"

    fixed = COUNTER.count([system("sys"), user("q3")], "m")
    one_turn = COUNTER.count([user("q2"), assistant("a2")], "m")
    tight = ContextBuilder(COUNTER, make_settings(context_budget_tokens=fixed + one_turn))
    trimmed = tight.build(ClientMode.STATEFUL, request, None, session(last_seq=4), "m")
    assert trimmed.messages == [system("sys"), user("q2"), assistant("a2"), user("q3")]
    assert trimmed.report.history_turns == 1
    assert trimmed.report.dropped_turns == 1


# spec 18
def test_stateful_to_record_backfills_when_session_is_empty():
    request = [
        system("sys"),
        user("q1"),
        assistant("a1"),
        {"role": "system", "content": "mid"},
        user("q2"),
    ]
    plan = builder().build(ClientMode.STATEFUL, request, None, session(last_seq=0), "m")
    assert plan.to_record == [user("q1"), assistant("a1"), user("q2")]


def test_stateful_to_record_is_the_tail_after_last_assistant():
    request = [system("sys"), user("q1"), assistant("a1"), user("q2")]
    plan = builder().build(ClientMode.STATEFUL, request, None, session(last_seq=2), "m")
    assert plan.to_record == [user("q2")]


def test_stateful_to_record_includes_tool_results_after_last_assistant():
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": ""}}],
    }
    result = {"role": "tool", "tool_call_id": "c", "content": "r"}
    request = [user("q1"), tool_call, result]
    plan = builder().build(ClientMode.STATEFUL, request, None, session(last_seq=2), "m")
    assert plan.to_record == [result]


def test_stateful_to_record_without_assistant_is_everything_but_system():
    request = [system("sys"), user("q1")]
    plan = builder().build(ClientMode.STATEFUL, request, None, session(last_seq=5), "m")
    assert plan.to_record == [user("q1")]


def test_stateful_with_only_system_messages_does_not_crash():
    plan = builder().build(ClientMode.STATEFUL, [system("s")], None, session(), "m")
    assert plan.messages == [system("s")]
    assert plan.to_record == []


# spec 19
def test_off_mode_passes_messages_through_untouched():
    request = [system("s"), user("q"), assistant("a"), user("q2")]
    plan = builder(budget=1).build(ClientMode.OFF, request, None, None, "m")
    assert plan.messages == request
    assert plan.to_record == []
    assert plan.report.mode == "off"
    assert plan.report.history_turns == 0
    assert plan.report.dropped_turns == 0
    assert plan.report.prompt_tokens == 0
    assert plan.report.summary_used is False


def test_degraded_passthrough_reports_degraded_mode():
    request = [user("q")]
    plan = ContextBuilder.passthrough(request, session_key="default")
    assert plan.report.mode == "degraded"
    assert plan.report.session_key == "default"
    assert plan.messages == request
    assert plan.to_record == []


# spec 20
def test_input_messages_are_not_mutated():
    turns = [make_turn(1, 10), make_turn(2, 10)]
    stateless_request = [system("sys"), user("now")]
    stateful_request = [system("sys"), user("q1"), assistant("a1"), user("q2")]
    for mode, request, win in (
        (ClientMode.STATELESS, stateless_request, window(turns, summary="S")),
        (ClientMode.STATEFUL, stateful_request, None),
        (ClientMode.OFF, stateless_request, None),
    ):
        before = copy.deepcopy(request)
        win_before = copy.deepcopy(win)
        plan = builder(budget=40).build(mode, request, win, session(), "m")
        assert request == before
        assert win == win_before
        assert plan.messages is not request


def test_session_key_comes_from_session_or_override():
    sess = session()
    plan = builder().build(ClientMode.STATELESS, [user("q")], None, sess, "m")
    assert plan.report.session_key == "default"
    plan = builder().build(ClientMode.STATELESS, [user("q")], None, None, "m", session_key="x")
    assert plan.report.session_key == "x"
