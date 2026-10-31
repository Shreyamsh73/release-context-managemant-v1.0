from context_gateway.core.turns import flatten_turns, group_turns
from tests.helpers import assistant, user


# spec 9
def test_group_turns_keeps_tool_calls_with_their_results():
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }
    tool_result = {"role": "tool", "tool_call_id": "c1", "content": "42"}
    messages = [user("q1"), tool_call, tool_result, assistant("done"), user("q2"), assistant("a2")]

    turns = group_turns(messages)

    assert len(turns) == 2
    assert turns[0] == [user("q1"), tool_call, tool_result, assistant("done")]
    assert turns[1] == [user("q2"), assistant("a2")]


def test_leading_assistant_message_forms_its_own_turn():
    turns = group_turns([assistant("hello"), user("q"), assistant("a")])
    assert turns == [[assistant("hello")], [user("q"), assistant("a")]]


def test_consecutive_user_messages_are_separate_turns():
    assert len(group_turns([user("a"), user("b")])) == 2


def test_empty_input_gives_no_turns():
    assert group_turns([]) == []


def test_flatten_turns_round_trips():
    messages = [user("q1"), assistant("a1"), user("q2")]
    assert flatten_turns(group_turns(messages)) == messages
