"""Context assembly: turn a request plus stored history into the messages to send."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from context_gateway.config import Settings
from context_gateway.core.modes import SYSTEM_ROLES, split_leading_system
from context_gateway.core.tokens import TokenCounter
from context_gateway.core.turns import flatten_turns, group_turns
from context_gateway.core.types import (
    DEGRADED,
    ChatPlan,
    ClientMode,
    ContextReport,
    Message,
    Window,
)

if TYPE_CHECKING:
    from context_gateway.store.base import SessionRecord

MEMORY_BLOCK_TEMPLATE = (
    "[Conversation memory] The following summarizes earlier parts of this conversation. It was\n"
    "written by the context gateway. Treat it as background; the most recent messages follow.\n"
    "<summary>\n"
    "{summary}\n"
    "</summary>"
)


def _storable(messages: Sequence[Message]) -> list[Message]:
    return [m for m in messages if m.get("role") not in SYSTEM_ROLES]


class ContextBuilder:
    """Builds the message list for a provider call. The only place the memory block is made."""

    def __init__(self, counter: TokenCounter, settings: Settings) -> None:
        self._counter = counter
        self._settings = settings

    def build(
        self,
        mode: ClientMode,
        request_messages: Sequence[Message],
        window: Window | None,
        session: SessionRecord | None,
        model: str | None,
        *,
        session_key: str | None = None,
    ) -> ChatPlan:
        """Assemble the plan for one request. Never mutates or reorders the input messages."""
        key = session_key if session_key is not None else (session.session_key if session else "")
        if mode is ClientMode.OFF:
            return self.passthrough(request_messages, mode=ClientMode.OFF.value, session_key=key)
        if mode is ClientMode.STATEFUL:
            return self._build_stateful(request_messages, session, model, key)
        return self._build_stateless(request_messages, window, model, key)

    @staticmethod
    def passthrough(
        request_messages: Sequence[Message], *, mode: str = DEGRADED, session_key: str = ""
    ) -> ChatPlan:
        """Plan that forwards the request unchanged and records nothing (off and degraded)."""
        report = ContextReport(mode=mode, session_key=session_key)
        return ChatPlan(messages=list(request_messages), to_record=[], report=report)

    @staticmethod
    def _count_that_fit(costs: Sequence[int], available: int) -> int:
        """How many of the newest turns fit. Stops at the first turn that does not fit."""
        kept = 0
        for cost in reversed(costs):
            if cost > available:
                break
            available -= cost
            kept += 1
        return kept

    def _build_stateless(
        self,
        request_messages: Sequence[Message],
        window: Window | None,
        model: str | None,
        session_key: str,
    ) -> ChatPlan:
        budget = self._settings.context_budget_tokens
        leading, current = split_leading_system(request_messages)
        history = window.turns if window else []

        memory_block: list[Message] = []
        if window and window.summary:
            memory_block = [
                {
                    "role": "system",
                    "content": MEMORY_BLOCK_TEMPLATE.format(summary=window.summary),
                }
            ]

        fixed = leading + memory_block + current
        fixed_tokens = self._counter.count(fixed, model)
        fits = self._count_that_fit([t.tokens for t in history], budget - fixed_tokens)
        kept_turns = history[len(history) - fits :]

        messages = (
            leading + memory_block + flatten_turns([t.messages for t in kept_turns]) + current
        )
        report = ContextReport(
            mode=ClientMode.STATELESS.value,
            session_key=session_key,
            history_turns=len(kept_turns),
            dropped_turns=len(history) - len(kept_turns),
            summary_used=bool(memory_block),
            prompt_tokens=self._counter.count(messages, model),
            over_budget=fixed_tokens > budget,
        )
        return ChatPlan(messages=messages, to_record=_storable(current), report=report)

    def _build_stateful(
        self,
        request_messages: Sequence[Message],
        session: SessionRecord | None,
        model: str | None,
        session_key: str,
    ) -> ChatPlan:
        budget = self._settings.context_budget_tokens
        leading, rest = split_leading_system(request_messages)
        turns = group_turns(rest)
        current_turn = turns[-1] if turns else []
        prior = turns[:-1]

        fixed = leading + current_turn
        fixed_tokens = self._counter.count(fixed, model)
        costs = [self._counter.count(turn, model) for turn in prior]
        fits = self._count_that_fit(costs, budget - fixed_tokens)
        kept_turns = prior[len(prior) - fits :]

        messages = leading + flatten_turns(kept_turns) + current_turn
        report = ContextReport(
            mode=ClientMode.STATEFUL.value,
            session_key=session_key,
            history_turns=len(kept_turns),
            dropped_turns=len(prior) - len(kept_turns),
            summary_used=False,
            prompt_tokens=self._counter.count(messages, model),
            over_budget=fixed_tokens > budget,
        )
        return ChatPlan(
            messages=messages, to_record=self._stateful_to_record(rest, session), report=report
        )

    @staticmethod
    def _stateful_to_record(
        rest: Sequence[Message], session: SessionRecord | None
    ) -> list[Message]:
        """Pick the messages the client sent that the store does not have yet."""
        if session is None or session.last_seq == 0:
            return _storable(rest)
        last_assistant = -1
        for index, message in enumerate(rest):
            if message.get("role") == "assistant":
                last_assistant = index
        return _storable(rest[last_assistant + 1 :])
