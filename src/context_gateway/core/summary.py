"""Rolling summary of old turns."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any
from uuid import UUID

from context_gateway.config import Settings
from context_gateway.core.content import flatten_content
from context_gateway.core.routing import resolve_summary_model
from context_gateway.core.tokens import TokenCounter
from context_gateway.core.types import Message, Turn
from context_gateway.llm.base import LLMClient
from context_gateway.store.base import SessionStore

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "... [truncated]"
MAX_MESSAGE_CHARS = 4000

SUMMARY_SYSTEM_PROMPT = """\
You maintain a running summary of a conversation between a user and an AI assistant.
Merge the new turns into the existing summary.
Keep: facts about the user, stated preferences, decisions, names, identifiers, file paths,
code symbols, numbers, constraints, and open questions or pending tasks.
Drop: greetings, filler, and anything superseded by newer information (say what changed).
Write in third person ("The user ..."). Use short bullet points grouped by topic.
Never invent details. Stay under 300 words.
Output only the updated summary."""


class SummaryOutcome(StrEnum):
    """Result of one summary run."""

    SKIPPED = "skipped"
    APPLIED = "applied"
    PARTIAL = "partial"
    STALE = "stale"


class SummaryError(Exception):
    """The summary model returned nothing usable. The job should be retried."""


def _clip(text: str) -> str:
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[:MAX_MESSAGE_CHARS] + TRUNCATION_MARKER


def _render_message(message: Message) -> list[str]:
    role = str(message.get("role", ""))
    text = flatten_content(message.get("content"))
    if role == "user":
        lines = [f"USER: {_clip(text)}"]
    elif role == "assistant":
        lines = [f"ASSISTANT: {_clip(text)}"] if text else []
    elif role in ("tool", "function"):
        label = message.get("name") or message.get("tool_call_id") or ""
        lines = [f"TOOL({label}): {_clip(text)}"]
    else:
        lines = [f"{role.upper()}: {_clip(text)}"]
    if role == "assistant":
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            lines.append(f"ASSISTANT called {function.get('name', '')}({_clip(arguments)})")
        legacy = message.get("function_call")
        if legacy:
            lines.append(
                f"ASSISTANT called {legacy.get('name', '')}({_clip(legacy.get('arguments', ''))})"
            )
    return lines


def render_transcript(turns: list[Turn]) -> str:
    """Render turns as a plain-text transcript for the summary model."""
    blocks = []
    for turn in turns:
        lines = [f"### Turn {turn.index}"]
        for message in turn.messages:
            lines.extend(_render_message(message))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_summary_prompt(previous_summary: str | None, turns: list[Turn]) -> list[dict[str, Any]]:
    """Build the two-message prompt that merges new turns into the existing summary."""
    user_text = (
        f"Existing summary:\n{previous_summary or '(none)'}\n\n"
        f"New turns to merge:\n{render_transcript(turns)}"
    )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


class Summarizer:
    """Folds the oldest unsummarized turns into the session's rolling summary."""

    def __init__(
        self, store: SessionStore, llm: LLMClient, counter: TokenCounter, settings: Settings
    ) -> None:
        self._store = store
        self._llm = llm
        self._counter = counter
        self._settings = settings

    async def run(self, session_id: UUID) -> SummaryOutcome:
        """Run one summary pass for the session."""
        settings = self._settings
        window = await self._store.load_window(session_id, max_turns=None)
        if window is None or window.unsummarized_tokens <= settings.summary_trigger_tokens:
            return SummaryOutcome.SKIPPED

        model = resolve_summary_model(settings)
        if model is None:
            logger.warning(
                "summary skipped: set CTX_SUMMARY_MODEL or CTX_DEFAULT_MODEL session=%s", session_id
            )
            return SummaryOutcome.SKIPPED

        # Keep the newest turns within the keep-recent budget, and at least one turn.
        keep: list[Turn] = []
        keep_budget = settings.summary_keep_recent_tokens
        for turn in reversed(window.turns):
            if keep and turn.tokens > keep_budget:
                break
            keep.insert(0, turn)
            keep_budget -= turn.tokens
        candidates = window.turns[: len(window.turns) - len(keep)]

        # Cap the work done per job.
        fold: list[Turn] = []
        folded_tokens = 0
        for turn in candidates:
            if fold and folded_tokens + turn.tokens > settings.summary_max_fold_tokens:
                break
            fold.append(turn)
            folded_tokens += turn.tokens
        if not fold:
            return SummaryOutcome.SKIPPED

        response = await self._llm.complete(
            model=model,
            messages=build_summary_prompt(window.summary, fold),
            params={"temperature": 0, "max_tokens": settings.summary_max_output_tokens},
        )
        summary = self._extract_text(response)
        if not summary:
            raise SummaryError("summary model returned empty output")

        applied = await self._store.apply_summary(
            session_id,
            summary=summary,
            through_turn=fold[-1].index,
            expected_through_turn=window.summarized_through_turn,
        )
        if not applied:
            logger.info("summary stale, another job won session=%s", session_id)
            return SummaryOutcome.STALE
        logger.info(
            "summary applied session=%s folded_turns=%d through_turn=%d",
            session_id,
            len(fold),
            fold[-1].index,
        )
        if len(fold) < len(candidates):
            return SummaryOutcome.PARTIAL
        return SummaryOutcome.APPLIED

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SummaryError("summary model returned an unexpected response") from exc
        return flatten_content(content).strip()
