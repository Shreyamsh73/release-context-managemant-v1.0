"""Helpers for message content: flattening for counting and summaries, cleaning responses."""

from __future__ import annotations

import json
from typing import Any

_IMAGE_TYPES = frozenset({"image_url", "image", "input_image"})

# Keys of an assistant response message that are worth storing.
_ASSISTANT_KEYS = ("role", "content", "name", "tool_calls", "function_call", "refusal")


def flatten_content(content: Any) -> str:
    """Turn message content (string, list of parts, or None) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    pieces.append(str(part.get("text", "")))
                elif part_type in _IMAGE_TYPES:
                    pieces.append("[image]")
                else:
                    pieces.append(f"[{part_type}]")
            else:
                pieces.append(json.dumps(part, ensure_ascii=False, default=str))
        return "\n".join(pieces)
    return json.dumps(content, ensure_ascii=False, default=str)


def clean_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only the storable keys of a response message. Role is forced to assistant."""
    cleaned: dict[str, Any] = {"role": "assistant"}
    for key in _ASSISTANT_KEYS[1:]:
        value = message.get(key)
        if value is not None:
            cleaned[key] = value
        elif key == "content" and message.get("tool_calls"):
            cleaned[key] = None
    return cleaned


def is_empty_assistant_message(message: dict[str, Any]) -> bool:
    """True when a cleaned assistant message carries nothing worth storing."""
    return not (
        message.get("content")
        or message.get("tool_calls")
        or message.get("function_call")
        or message.get("refusal")
    )
