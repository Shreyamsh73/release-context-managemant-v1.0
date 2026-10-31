"""Logging setup: one line format, `ts level logger msg key=value ...`."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

_HANDLER_MARK = "_context_gateway_handler"


def kv(**fields: Any) -> str:
    """Format fields as `key=value` pairs. Values with spaces or quotes are JSON-quoted."""
    parts = []
    for key, value in fields.items():
        if value is None:
            text = "-"
        elif isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        if not text or any(ch in text for ch in ' ="\n'):
            text = json.dumps(text, ensure_ascii=False)
        parts.append(f"{key}={text}")
    return " ".join(parts)


class _Formatter(logging.Formatter):
    converter = time.gmtime

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )


def configure_logging(level: str = "INFO") -> None:
    """Attach one stream handler to the `context_gateway` logger. Safe to call repeatedly."""
    root = logging.getLogger("context_gateway")
    root.setLevel(level.upper())
    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARK, False):
            handler.setLevel(level.upper())
            return
    handler = logging.StreamHandler()
    handler.setFormatter(_Formatter())
    handler.setLevel(level.upper())
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)
