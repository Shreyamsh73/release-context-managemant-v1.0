"""Shared fixtures. Integration fixtures for Postgres and Redis live in this file too."""

from __future__ import annotations

import os

import pytest

# Keep unit and API tests free of LiteLLM and of any developer machine config.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def _clean_ctx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove CTX_* variables so settings in tests only come from explicit arguments."""
    for key in list(os.environ):
        if key.startswith("CTX_"):
            monkeypatch.delenv(key, raising=False)
