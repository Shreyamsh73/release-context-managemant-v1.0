# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependency layer: only changes when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Project layer.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH=/app/.venv/bin:$PATH \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('CTX_PORT', '8080') + '/healthz', timeout=3)"]
CMD ["context-gateway", "serve"]
