# Decisions and deviations

Each entry records a place where the code differs from `CLAUDE.md`, or where the spec was
ambiguous and the simplest option was chosen. One line of reason per entry.

1. `pyproject.toml` sets `[tool.ruff.format] exclude = ["*.md"]`. Reason: Ruff 0.17 formats
   Python code blocks inside Markdown and rewrote `CLAUDE.md`, and `ruff format --check .` in
   CI would fail on it.
2. Migration `0002` calls `exec_driver_sql(..., execution_options={"no_parameters": True})`.
   Reason: `exec_driver_sql` alone still fails because psycopg 3 parses the `%` in
   procrastinate's `RAISE '... %'` statements when an empty parameter tuple is passed.
3. `ContextBuilder.build` takes an extra keyword-only `session_key` and the class has a static
   `passthrough()` for off and degraded plans. Reason: the report needs the session key when
   no session row exists (preview of a missing session), and degraded is not a `ClientMode`.
4. Summaries need a model, so the end-to-end check and the no-Docker quickstart also set
   `CTX_DEFAULT_MODEL=echo`. Reason: spec 7.6 skips summaries when neither `CTX_SUMMARY_MODEL`
   nor `CTX_DEFAULT_MODEL` is set, and the spec 19.6 command line sets neither.
5. `GET /memory/v1/sessions/.../messages` returns `{"object": "list", "data": [...]}`.
   Reason: the spec gives no body shape; this matches `/v1/models`.
6. An assistant reply with no content, no tool calls, no function call and no refusal is not
   stored (the user messages of that turn still are). Reason: an empty assistant message makes
   later provider calls fail on some providers (Bedrock rejects empty content).
7. The response `model` field is rewritten to the client's model name only when the client sent
   one. Reason: with no model sent there is no client name to restore.
8. In `off` mode the scope is not resolved at all, so invalid scope headers are not rejected
   and no `X-Memory-App/User/Session` headers are sent. Reason: spec 8.4 says scope headers
   are omitted when not resolved, and off skips storage entirely.
9. Preview resolves scope as: headers, then the body `scope` object, then the body `user`
   field, then defaults, and the mode as: `X-Memory-Mode` header, then body `memory_mode`.
   Reason: the spec says "headers also work" without giving the order.
10. `PostgresSessionStore` takes the async SQLAlchemy URL (`postgresql+psycopg://`).
    Reason: keeps URL derivation in `Settings` only.
11. `CachedSessionStore` takes a keyword-only `cache_max_turns` (default 200) next to
    `ttl_seconds`. Reason: spec 7.8 caches only when `max_turns` equals the configured default,
    so the decorator has to know that number. Only that one key per session is written, so
    invalidation deletes exactly that key.
12. `ProcrastinateJobQueue` defers through `app.configure_task("summarize_session", ...)`
    instead of importing the task and calling `summarize_session.configure(...)`. Reason: same
    behavior, but it uses the app passed to the queue and avoids importing the worker task
    module from the queue module. The `PARTIAL` self-deferral inside the task uses the task
    object as written in the spec.
