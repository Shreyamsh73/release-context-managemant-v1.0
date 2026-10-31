# CLAUDE.md: Context Gateway technical specification

This file is the source of truth for building this repository. It was written by the
architect (Claude Opus 5.5) for the coding agent (Claude Sonnet 5.5) and for any future
contributor.

**Read this whole file before writing any code.** Every section matters. When the code and
this file disagree, this file wins unless a deviation is recorded in `docs/DECISIONS.md`
(see section 19).

---

## 0. TL;DR

Build **Context Gateway**: a self-hosted HTTP service that sits between any app and any LLM
provider and gives stateless LLM APIs a memory.

- Apps point their OpenAI-compatible client at the gateway instead of the provider. That is
  the only integration step ("change your base URL").
- The gateway stores each conversation, rebuilds the right context for every call, keeps it
  inside a token budget, and folds old turns into a rolling summary.
- Provider calls go through the LiteLLM Python SDK. AWS Bedrock (Claude) is the default
  provider, but any LiteLLM provider works.
- Stack: Python 3.12, FastAPI, LiteLLM, SQLAlchemy 2 (async, Core), psycopg 3, Postgres,
  Redis (optional cache), procrastinate (Postgres job queue), Alembic.

**This task is Phase 1 only: short-term (session) memory.** Long-term fact memory, vectors,
UI and MCP are later phases (section 20). Do not build them now.

---

## 1. Problem and positioning

LLM APIs (Bedrock Converse, OpenAI Chat Completions, Anthropic Messages) are stateless.
Every call must carry the full conversation. Developers who call these APIs directly end up
rebuilding the same plumbing in every project: store history, trim it to fit the context
window, summarize old turns, separate users and sessions.

Context Gateway makes that plumbing a reusable, self-hosted service:

| Property | What it means |
|---|---|
| Plug and play | Any OpenAI-compatible client works by changing the base URL. No SDK required. |
| Self-hosted | Data and provider credentials stay on the operator's infrastructure. |
| Provider-agnostic | LiteLLM translates OpenAI format to Bedrock, Anthropic, OpenAI, Gemini and more. |
| Transparent | Every response carries headers describing what memory was injected. |
| Safe by default | If the memory layer fails, requests still go through (degraded passthrough). |

Two kinds of clients must both work correctly:

1. **Stateless clients** send only the newest user message (for example a script calling
   the API once per question). The gateway must inject the stored history.
2. **Stateful clients** already resend the full history every call (for example IDE chat
   plugins such as Continue). The gateway must not inject history again, or it would be
   duplicated. It still records the conversation and enforces the token budget.

---

## 2. Phase 1 scope

### In scope

1. OpenAI-compatible `POST /v1/chat/completions`, streaming and non-streaming.
2. Scoped route variant `POST /scope/{app}/{user}/{session}/v1/chat/completions` for clients
   that cannot set custom headers.
3. `GET /v1/models` (and scoped variant) listing configured model aliases.
4. Scope resolution (app, user, session) from headers, path, body.
5. Client mode detection (stateless, stateful, off) with an override header.
6. Session storage in Postgres, with an in-memory store for tests and quick local runs.
7. Context assembly under a token budget, never splitting a tool-call group.
8. Rolling summary of old turns, computed asynchronously by a job queue
   (inline asyncio backend and procrastinate backend).
9. Optional Redis cache for session windows.
10. Memory admin endpoints under `/memory/v1` (inspect, list messages, delete, preview,
    trigger summary).
11. Optional API-key auth for the gateway itself.
12. Degraded passthrough when storage fails.
13. `echo` LLM backend for local demos and end-to-end tests without provider credentials.
14. CLI: `context-gateway serve | migrate | worker`.
15. Alembic migrations (our tables plus the procrastinate schema).
16. Tests: unit, API (with a fake LLM), integration (real Postgres and Redis).
17. Dockerfile, docker-compose, GitHub Actions CI, README.

### Out of scope for Phase 1 (do not build)

- Long-term fact extraction, reconcile logic, embeddings, pgvector queries.
- Context inspector UI, dashboards.
- MCP server.
- Per-tenant API key management, rate limiting, billing.
- Native (non-OpenAI) inbound APIs such as Bedrock Converse or Anthropic Messages format.
- Summaries for stateful sessions (stateful clients own their history in Phase 1).

---

## 3. Tech stack and dependency rules

Versions are locked in `uv.lock` (already created). Current locked versions:

| Package | Locked | Purpose |
|---|---|---|
| Python | 3.12 (`.python-version`) | Runtime. Code must also run on 3.13. |
| fastapi | 0.162.0 | HTTP API |
| uvicorn[standard] | 0.54.0 | ASGI server |
| pydantic / pydantic-settings | 2.14.2 / latest | Models and config |
| sqlalchemy[asyncio] | 2.1.6 | DB access. Use **Core** (Table + select/insert), not the ORM. |
| psycopg[binary] | 3.3.6 | Postgres driver for SQLAlchemy (`postgresql+psycopg://`) and procrastinate |
| alembic | 1.20.1 | Migrations |
| redis | 8.1.0 | `redis.asyncio` client for the cache |
| procrastinate | 3.10.0 | Postgres-backed job queue for summaries |
| litellm | 1.106.6 | Provider translation |
| boto3 | 1.43.x | AWS credentials and signing for Bedrock (used by LiteLLM) |
| dev: pytest, pytest-asyncio, httpx, ruff, fakeredis, testcontainers | see lock | Tests and lint |

Rules:

1. **LiteLLM is security-sensitive.** Versions 1.82.7 and 1.82.8 on PyPI were malicious
   (March 2026 supply-chain incident). Never loosen the lock casually. Never upgrade LiteLLM
   in the same commit as feature work.
2. **Set `LITELLM_LOCAL_MODEL_COST_MAP=True`** before LiteLLM is imported (in
   `context_gateway/__init__.py` via `os.environ.setdefault`, and in the Dockerfile).
   Otherwise LiteLLM fetches a cost map from the internet at import time.
3. **Import LiteLLM lazily.** `import litellm` takes about 8 seconds. Only
   `llm/litellm_client.py` and `core/tokens.py` (`LiteLLMTokenCounter`) may import it, and
   only inside functions or `__init__`, never at module top level. Unit and API tests must
   never import LiteLLM.
4. Call LiteLLM with `drop_params=True` so OpenAI parameters a provider does not support
   (for example `user`, `logit_bias` on Bedrock) are dropped instead of raising.
5. Add a dependency only when it removes real code. Use `uv add` (or `uv add --dev`) so the
   lock stays consistent. Commit `uv.lock`.

---

## 4. Repository layout

Create exactly this structure (small helper modules may be added if needed):

```
.
├── CLAUDE.md                       # this spec
├── README.md                       # user-facing docs (section 18)
├── LICENSE                         # Apache-2.0 full text
├── pyproject.toml                  # already drafted, adjust if needed
├── uv.lock
├── .python-version                 # 3.12
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── docs/
│   └── DECISIONS.md                # deviations from this spec, one entry each
├── scripts/
│   └── dev-services.sh             # start/stop local Postgres + Redis without Docker
├── src/context_gateway/
│   ├── __init__.py                 # __version__, LITELLM_LOCAL_MODEL_COST_MAP default
│   ├── cli.py                      # serve | migrate | worker
│   ├── config.py                   # Settings (pydantic-settings, prefix CTX_)
│   ├── app.py                      # create_app(), lifespan, middleware, error handlers
│   ├── deps.py                     # Deps container + FastAPI dependency getters
│   ├── errors.py                   # GatewayError and OpenAI-style error rendering
│   ├── logging.py                  # logging setup
│   ├── api/
│   │   ├── chat.py                 # chat completions (plain + scoped routes)
│   │   ├── models.py               # /v1/models (plain + scoped)
│   │   ├── memory.py               # /memory/v1/*
│   │   └── health.py               # /healthz, /readyz
│   ├── core/
│   │   ├── types.py                # Scope, Turn, Window, ClientMode, ContextReport, ChatPlan
│   │   ├── scope.py                # resolve_scope, validate ids, auto session key
│   │   ├── modes.py                # detect_mode, split_leading_system
│   │   ├── turns.py                # group_turns, flatten_turns
│   │   ├── content.py              # content flattening, assistant message cleaning
│   │   ├── tokens.py               # TokenCounter protocol + implementations
│   │   ├── context.py              # ContextBuilder (assembly + budget)
│   │   ├── recorder.py             # which messages get stored
│   │   ├── routing.py              # model alias resolution
│   │   ├── stream.py               # StreamAccumulator, SSE encoding
│   │   └── summary.py              # Summarizer + prompt
│   ├── llm/
│   │   ├── base.py                 # LLMClient protocol, LLMError
│   │   ├── litellm_client.py       # LiteLLM implementation
│   │   └── echo.py                 # EchoLLMClient (dev/demo/e2e)
│   ├── store/
│   │   ├── base.py                 # SessionStore protocol, records
│   │   ├── tables.py               # SQLAlchemy Core tables
│   │   ├── memory.py               # InMemorySessionStore
│   │   ├── postgres.py             # PostgresSessionStore
│   │   └── cache.py                # CachedSessionStore (Redis decorator)
│   ├── jobs/
│   │   ├── base.py                 # JobQueue protocol
│   │   ├── inline.py               # InlineJobQueue (asyncio tasks)
│   │   └── procrastinate_queue.py  # ProcrastinateJobQueue
│   ├── worker/
│   │   ├── app.py                  # procrastinate App
│   │   └── tasks.py                # summarize_session task
│   └── migrations/                 # Alembic, shipped inside the package
│       ├── env.py
│       ├── script.py.mako
│       └── versions/
│           ├── 0001_initial.py
│           └── 0002_procrastinate.py
└── tests/
    ├── conftest.py
    ├── fakes.py                    # FakeLLM and helpers
    ├── unit/
    ├── api/
    └── integration/
```

Already present when you start: `pyproject.toml`, `uv.lock`, `.python-version`,
`.gitignore`, this file, and a synced `.venv`. `src/context_gateway/__init__.py` and
`README.md` exist but are empty.

---

## 5. Core concepts

| Term | Definition |
|---|---|
| **Scope** | `(app, user, session)` triple identifying one conversation. Strings, validated (section 7.1). |
| **Session** | One stored conversation for a scope. Row in `cg_sessions`. Has a UUID `id`. |
| **Message** | One OpenAI chat message dict (`role`, `content`, optional `name`, `tool_calls`, `tool_call_id`). Stored verbatim as JSONB. |
| **Turn** | A group of consecutive messages starting at a `user` message and including the following assistant/tool messages up to the next `user` message. Turns are the unit of trimming and summarizing, so tool-call groups are never split. Turn indexes start at 1. |
| **Window** | What the store returns for context building: the summary, `summarized_through_turn`, and the unsummarized turns in ascending order. |
| **Summary** | Rolling text summary of all turns with `turn_index <= summarized_through_turn`. |
| **Client mode** | `stateless`, `stateful`, or `off`. Decides whether the gateway injects history. Plus `degraded` as a reported status when memory fails. |
| **Budget** | Max estimated prompt tokens the gateway will send (`CTX_CONTEXT_BUDGET_TOKENS`). |
| **Leading system block** | The contiguous run of `system` or `developer` messages at the start of the request. Only these are treated as system prompt. LiteLLM's Bedrock adapter only hoists leading system messages, so the memory block must be placed inside this run (at its end). |

---

## 6. Architecture and request lifecycle

```mermaid
flowchart LR
    C[Client app<br/>IDE, website, script] -->|OpenAI format| G
    subgraph G[Context Gateway]
      S[Resolve scope] --> M[Detect mode]
      M --> B[Build context<br/>window + budget]
      B --> L[LLM client<br/>LiteLLM]
      L --> R[Record turn]
      R --> Q[Enqueue summary job]
    end
    L -->|Converse / provider API| P[(Bedrock or<br/>any provider)]
    B <--> ST[(Postgres<br/>sessions, messages)]
    B <--> RC[(Redis cache<br/>optional)]
    Q --> W[Worker<br/>procrastinate] --> ST
```

### 6.1 Non-streaming lifecycle (`stream` false or absent)

1. Middleware assigns `request_id` (reuse incoming `X-Request-Id` if present and sane).
2. Auth check (section 8.1).
3. Parse body into `ChatCompletionRequest` (section 8.3). Validation failure gives a 400
   OpenAI-style error.
4. Resolve scope (7.1). Detect mode (7.2). Resolve model alias (7.11).
5. If mode is `off`: forward messages unchanged, skip storage entirely, return.
6. Memory phase (wrapped in try/except, see 7.9 degraded mode):
   a. `session = store.get_or_create_session(scope)`
   b. Stateless: `window = store.load_window(session.id, max_turns=CTX_WINDOW_MAX_TURNS)`.
      Stateful: no window load.
   c. `plan = ContextBuilder.build(...)` gives messages to send, messages to record, report.
7. Call `llm.complete(model=..., messages=plan.messages, params=passthrough_params)`.
   On LLM error: return OpenAI-style error with mapped status (8.6). Record nothing.
8. Set `response["model"]` to the model name the client sent (not the provider id).
9. Record: `store.append_messages(session.id, plan.to_record + [clean(response message)])`.
   Failure here is logged, the response is still returned (status stays as computed).
10. If mode is `stateless` and the session's unsummarized tokens after the append exceed
    `CTX_SUMMARY_TRIGGER_TOKENS`: `queue.enqueue_summarize(session.id)`. Failures logged.
11. Return JSON with memory report headers (8.4).

### 6.2 Streaming lifecycle (`stream: true`)

Steps 1 to 6 are the same. Then:

7. `iterator = await llm.stream(...)`. Pull the **first chunk before** creating the
   response, so connection and auth errors still return a normal JSON error with the right
   status code.
8. Return `StreamingResponse(media_type="text/event-stream")` with headers
   `Cache-Control: no-cache`, `X-Accel-Buffering: no`, plus the memory report headers.
9. For each chunk: feed it to `StreamAccumulator`, rewrite `chunk["model"]` to the client's
   model name, emit `data: <json>\n\n`. After the last chunk emit `data: [DONE]\n\n`.
10. A mid-stream upstream error emits `data: {"error": {...}}\n\n` then ends the stream.
    Nothing is recorded.
11. Recording and summary enqueue happen in a Starlette `BackgroundTask` attached to the
    response, and only if the accumulator reports the stream completed normally. A client
    disconnect means incomplete, so nothing is recorded.

---

## 7. Algorithms (implement exactly)

### 7.1 Scope resolution (`core/scope.py`)

Precedence, first match wins:

| Field | 1st | 2nd | 3rd | Default |
|---|---|---|---|---|
| app | header `X-Memory-App` | path `{app}` | (none) | `CTX_DEFAULT_APP` (`"default"`) |
| user | header `X-Memory-User` | path `{user}` | body field `user` | `"anonymous"` |
| session | header `X-Memory-Session` | path `{session}` | (none) | `"auto"` |

Validation: each value must match `^[A-Za-z0-9._:@-]{1,128}$`. Otherwise raise
`GatewayError(400, "invalid_scope", ...)` naming the field. Empty header values count as
absent.

`"auto"` session key resolution happens **after** mode detection:

- Stateful mode: `session_key = "fp-" + sha256(canonical_json(first_user_message.content))[:16]`
  where `canonical_json` is `json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
  If there is no user message, use the first non-system message. This gives each new chat
  in an IDE its own session automatically, and the same chat maps to the same session on
  every call.
- Stateless mode: `session_key = "default"` (one continuous conversation per app+user).
- Off mode: no session.

Return a frozen dataclass `Scope(app: str, user: str, session: str)`.

### 7.2 Client mode detection (`core/modes.py`)

Header `X-Memory-Mode`: `auto` (default), `stateless`, `stateful`, `off`. Any other value
is a 400 `invalid_request_error` with code `invalid_memory_mode`.

`auto` rule: if any message has role in `{"assistant", "tool", "function"}` the client is
`stateful`, otherwise `stateless`.

`split_leading_system(messages) -> (leading_system, rest)`: leading system is the longest
prefix whose roles are in `{"system", "developer"}`.

Request validation (raise 400 `invalid_request_error`):
- `messages` must be a non-empty list of dicts, each with a string `role`.
- In `stateless` mode (detected or forced), `rest` must contain at least one `user` message.

### 7.3 Turn grouping (`core/turns.py`)

```
group_turns(messages) -> list[list[dict]]:
    turns = []
    for m in messages:
        if m.role == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    return turns
```

`flatten_turns(turns) -> list[dict]` concatenates. Callers pass non-system messages only.

### 7.4 Context assembly (`core/context.py`)

`ContextBuilder(counter: TokenCounter, settings)` with
`build(mode, request_messages, window | None, session | None, model) -> ChatPlan`.

`ChatPlan` fields: `messages` (to send), `to_record` (to store before the response
message), `report: ContextReport`.

`ContextReport` fields: `mode`, `session_key`, `history_turns`, `dropped_turns`,
`summary_used: bool`, `prompt_tokens` (estimate of `messages`), `over_budget: bool`.

**Stateless:**

```
leading, rest = split_leading_system(request_messages)
current = rest                                   # the new input, sent as-is
memory_block = []
if window.summary:
    memory_block = [{"role": "system", "content": MEMORY_BLOCK_TEMPLATE.format(summary=window.summary)}]
fixed = leading + memory_block + current
available = budget - counter.count(fixed, model)
kept = []
for turn in reversed(window.turns):              # newest first
    t = turn.tokens
    if t > available: break                      # stop at first turn that does not fit
    kept.insert(0, turn); available -= t
messages = leading + memory_block + flatten(kept) + current
dropped = len(window.turns) - len(kept)
over_budget = counter.count(fixed, model) > budget
to_record = [m for m in current if m.role not in ("system", "developer")]
```

`MEMORY_BLOCK_TEMPLATE`:

```
[Conversation memory] The following summarizes earlier parts of this conversation. It was
written by the context gateway. Treat it as background; the most recent messages follow.
<summary>
{summary}
</summary>
```

Note the memory block goes at the **end of the leading system block**, before any
non-system message.

**Stateful:**

```
leading, rest = split_leading_system(request_messages)
turns = group_turns(rest)
current_turn = turns[-1]; prior = turns[:-1]
fixed = leading + current_turn
available = budget - count(fixed)
kept = newest prior turns that fit (same loop as above, using per-turn counts computed now)
messages = leading + flatten(kept) + current_turn
dropped = len(prior) - len(kept)
to_record:
    if session.last_seq == 0:  all messages in rest whose role is not system/developer (backfill)
    else: messages in rest after the last assistant message (the new user input or tool
          results); if there is no assistant message, all of rest
    (always excluding system/developer roles)
summary_used = False
```

**Off:** `messages = request_messages`, `to_record = []`, report with zeros.

**Degraded:** same as off but `report.mode = "degraded"`.

Never reorder messages. Never modify message dicts in place (copy when needed).

### 7.5 Recording (`core/recorder.py` + store)

- System and developer messages are never stored.
- The assistant response is cleaned with `clean_assistant_message(choice_message)`: keep only
  keys `role`, `content`, `name`, `tool_calls`, `function_call`, `refusal` when present and
  not None (keep `content` even when None if `tool_calls` exist). Role forced to
  `"assistant"`.
- Only `choices[0]` is recorded when `n > 1`.
- Do not record if there are no choices or the response has no message.
- Each stored message gets `token_count = counter.count([message], model)` and the
  provider-facing `model` string.
- Turn assignment happens inside `append_messages` using the session's `last_turn`:
  `if role == "user" or last_turn == 0: last_turn += 1`. Then `turn_index = last_turn`.
  `seq` increments by 1 per message from `last_seq`.

### 7.6 Rolling summary (`core/summary.py`)

Trigger (after recording, stateless mode only):
`window.unsummarized_tokens_after_append > CTX_SUMMARY_TRIGGER_TOKENS`. `append_messages`
returns the new `unsummarized_tokens` so no extra query is needed.

`Summarizer(store, llm, counter, settings).run(session_id) -> SummaryOutcome`:

```
window = store.load_window(session_id, max_turns=None)       # all unsummarized turns
if window is None or window.unsummarized_tokens <= trigger: return SKIPPED
# keep the newest turns within keep_recent budget, always keep at least 1 turn
keep = []; budget = CTX_SUMMARY_KEEP_RECENT_TOKENS
for turn in reversed(window.turns):
    if keep and turn.tokens > budget: break
    keep.insert(0, turn); budget -= turn.tokens
candidates = window.turns[: len(window.turns) - len(keep)]  # oldest first
# cap work per job
fold = []; acc = 0
for turn in candidates:
    if fold and acc + turn.tokens > CTX_SUMMARY_MAX_FOLD_TOKENS: break
    fold.append(turn); acc += turn.tokens
if not fold: return SKIPPED
text = await llm.complete(model=summary_model, messages=build_summary_prompt(window.summary, fold),
                          params={"temperature": 0, "max_tokens": CTX_SUMMARY_MAX_OUTPUT_TOKENS})
summary = first choice content, stripped; empty -> raise SummaryError (job retries)
ok = await store.apply_summary(session_id, summary=summary,
                               through_turn=fold[-1].index,
                               expected_through_turn=window.summarized_through_turn)
if not ok: return STALE            # another job won the race; no error
if more candidates remain beyond fold: return PARTIAL (caller re-enqueues)
return APPLIED
```

`apply_summary` is a compare-and-set: update only where
`summarized_through_turn = expected_through_turn`. Return whether a row changed.

Summary model: `CTX_SUMMARY_MODEL`, falling back to `CTX_DEFAULT_MODEL` (both resolved
through aliases). If neither is set, log a warning and skip summarization.

Prompt (`build_summary_prompt`):

System message:

```
You maintain a running summary of a conversation between a user and an AI assistant.
Merge the new turns into the existing summary.
Keep: facts about the user, stated preferences, decisions, names, identifiers, file paths,
code symbols, numbers, constraints, and open questions or pending tasks.
Drop: greetings, filler, and anything superseded by newer information (say what changed).
Write in third person ("The user ..."). Use short bullet points grouped by topic.
Never invent details. Stay under 300 words.
Output only the updated summary.
```

User message:

```
Existing summary:
{previous_summary or "(none)"}

New turns to merge:
{transcript}
```

Transcript format: one block per turn, `### Turn {index}` then one line per message:
`USER: ...`, `ASSISTANT: ...`, `TOOL({name or tool_call_id}): ...`. Assistant tool calls
render as `ASSISTANT called {function name}({arguments})`. Text comes from
`flatten_content` (7.10). Truncate each message text to 4000 characters with a
`... [truncated]` marker.

### 7.7 Stream accumulation (`core/stream.py`)

`StreamAccumulator.add(chunk: dict)` per chunk:

- Take `choices[0]` if present (ignore other choice indexes). From `delta`:
  - `role`: remember it.
  - `content` (str): append to a list of text parts.
  - `tool_calls`: list of partial calls. Merge by `index`: set `id`, `type` and
    `function.name` when present; concatenate `function.arguments` strings.
  - `function_call` (legacy): merge name and concatenate arguments.
  - `finish_reason` not None: store it, mark `finished = True`.
- Chunks with a `usage` field: store the usage dict.

`message()` returns `{"role": "assistant", "content": text or None, "tool_calls": [...]}`
(omit empty keys, keep `content: None` when there are tool calls and no text). Tool calls are
ordered by index.

`completed` property: `finished` is True (a finish_reason was seen). Some providers end
without one, so the streaming loop also calls `mark_done()` when the upstream iterator is
exhausted without error.

`sse_event(data: dict | str) -> bytes`: `b"data: " + json + b"\n\n"`; `sse_done()` returns
`b"data: [DONE]\n\n"`. JSON uses `separators=(",", ":")` and `ensure_ascii=False`.

### 7.8 Redis cache (`store/cache.py`)

`CachedSessionStore(inner: SessionStore, redis, ttl_seconds)` implements the full
`SessionStore` protocol by delegating, plus:

- `load_window(session_id, max_turns)`: only cache when `max_turns` equals the configured
  default (the hot path). Key: `cg:window:{session_id}:{max_turns}`. Value: JSON of the
  Window. TTL `CTX_CACHE_TTL_SECONDS`.
- Invalidate (delete all `cg:window:{session_id}:*` keys, or track the one key you write)
  on `append_messages`, `apply_summary`, `delete_session`.
- Any Redis error is logged at WARNING and falls through to `inner`. Redis must never fail
  a request.

### 7.9 Degraded mode

Wrap step 6 (memory phase) in `try/except Exception`. On failure: log with
`exc_info=True`, set mode to `degraded`, forward `request_messages` unchanged, skip
recording. The client still gets an answer. `X-Memory-Mode: degraded`.

LLM errors are **not** degraded mode; they are returned to the client as errors.

### 7.10 Content helpers (`core/content.py`)

`flatten_content(content) -> str`:
- `str` returns as-is. `None` returns `""`.
- list of parts: join `text` of parts with type `text` by `"\n"`; image parts become
  `[image]`; other part types become `[{type}]`.
- anything else: `json.dumps`.

### 7.11 Model routing (`core/routing.py`)

`resolve_model(requested: str, settings) -> str` (provider-facing LiteLLM model string):
1. `requested` in `CTX_MODEL_ALIASES` gives the mapped value.
2. `requested` in `("", "default")` and `CTX_DEFAULT_MODEL` set: resolve
   `CTX_DEFAULT_MODEL` through aliases (it may itself be an alias), else use it raw.
3. `CTX_ALLOW_UNLISTED_MODELS` true: pass `requested` through unchanged.
4. Otherwise 400 `invalid_request_error`, code `model_not_found`.

`/v1/models` returns the alias names (plus `"default"` when `CTX_DEFAULT_MODEL` is set):

```json
{"object": "list", "data": [{"id": "claude", "object": "model", "created": 0, "owned_by": "context-gateway"}]}
```

### 7.12 Token counting (`core/tokens.py`)

```python
class TokenCounter(Protocol):
    def count(self, messages: Sequence[dict], model: str | None) -> int: ...
```

- `LiteLLMTokenCounter`: lazily imports litellm, calls
  `litellm.token_counter(model=model or "", messages=list(messages))`. On any exception,
  fall back to `HeuristicTokenCounter`. LiteLLM bundles tokenizers, so this works offline.
- `HeuristicTokenCounter`: per message `4 + ceil(len(text) / 4)` where text is
  `flatten_content(content)` plus `json.dumps(tool_calls)` when present. Deterministic.
  Tests use this one.
- Settings `CTX_TOKEN_COUNTER=litellm|heuristic` (default `litellm`).

Counts are estimates. The default budget leaves headroom; do not try to be exact.

---

## 8. HTTP API contract

### 8.1 Authentication

- `CTX_API_KEYS` empty: auth disabled.
- Otherwise every route except `/healthz` and `/readyz` requires
  `Authorization: Bearer <key>` or `X-Api-Key: <key>` matching one configured key.
  Compare with `hmac.compare_digest`. Failure: 401 `authentication_error`, code
  `invalid_api_key`.
- Provider credentials are never sent by clients. They live in the gateway environment
  (AWS env vars or profile for Bedrock, `OPENAI_API_KEY`, and so on).

### 8.2 Routes

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | Chat with memory. Scope from headers/body. |
| POST | `/scope/{app}/{user}/{session}/v1/chat/completions` | Same, scope from path (headers still override). |
| GET | `/v1/models` | Model aliases. |
| GET | `/scope/{app}/{user}/{session}/v1/models` | Same (IDE plugins call `{base}/models`). |
| GET | `/memory/v1/sessions/{app}/{user}/{session}` | Session info. 404 if missing. |
| GET | `/memory/v1/sessions/{app}/{user}/{session}/messages?limit=50` | Most recent stored messages, ascending. `limit` 1 to 500. |
| DELETE | `/memory/v1/sessions/{app}/{user}/{session}` | Delete session and messages. 204, or 404 if missing. |
| POST | `/memory/v1/sessions/{app}/{user}/{session}/summarize` | Enqueue a summary job now. 202 `{"enqueued": true}`. 404 if missing. |
| POST | `/memory/v1/context/preview` | Dry run of context assembly. No LLM call, no writes. |
| GET | `/healthz` | Liveness: `{"status": "ok"}`. |
| GET | `/readyz` | Readiness: checks store (`SELECT 1` for Postgres) and Redis ping if configured. 200 or 503 with per-check details. |

For the admin routes, the path segments are literal scope values (use `default` explicitly;
`auto` is not resolved there).

Session info response:

```json
{
  "app": "intellij", "user": "shreyamsh", "session": "fp-3f9a1c0b2d4e5f60",
  "id": "uuid", "summary": "text or null", "summarized_through_turn": 4,
  "turn_count": 9, "message_count": 21, "unsummarized_tokens": 3120,
  "created_at": "iso8601", "updated_at": "iso8601"
}
```

Preview request: the chat completions body plus an optional `"scope": {"app", "user",
"session"}` object (headers also work) and optional `"memory_mode"`. Use
`store.find_session` (never create); a missing session behaves as an empty window. Response:

```json
{"mode": "stateless", "session": "default", "model": "provider/model-id",
 "messages": [ ... ], "report": { ...ContextReport... }}
```

### 8.3 Chat request model

```python
class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = ""
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    user: str | None = None
```

Passthrough params to the LLM: all fields except `model`, `messages`, `stream`, `user`
(including extras such as `temperature`, `max_tokens`, `tools`, `tool_choice`,
`response_format`, `stop`, `top_p`). Drop `stream_options` when not streaming.

### 8.4 Memory report headers (on every chat response, including streams)

| Header | Value |
|---|---|
| `X-Memory-Mode` | `stateless`, `stateful`, `off`, `degraded` |
| `X-Memory-App` / `X-Memory-User` / `X-Memory-Session` | resolved scope (session is the resolved key, never `auto`) |
| `X-Memory-History-Turns` | history turns included |
| `X-Memory-Dropped-Turns` | history turns dropped for budget |
| `X-Memory-Summary` | `used` or `none` |
| `X-Memory-Prompt-Tokens` | estimated prompt tokens sent |
| `X-Request-Id` | request id |

In `off` and `degraded`, scope headers are omitted when not resolved, counts are `0`.

### 8.5 CORS

`CTX_CORS_ORIGINS` (comma-separated). Empty disables CORS middleware. When set, expose all
`X-Memory-*` and `X-Request-Id` headers.

### 8.6 Errors

Always OpenAI-shaped:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": "invalid_scope"}}
```

| Situation | Status | type | code |
|---|---|---|---|
| Body validation failure | 400 | invalid_request_error | `invalid_request` |
| Bad scope value | 400 | invalid_request_error | `invalid_scope` |
| Bad `X-Memory-Mode` | 400 | invalid_request_error | `invalid_memory_mode` |
| Unknown model (unlisted not allowed) | 400 | invalid_request_error | `model_not_found` |
| Missing/invalid gateway key | 401 | authentication_error | `invalid_api_key` |
| Session not found (admin) | 404 | not_found_error | `session_not_found` |
| Upstream LLM error with status 400 to 499 | same status | upstream_error | `upstream_error` |
| Upstream LLM error, 5xx or unknown | 502 | upstream_error | `upstream_error` |
| Unhandled exception | 500 | server_error | `internal_error` (no stack trace in body) |

`LLMError(status_code: int | None, message: str)` is raised by LLM clients. The LiteLLM
client maps LiteLLM exceptions by reading `getattr(exc, "status_code", None)`.

FastAPI's `RequestValidationError` and `HTTPException` must also render in this shape.

---

## 9. Data model

Use SQLAlchemy Core tables in `store/tables.py` with a shared `MetaData`. Table names are
prefixed `cg_` so they never collide with procrastinate's tables in the same database.

```sql
CREATE TABLE cg_sessions (
    id                       UUID PRIMARY KEY,
    app_id                   TEXT NOT NULL,
    user_id                  TEXT NOT NULL,
    session_key              TEXT NOT NULL,
    summary                  TEXT NULL,
    summarized_through_turn  INTEGER NOT NULL DEFAULT 0,
    last_seq                 INTEGER NOT NULL DEFAULT 0,
    last_turn                INTEGER NOT NULL DEFAULT 0,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_cg_sessions_scope UNIQUE (app_id, user_id, session_key)
);

CREATE TABLE cg_messages (
    id           BIGSERIAL PRIMARY KEY,
    session_id   UUID NOT NULL REFERENCES cg_sessions(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    turn_index   INTEGER NOT NULL,
    role         TEXT NOT NULL,
    message      JSONB NOT NULL,
    token_count  INTEGER NOT NULL,
    model        TEXT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_cg_messages_seq UNIQUE (session_id, seq)
);

CREATE INDEX ix_cg_messages_session_turn ON cg_messages (session_id, turn_index);
```

Store operations (Postgres):

- `get_or_create_session(scope)`: one statement,
  `INSERT ... VALUES (gen uuid4 in Python, ...) ON CONFLICT (app_id, user_id, session_key)
  DO UPDATE SET updated_at = now() RETURNING *`.
- `append_messages(session_id, messages)`: one transaction.
  `SELECT last_seq, last_turn, summarized_through_turn FROM cg_sessions WHERE id = :id FOR UPDATE`,
  assign seq and turn (7.5), bulk insert, update `last_seq`, `last_turn`, `updated_at`,
  then compute `unsummarized_tokens = SUM(token_count) WHERE turn_index > summarized_through_turn`
  and return `AppendResult(last_seq, last_turn, unsummarized_tokens)`. The row lock makes
  concurrent appends to the same session safe.
- `load_window(session_id, max_turns)`: read session row; select messages with
  `turn_index > summarized_through_turn`, and when `max_turns` is set only the newest
  `max_turns` turns (`turn_index > last_turn - max_turns`). Order by `seq`. Group by
  `turn_index` into `Turn(index, messages, tokens)`. `unsummarized_tokens` is the sum over
  **all** unsummarized messages (separate aggregate query), `truncated` is True when turns
  were cut by `max_turns`. Returns `None` if the session does not exist.
- `apply_summary(...)`: `UPDATE cg_sessions SET summary=:s, summarized_through_turn=:t,
  updated_at=now() WHERE id=:id AND summarized_through_turn=:expected` and return
  `rowcount == 1`.
- `delete_session(session_id)`: delete the session row (messages cascade).

`InMemorySessionStore` implements identical semantics with dicts and an `asyncio.Lock`.
Every behavior test for the Postgres store must also pass for the in-memory store (write
one shared, parametrized contract test suite).

### 9.1 Store protocol (`store/base.py`)

```python
@dataclass(frozen=True)
class SessionRecord:
    id: UUID; app: str; user: str; session_key: str
    summary: str | None; summarized_through_turn: int
    last_seq: int; last_turn: int
    created_at: datetime; updated_at: datetime

@dataclass(frozen=True)
class NewMessage:
    message: dict[str, Any]; role: str; token_count: int; model: str | None

@dataclass(frozen=True)
class StoredMessage:
    seq: int; turn_index: int; role: str; message: dict[str, Any]
    token_count: int; model: str | None; created_at: datetime

@dataclass(frozen=True)
class AppendResult:
    last_seq: int; last_turn: int; unsummarized_tokens: int

class SessionStore(Protocol):
    async def get_or_create_session(self, scope: Scope) -> SessionRecord: ...
    async def find_session(self, scope: Scope) -> SessionRecord | None: ...
    async def get_session(self, session_id: UUID) -> SessionRecord | None: ...
    async def load_window(self, session_id: UUID, *, max_turns: int | None) -> Window | None: ...
    async def append_messages(self, session_id: UUID, messages: Sequence[NewMessage]) -> AppendResult: ...
    async def apply_summary(self, session_id: UUID, *, summary: str, through_turn: int,
                            expected_through_turn: int) -> bool: ...
    async def list_messages(self, session_id: UUID, *, limit: int) -> list[StoredMessage]: ...
    async def delete_session(self, session_id: UUID) -> bool: ...
    async def ping(self) -> None: ...
    async def close(self) -> None: ...
```

`Window` (in `core/types.py`): `session_id`, `summary`, `summarized_through_turn`,
`turns: list[Turn]`, `unsummarized_tokens`, `truncated`. Must be JSON round-trippable for
the cache (`to_dict` / `from_dict`).

### 9.2 Migrations

- Alembic lives in `src/context_gateway/migrations` so it ships in the wheel. Build the
  Alembic `Config` programmatically in `cli.py` (`script_location` = package path,
  `sqlalchemy.url` = sync URL). No `alembic.ini` required at runtime (you may add one at the
  repo root for developer convenience pointing to the same location).
- `0001_initial`: the two tables and index above.
- `0002_procrastinate`: apply the procrastinate schema with
  `op.get_bind().exec_driver_sql(procrastinate.schema.SchemaManager.get_schema())`.
  Use `exec_driver_sql` (not `op.execute(text(...))`) so SQLAlchemy does not parse `:` or
  `%` in the SQL. Downgrade may raise `NotImplementedError`.
- Note in the README that upgrading procrastinate later requires applying its own migration
  files in a new Alembic revision.

### 9.3 Database URLs

One setting, `CTX_DATABASE_URL`, in libpq form: `postgresql://user:pass@host:5432/db`.
Derive:
- async SQLAlchemy: `postgresql+psycopg://...`
- sync (Alembic): `postgresql+psycopg://...` with a sync engine
- procrastinate `PsycopgConnector(conninfo=...)`: the plain `postgresql://` form

Accept inputs that already use `postgresql+psycopg://` or `postgres://` and normalize.

---

## 10. Configuration (`config.py`)

`Settings(BaseSettings)` with `env_prefix="CTX_"`, `.env` support, cached via
`functools.lru_cache` getter. All of these:

| Env var | Type | Default | Notes |
|---|---|---|---|
| `CTX_STORE` | `postgres` or `memory` | `postgres` | `memory` for demos and tests |
| `CTX_DATABASE_URL` | str | `postgresql://cg:cg@localhost:5432/cg` | |
| `CTX_REDIS_URL` | str or empty | empty | empty disables cache |
| `CTX_JOB_BACKEND` | `inline` or `procrastinate` | `inline` | compose uses `procrastinate` |
| `CTX_LLM_BACKEND` | `litellm` or `echo` | `litellm` | `echo` for demos/e2e without credentials |
| `CTX_TOKEN_COUNTER` | `litellm` or `heuristic` | `litellm` | |
| `CTX_DEFAULT_MODEL` | str | empty | alias or LiteLLM model string |
| `CTX_MODEL_ALIASES` | JSON object | `{}` | e.g. `{"claude":"bedrock/converse/us.anthropic.claude-sonnet-4-6"}` |
| `CTX_ALLOW_UNLISTED_MODELS` | bool | `true` | |
| `CTX_SUMMARY_MODEL` | str | empty | falls back to default model |
| `CTX_CONTEXT_BUDGET_TOKENS` | int | `24000` | |
| `CTX_SUMMARY_TRIGGER_TOKENS` | int | `8000` | |
| `CTX_SUMMARY_KEEP_RECENT_TOKENS` | int | `3000` | |
| `CTX_SUMMARY_MAX_FOLD_TOKENS` | int | `12000` | |
| `CTX_SUMMARY_MAX_OUTPUT_TOKENS` | int | `800` | |
| `CTX_WINDOW_MAX_TURNS` | int | `200` | |
| `CTX_CACHE_TTL_SECONDS` | int | `600` | |
| `CTX_API_KEYS` | comma list | empty | empty disables auth |
| `CTX_DEFAULT_APP` | str | `default` | |
| `CTX_CORS_ORIGINS` | comma list | empty | |
| `CTX_LOG_LEVEL` | str | `INFO` | |
| `CTX_LOG_CONTENT` | bool | `false` | never log message text unless true |
| `CTX_HOST` / `CTX_PORT` | str / int | `0.0.0.0` / `8080` | used by `serve` |

Validate at startup: budget > 0, keep_recent < trigger, aliases is a dict of str to str.
Invalid config must fail fast with a clear message.

Provider credentials are standard env vars read by LiteLLM/boto3 (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION_NAME`, `AWS_PROFILE`,
`OPENAI_API_KEY`, ...). Do not wrap them in `CTX_` settings.

`.env.example` documents every variable with a one-line comment and an example Bedrock
alias. Bedrock model ids are region and account specific; the example must say "replace with
a model id enabled in your Bedrock console".

---

## 11. Components and interfaces

### 11.1 LLM clients (`llm/`)

```python
class LLMClient(Protocol):
    async def complete(self, *, model: str, messages: list[dict], params: dict) -> dict: ...
    async def stream(self, *, model: str, messages: list[dict], params: dict) -> AsyncIterator[dict]: ...
    async def close(self) -> None: ...
```

- `LiteLLMClient.complete`: `resp = await litellm.acompletion(model=model, messages=messages,
  drop_params=True, **params)`; return `resp.model_dump()`.
- `LiteLLMClient.stream`: `await litellm.acompletion(..., stream=True)`; return an async
  generator yielding `chunk.model_dump(exclude_none=True)` dicts.
- Both wrap LiteLLM exceptions into `LLMError` (status from `exc.status_code` when present).
- `EchoLLMClient`: deterministic, no network. Replies with
  `"echo: {last user text} | history_messages={number of non-system messages minus 1} | summary={yes/no}"`,
  where summary is yes when a system message contains `<summary>`. Supports streaming by
  splitting the reply into 3 to 5 chunks with OpenAI chunk shape and a final chunk with
  `finish_reason: "stop"`. For summary requests (detected by the summary system prompt) it
  returns a short deterministic summary such as `"- summary of N turns"`. Response shape
  must match OpenAI exactly (`id`, `object`, `created`, `model`, `choices`, `usage`).

### 11.2 Job queues (`jobs/`)

```python
class JobQueue(Protocol):
    async def enqueue_summarize(self, session_id: UUID) -> None: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...
```

- `InlineJobQueue(runner: Callable[[UUID], Awaitable[SummaryOutcome]])`: one asyncio task per
  session at a time (skip if one is running for that session). Keep task references in a
  set. Re-enqueue once when the outcome is `PARTIAL`. Log exceptions. `close()` waits for
  pending tasks up to 10 seconds, then cancels.
- `ProcrastinateJobQueue(app)`: `await summarize_session.configure(queueing_lock=str(id),
  lock=str(id)).defer_async(session_id=str(id))`. Catch
  `procrastinate.exceptions.AlreadyEnqueued` and treat as success. `start()` opens the app
  connector (`await app.open_async()`), `close()` closes it.

### 11.3 Worker (`worker/`)

- `worker/app.py`: `app = procrastinate.App(connector=PsycopgConnector(conninfo=<derived>),
  import_paths=["context_gateway.worker.tasks"])`. Settings read at import.
- `worker/tasks.py`: `@app.task(name="summarize_session", queue="summaries", retry=3)`
  `async def summarize_session(session_id: str)`. Builds store, LLM client and counter once
  per process (module-level lazy singletons), runs `Summarizer.run`, and on `PARTIAL`
  defers itself again.
- `context-gateway worker` runs `await app.run_worker_async(queues=["summaries"])` inside
  `async with app.open_async()`.

### 11.4 App factory (`app.py`, `deps.py`)

```python
def create_app(settings: Settings | None = None, *, llm: LLMClient | None = None,
               store: SessionStore | None = None, queue: JobQueue | None = None,
               counter: TokenCounter | None = None) -> FastAPI
```

- Lifespan builds anything not injected from settings, wires `Summarizer` into the inline
  queue, starts the queue, stores everything in `app.state.deps` (`Deps` dataclass), and
  closes all of it on shutdown.
- Route handlers get dependencies through small `Depends` getters reading `app.state.deps`.
  No module-level globals in request paths.
- Middleware: request id, structured access log line per request.
- Exception handlers for `GatewayError`, `LLMError`, `RequestValidationError`,
  `HTTPException`, and `Exception`.

### 11.5 CLI (`cli.py`)

`argparse` subcommands:
- `serve [--host] [--port] [--reload]`: `uvicorn.run("context_gateway.app:create_app", factory=True, ...)`.
- `migrate`: Alembic `upgrade head`.
- `worker`: run the procrastinate worker.

---

## 12. Logging and privacy

- `logging` with a single formatter: `ts level logger msg key=value ...`.
- One access log line per chat request with: request_id, app, user, session, mode,
  history_turns, dropped_turns, prompt_tokens, model, status, latency_ms.
- Never log message content, summaries, or provider credentials unless
  `CTX_LOG_CONTENT=true` (content only, never credentials).
- Never echo provider exception details that might contain credentials. LiteLLM messages
  are generally safe, but truncate them to 500 characters.

---

## 13. Security notes

- Gateway keys compared in constant time.
- Scope values are validated by regex, so they are safe as path segments, cache keys and
  log fields.
- SQL only through SQLAlchemy Core with bound parameters.
- Docker image runs as a non-root user.
- The gateway holds provider credentials; document that it should run on a private network
  or behind TLS with `CTX_API_KEYS` set when exposed.

---

## 14. Testing strategy

Use `pytest` with `asyncio_mode = "auto"` (already configured). Use `httpx.AsyncClient`
with `httpx.ASGITransport(app=...)` for API tests, wrapped in the app lifespan (use
`asgi-lifespan`'s `LifespanManager` if needed, or call the lifespan context directly; add
`asgi-lifespan` as a dev dependency if you use it).

### 14.1 Fakes (`tests/fakes.py`)

`FakeLLM(LLMClient)`:
- Scripted replies (`queue_text("...")`, `queue_tool_call(name, args)`, `queue_error(status, msg)`),
  falling back to a default reply.
- Records every call (`calls: list[Call(model, messages, params, stream)]`) for assertions.
- Streaming yields OpenAI-shaped chunks, splitting text across several chunks, tool call
  arguments across several chunks, final chunk with `finish_reason`.
- Option to raise mid-stream.

Use `HeuristicTokenCounter`, `InMemorySessionStore` and `InlineJobQueue` for API tests.

### 14.2 Required unit tests (`tests/unit/`)

Scope (`test_scope.py`):
1. Header beats path beats body beats default, for each field.
2. Empty header value is treated as absent.
3. Invalid characters and over-long values raise `invalid_scope` naming the field.
4. Auto session in stateful mode is `fp-` plus 16 hex chars and is stable for the same first
   user message; different first messages give different keys.
5. Auto session in stateless mode is `default`.

Modes and turns (`test_modes.py`, `test_turns.py`):
6. Only system + user gives stateless; any assistant/tool gives stateful.
7. `X-Memory-Mode` override works; invalid value raises.
8. `split_leading_system` handles `developer`, no system, and a system message in the
   middle (stays in `rest`).
9. `group_turns` keeps assistant tool_calls with their tool results; leading assistant
   message forms its own turn.

Context builder (`test_context.py`):
10. Stateless with empty window sends leading system + current only.
11. Stateless with history sends history in order between system and current.
12. Summary present: memory block is the last message of the leading system block and
    contains `<summary>`.
13. Budget trimming drops oldest turns first and reports `dropped_turns`.
14. A turn that does not fit stops inclusion (no skipping to older smaller turns).
15. Tool-call turn is kept or dropped as a whole.
16. Over-budget fixed part sets `over_budget` and includes no history.
17. Stateful: no stored history injected; budget trims client's oldest prior turns.
18. Stateful `to_record`: backfill when `last_seq == 0`; tail after last assistant
    otherwise; system messages excluded.
19. Off mode passes messages through untouched.
20. Input message dicts are not mutated.

Recorder and content (`test_recorder.py`, `test_content.py`):
21. `clean_assistant_message` keeps only allowed keys, keeps `content: None` with tool calls.
22. `flatten_content` for str, None, text parts, image parts, unknown parts.

Stream (`test_stream.py`):
23. Text deltas concatenate; role captured; finish_reason sets completed.
24. Tool call deltas merge by index across chunks with argument concatenation.
25. Two parallel tool calls (index 0 and 1) interleaved merge correctly.
26. `sse_event` and `sse_done` byte format.

Summary (`test_summary.py`, using FakeLLM + InMemorySessionStore):
27. Below trigger gives SKIPPED and no LLM call.
28. Above trigger folds oldest turns, keeps newest within keep budget, applies summary,
    `summarized_through_turn` updated, prompt contains previous summary and transcript.
29. Always keeps at least one turn even if it alone exceeds keep budget.
30. `max_fold_tokens` cap gives PARTIAL and leaves later turns unsummarized.
31. CAS conflict (summary changed meanwhile) gives STALE without error.
32. Empty LLM output raises `SummaryError`.

Routing and tokens (`test_routing.py`, `test_tokens.py`):
33. Alias, default, passthrough, and rejection paths.
34. Heuristic counter is deterministic and counts tool calls.

Config (`test_config.py`):
35. JSON aliases parse; invalid budget/keep/trigger combinations fail validation.

### 14.3 Required API tests (`tests/api/`)

36. Stateless conversation: call 1 "my name is Shreyamsh", call 2 "what is my name": the
    second LLM call (FakeLLM.calls[1]) contains call 1's user and assistant messages before
    the new user message. Headers report `stateless`, history turns 1.
37. Stateful client sending full history: FakeLLM receives exactly the client's messages
    (no duplication); store holds the conversation without duplicates after 3 calls.
38. Scoped path route resolves scope from path; header overrides path.
39. Different users and sessions are isolated.
40. Streaming: SSE events parse, final `[DONE]`, assembled text equals the scripted reply,
    `model` in chunks equals the requested name, turn recorded after completion.
41. Streaming with tool calls records the assembled tool_calls message.
42. Streaming upstream error before first chunk returns JSON error with mapped status.
43. Streaming mid-stream error emits an error event and records nothing.
44. Non-streaming upstream 429 returns 429 `upstream_error`; 500 returns 502.
45. `X-Memory-Mode: off` forwards untouched and stores nothing.
46. Degraded: a store whose methods raise still returns 200 with `X-Memory-Mode: degraded`.
47. Auth: missing key 401, `Bearer` works, `X-Api-Key` works, `/healthz` open.
48. `/v1/models` lists aliases and `default`.
49. Memory admin: get session, list messages, preview (no writes, no LLM call), summarize
    returns 202, delete returns 204 then 404.
50. Summary end to end with inline queue and tiny thresholds: after enough turns the next
    request includes the memory block with the FakeLLM summary text.
51. Validation errors (empty messages, bad JSON) return OpenAI-shaped 400.

### 14.4 Integration tests (`tests/integration/`, marker `integration`)

Fixtures in `tests/conftest.py`:
- Postgres: use `TEST_DATABASE_URL` if set. Else, if Docker is available, start
  `pgvector/pgvector:pg16` with testcontainers. Else skip with a clear reason.
  Run migrations once per session (Alembic upgrade head). Truncate `cg_` tables between
  tests (`TRUNCATE cg_messages, cg_sessions CASCADE`).
- Redis: `TEST_REDIS_URL`, else testcontainers `redis:7-alpine`, else skip.

Tests:
52. Run the shared SessionStore contract suite against `PostgresSessionStore` (the same
    suite also runs against `InMemorySessionStore` in unit tests).
53. Concurrent `append_messages` on one session (e.g. 20 parallel appends) gives unique,
    gapless `seq` values.
54. `apply_summary` CAS against the real DB.
55. Migrations: upgrade head creates `cg_` tables and procrastinate tables
    (`procrastinate_jobs`); running upgrade twice is a no-op.
56. `CachedSessionStore` with real Redis: second `load_window` is served from cache (assert
    via a spy on the inner store), invalidated after `append_messages`, and Redis being down
    (point at a closed port) still returns correct data.
57. ProcrastinateJobQueue: deferring twice for the same session results in one queued job
    (`AlreadyEnqueued` handled); running the worker once (`run_worker_async(wait=False)` or
    equivalent) executes the summary against the real store with an `EchoLLMClient`.

### 14.5 Running tests in the agent sandbox

The sandbox has **no access to Docker Hub**, so testcontainers cannot pull images here.
Use local services instead:

- Postgres 16 binaries: `/usr/lib/postgresql/16/bin` (`initdb`, `pg_ctl`). Postgres refuses
  to run as root, so run it as the `postgres` system user (for example
  `runuser -u postgres -- /usr/lib/postgresql/16/bin/initdb -D <dir>`), with the data dir
  owned by `postgres`. Listen on 127.0.0.1:55432, create role and database `cg` / `cg_test`.
- Redis: `/usr/bin/redis-server --port 56379 --daemonize yes`.
- `scripts/dev-services.sh start|stop|status` must automate exactly this (and work on a
  developer laptop with Postgres and Redis installed). Data under `.devdata/` (gitignored).
- Then: `TEST_DATABASE_URL=postgresql://cg:cg@127.0.0.1:55432/cg_test TEST_REDIS_URL=redis://127.0.0.1:56379/0 uv run pytest`.

pgvector is not installed locally and is not needed in Phase 1. The compose and CI images
use `pgvector/pgvector:pg16` so Phase 2 can add vectors without changing infrastructure.

### 14.6 CI (`.github/workflows/ci.yml`)

- Trigger: push and pull_request.
- Job `lint`: `astral-sh/setup-uv`, `uv sync --locked`, `uv run ruff check .`,
  `uv run ruff format --check .`.
- Job `test`: services `pgvector/pgvector:pg16` (env POSTGRES_USER/PASSWORD/DB = cg/cg/cg_test,
  port 5432, health check `pg_isready`) and `redis:7-alpine` (health check `redis-cli ping`).
  Env `TEST_DATABASE_URL`, `TEST_REDIS_URL`. `uv sync --locked`, `uv run pytest -q`.
- Python 3.12. Cache uv.

---

## 15. Docker

`Dockerfile` (multi-stage):
- Builder: `python:3.12-slim`, copy `uv` from `ghcr.io/astral-sh/uv:0.11` image,
  `UV_COMPILE_BYTECODE=1`, `uv sync --locked --no-dev --no-install-project` (dependency
  layer), then copy source and `uv sync --locked --no-dev`.
- Runtime: `python:3.12-slim`, copy the venv and source, non-root user `app`,
  `ENV PATH=/app/.venv/bin:$PATH LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1`,
  `EXPOSE 8080`, `HEALTHCHECK` hitting `/healthz` with Python's `urllib`,
  `CMD ["context-gateway", "serve"]`.
- `.dockerignore`: `.venv`, `.git`, `.devdata`, caches, tests not needed in image.

`docker-compose.yml` services:
- `postgres`: `pgvector/pgvector:pg16`, user/password/db `cg`, volume, health check.
- `redis`: `redis:7-alpine`, health check.
- `migrate`: build `.`, `command: ["context-gateway", "migrate"]`, depends on postgres
  healthy, `restart: "no"`.
- `gateway`: build `.`, ports `8080:8080`, `env_file: .env`, environment
  `CTX_DATABASE_URL=postgresql://cg:cg@postgres:5432/cg`, `CTX_REDIS_URL=redis://redis:6379/0`,
  `CTX_JOB_BACKEND=procrastinate`, AWS env passthrough (`AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION_NAME`, `AWS_PROFILE`), optional
  read-only mount of `~/.aws` to `/home/app/.aws`, depends on `migrate` completed
  successfully and redis healthy.
- `worker`: same image and env, `command: ["context-gateway", "worker"]`, depends on
  `migrate` completed successfully.

The sandbox cannot pull images, so you cannot run compose here. Keep the files simple and
correct; validate YAML syntax with Python (`yaml.safe_load`) if PyYAML is available.

---

## 16. Coding conventions

- Type hints everywhere. `from __future__ import annotations` is fine. Dataclasses
  (`frozen=True`, `slots=True` where simple) for internal records, Pydantic for HTTP models
  and settings.
- Async for all I/O. No blocking calls in request paths (token counting is CPU-only and
  acceptable).
- Small modules, pure functions in `core/` (no I/O in `core/` except `summary.py`, which
  talks to the store and LLM through protocols).
- No global mutable state in request handling.
- Docstrings: one or two lines on public classes and functions explaining intent, not
  restating code.
- Ruff must pass with the configured rules; format with `ruff format`.
- User-facing docs (README, docs/): short, direct sentences, no em-dashes, no marketing
  filler. The repo owner explicitly prefers this style.

---

## 17. Implementation order (suggested)

1. `config.py`, `core/types.py`, `core/content.py`, `core/scope.py`, `core/modes.py`,
   `core/turns.py`, `core/tokens.py`, `core/routing.py` + their unit tests.
2. `core/context.py`, `core/recorder.py`, `core/stream.py` + unit tests.
3. `store/base.py`, `store/memory.py`, contract tests.
4. `llm/base.py`, `llm/echo.py`, `tests/fakes.py`, `core/summary.py` + tests.
5. `jobs/inline.py`, `errors.py`, `deps.py`, `app.py`, `api/*` + API tests.
6. `store/tables.py`, `store/postgres.py`, migrations, `scripts/dev-services.sh`,
   integration tests against local Postgres.
7. `store/cache.py` + Redis integration tests.
8. `llm/litellm_client.py`, `jobs/procrastinate_queue.py`, `worker/*`, `cli.py` + tests.
9. Dockerfile, compose, `.env.example`, CI, LICENSE, README, `docs/DECISIONS.md`.
10. End-to-end check (section 19, item 6).

Commit after each step that leaves tests green (small, logical commits).

---

## 18. README requirements

Sections, in this order, short and concrete:

1. One-paragraph description and the problem it solves.
2. How it works (5 bullets: scope, modes, budget, summary, headers) and a small diagram
   (mermaid is fine).
3. Quickstart with Docker Compose: copy `.env.example`, set AWS creds and a model alias,
   `docker compose up`, a `curl` example showing two calls that remember.
4. Quickstart without Docker: `uv sync`, `scripts/dev-services.sh start`,
   `context-gateway migrate`, `CTX_LLM_BACKEND=echo context-gateway serve`.
5. Use from code: Python `openai` SDK with `base_url` and `default_headers`; TypeScript
   `openai` package; plain `curl`.
6. Use from IntelliJ (Continue plugin): a `config.yaml` model entry with
   `provider: openai`, `apiBase: http://localhost:8080/scope/intellij/<you>/auto/v1`,
   `model: claude`, `apiKey: <gateway key or any string>`. Mention any plugin that accepts a
   custom OpenAI-compatible base URL works the same way.
7. Memory modes explained (stateless, stateful, off, degraded) and the `X-Memory-*`
   request and response headers.
8. Configuration reference (table of `CTX_*` vars) and provider credentials.
9. Memory admin API (routes table).
10. Development: tests (unit, integration with `TEST_*` env), lint, project layout pointer
    to CLAUDE.md.
11. Roadmap (Phases 2 to 4) and license.

---

## 19. Definition of done (Phase 1)

All must be true before you report completion:

1. `uv run ruff check .` and `uv run ruff format --check .` pass.
2. `uv run pytest` passes with **zero skipped integration tests** when run with local
   services (`TEST_DATABASE_URL`, `TEST_REDIS_URL` set via `scripts/dev-services.sh`).
3. Every numbered test in section 14 exists (names may differ; keep a comment with the
   spec number, e.g. `# spec 36`).
4. `context-gateway migrate` works against the local Postgres; running it twice is safe.
5. The app imports and starts without LiteLLM network access
   (`LITELLM_LOCAL_MODEL_COST_MAP=True`).
6. End-to-end manual check, results pasted in your final report:
   - Start Postgres + Redis (`scripts/dev-services.sh start`), run `migrate`.
   - Start `context-gateway worker` and `context-gateway serve` with
     `CTX_LLM_BACKEND=echo CTX_JOB_BACKEND=procrastinate CTX_SUMMARY_TRIGGER_TOKENS=60
     CTX_SUMMARY_KEEP_RECENT_TOKENS=20` (small values to force summaries).
   - Two stateless `curl` calls show `history_messages` growing in the echo reply.
   - A streaming `curl -N` call shows SSE chunks and `[DONE]`.
   - After a few more calls, `GET /memory/v1/sessions/...` shows a non-null summary written
     by the worker, and the next echo reply shows `summary=yes`.
7. `docs/DECISIONS.md` lists every deviation from this spec with a one-line reason (or says
   "None").
8. README complete per section 18. LICENSE is the full Apache-2.0 text.
9. All work committed on `main` in logical commits. Do not push (the orchestrator pushes
   after review).

Commit message format: imperative subject under 72 chars, blank line, short body if needed,
then these trailer lines exactly (coding agent):

```
Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EFpEGZNYkbb7uzzMNoES3y
```

---

## 20. Roadmap (context only, do not build in Phase 1)

| Phase | Scope |
|---|---|
| 2 | Long-term memory: async fact extraction with a cheap model, reconcile (add, update, supersede, no-op) with `valid_from` / `superseded_by`, embeddings via LiteLLM, pgvector + Postgres full-text hybrid retrieval, memory REST API (`remember`, `recall`, `forget`), Python and TypeScript SDKs. Optional `group_id` scope for multi-user chats. |
| 3 | Context inspector UI (what was injected, why, token cost), LongMemEval subset eval suite, per-app API keys, retention/TTL and forget-everything for a user. |
| 4 | MCP server exposing remember/recall/forget tools, time-aware queries, adapters for Vercel AI SDK middleware and LangGraph. |

Design Phase 1 so these slot in without rewrites: keep `core/` provider-agnostic, keep
storage behind `SessionStore`, keep jobs behind `JobQueue`, and keep the memory block
assembly in one place (`ContextBuilder`).
