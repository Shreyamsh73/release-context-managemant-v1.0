# Local setup and testing

Follow this after cloning. Pick one path.

| Path | Needs | Real model calls | Time |
|---|---|---|---|
| A. Echo test | Python tooling, Postgres | No | 5 min |
| B. Docker with Bedrock | Docker, AWS credentials | Yes | 10 min |
| C. Run the test suite | Same as A | No | 2 min |

Do A first. It proves the memory logic works before you spend anything on a model.

## Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`)
- For path A and C: Postgres 14 or newer installed locally. Redis is optional.
- For path B: Docker Desktop or Docker Engine with Compose.
- `scripts/dev-services.sh` is a bash script. On Windows use WSL, or use path B.

```bash
git clone https://github.com/Shreyamsh73/release-context-managemant-v1.0.git
cd release-context-managemant-v1.0
uv sync
```

`uv sync` installs Python 3.12 and the locked dependencies. LiteLLM is pinned on purpose. Do not upgrade it casually.

## A. Echo test (no credentials)

The `echo` backend replies with what it received, so you can see exactly what memory the gateway added.

**1. Start local Postgres and Redis**

```bash
scripts/dev-services.sh start
eval "$(scripts/dev-services.sh env)"
```

This runs private copies under `.devdata/` on ports 55432 and 56379. It does not touch any database you already have. If you would rather use your own Postgres, set `CTX_DATABASE_URL` yourself and skip this step.

**2. Create the tables**

```bash
uv run context-gateway migrate
```

Run it twice if you like. The second run does nothing.

**3. Start the worker and the gateway**

Use two terminals, both in the repo folder. In each, run the `eval` line from step 1 first. The small thresholds below force a summary after a few messages.

Terminal 1, worker:

```bash
export CTX_LLM_BACKEND=echo CTX_DEFAULT_MODEL=echo CTX_JOB_BACKEND=procrastinate \
  CTX_SUMMARY_TRIGGER_TOKENS=60 CTX_SUMMARY_KEEP_RECENT_TOKENS=20
uv run context-gateway worker
```

Terminal 2, gateway:

```bash
export CTX_LLM_BACKEND=echo CTX_DEFAULT_MODEL=echo CTX_JOB_BACKEND=procrastinate \
  CTX_SUMMARY_TRIGGER_TOKENS=60 CTX_SUMMARY_KEEP_RECENT_TOKENS=20 CTX_PORT=8080
uv run context-gateway serve
```

Check it is up: `curl localhost:8080/readyz`. It should say `"status":"ok"`.

**4. Send messages**

```bash
chat() {
  curl -s -D /dev/stderr http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" -H "X-Memory-User: me" \
    -d "{\"model\":\"claude\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" \
    2>&1 >/dev/null | grep -i "x-memory-\(history\|summary\)"
}
chat "my name is Shreyamsh"
chat "I prefer short answers"
chat "I work on Spring Boot services"
chat "my favourite number is 7"
```

What to look for:

- `x-memory-history-turns` grows: 0, 1, 2 ...
- After a few messages `x-memory-summary` changes from `none` to `used`. That means the worker wrote a summary and the gateway injected it.

See the stored session and its summary:

```bash
curl -s localhost:8080/memory/v1/sessions/default/me/default | python3 -m json.tool
```

Dry run of what would be sent next, with no model call:

```bash
curl -s localhost:8080/memory/v1/context/preview \
  -H "Content-Type: application/json" -H "X-Memory-User: me" \
  -d '{"model":"claude","messages":[{"role":"user","content":"what do you know about me"}]}'
```

**5. Try the other modes**

IDE-style client that resends its whole history:

```bash
curl -s -i localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-Memory-Mode: stateful" -H "X-Memory-User: me" \
  -d '{"model":"claude","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"second question"}]}'
```

The headers show `x-memory-mode: stateful` and a session named `fp-...`. The gateway did not add history because the client already sent it.

Streaming: add `"stream": true` to the body and use `curl -N`. You get SSE chunks and a final `data: [DONE]`.

Memory off: send `-H "X-Memory-Mode: off"`. Nothing is read or stored.

**6. Stop**

```bash
scripts/dev-services.sh stop
```

## B. Docker with AWS Bedrock

**1. Turn on a model in Bedrock**

In the AWS console, open Bedrock, then Model access, and enable a Claude model in your region. Copy its model id. Many accounts need the regional form, for example `us.anthropic.claude-sonnet-4-6`. Check the exact id in the console.

**2. Configure**

```bash
cp .env.example .env
```

Edit `.env`:

```
CTX_DEFAULT_MODEL=claude
CTX_MODEL_ALIASES={"claude":"bedrock/converse/<your-model-id>"}
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...        # only for temporary credentials
AWS_REGION_NAME=us-east-1    # the region where you enabled the model
```

Using an AWS profile instead? Set `AWS_PROFILE=<name>` and leave the key lines out. Compose mounts `~/.aws` read-only.

Optional: set `CTX_API_KEYS=some-secret` so only callers with that key can use the gateway. Then add `-H "Authorization: Bearer some-secret"` to every curl below.

**3. Start**

```bash
docker compose up --build
```

This starts Postgres, Redis, a one-time migration, the gateway on port 8080 and the summary worker. Wait for the gateway log line `started`.

**4. Test memory with a real model**

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-Memory-User: me" \
  -d '{"model":"claude","messages":[{"role":"user","content":"My name is Shreyamsh."}]}'

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-Memory-User: me" \
  -d '{"model":"claude","messages":[{"role":"user","content":"What is my name?"}]}'
```

The second answer should say Shreyamsh. This is the check that matters. It also covers the part that is not yet verified: real Bedrock requests and streaming through LiteLLM.

Test streaming too: add `"stream": true` and use `curl -N`.

**5. Test summaries with a real model**

Real summaries need real volume. For a quick test, set `CTX_SUMMARY_TRIGGER_TOKENS=300` and `CTX_SUMMARY_KEEP_RECENT_TOKENS=100` in `.env`, run `docker compose up -d --force-recreate`, then send 6 to 8 longer messages. Read the summary at `GET /memory/v1/sessions/default/me/default`. Check that it keeps your name and preferences.

**6. Stop**

```bash
docker compose down        # keep data
docker compose down -v     # also delete the database
```

## Use it from IntelliJ

Install the Continue plugin, then add this model to its `config.yaml`:

```yaml
models:
  - name: Claude via Context Gateway
    provider: openai
    model: claude
    apiBase: http://localhost:8080/scope/intellij/<your-name>/auto/v1
    apiKey: any-string          # or your CTX_API_KEYS value
    requestOptions:
      headers:
        X-Memory-Mode: stateful
```

Open a chat, ask something, then check `GET /memory/v1/sessions/intellij/<your-name>/<session>`. The session name starts with `fp-`. List sessions with `docker compose exec postgres psql -U cg -c "select session_key, updated_at from cg_sessions"`.

A new chat in the IDE gets a new session. Two chats that start with the exact same first message share one.

## C. Run the test suite

```bash
scripts/dev-services.sh start
eval "$(scripts/dev-services.sh env)"
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
scripts/dev-services.sh stop
```

Expected: 417 passed, 0 skipped. Without Postgres and Redis running, the 91 integration tests skip. To run only the fast tests, use `uv run pytest -q -m "not integration"`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `connection refused` on port 55432 | Run `scripts/dev-services.sh start`, then the `eval` line. |
| `relation "cg_sessions" does not exist` | Run `uv run context-gateway migrate`. |
| Summary never appears | The worker is not running, or `CTX_DEFAULT_MODEL` is empty, or the thresholds are too high for your message size. Check the worker log. |
| `x-memory-mode: degraded` | The database is down or unreachable. The request still went to the model. Check the gateway log. |
| 401 from the gateway | `CTX_API_KEYS` is set. Send `Authorization: Bearer <key>`. |
| 502 or `upstream_error` from Bedrock | Wrong model id, model not enabled in that region, or bad credentials. Test the same id with the AWS CLI. |
| Bedrock `AccessDeniedException` | Your IAM user needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`. |
| First request is slow | LiteLLM loads on first use, about 8 seconds. Later requests are fast. |
| Docker build fails on `uv sync --locked` | Run `uv lock` locally and commit the result. |

## Reset everything

```bash
scripts/dev-services.sh stop && rm -rf .devdata      # path A and C
docker compose down -v                                # path B
```
