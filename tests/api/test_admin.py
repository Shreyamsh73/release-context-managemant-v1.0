import pytest

from context_gateway.core.types import Scope
from tests.api.conftest import running_app
from tests.fakes import FakeLLM
from tests.helpers import assistant, system, user

BASE = "/memory/v1/sessions/default/anonymous/default"


async def seed(h, turns=3):
    for i in range(1, turns + 1):
        await h.chat(user(f"question {i}"))


# spec 47
async def test_auth_missing_key_is_401_and_health_stays_open():
    async with running_app(api_keys=["secret-key", "other-key"]) as h:
        missing = await h.chat(user("hi"))
        assert missing.status_code == 401
        assert missing.json() == {
            "error": {
                "message": "Missing or invalid API key",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
        wrong = await h.chat(user("hi"), headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        wrong_header = await h.chat(user("hi"), headers={"X-Api-Key": "nope"})
        assert wrong_header.status_code == 401
        no_scheme = await h.chat(user("hi"), headers={"Authorization": "secret-key"})
        assert no_scheme.status_code == 401
        assert h.llm.calls == []

        assert (await h.client.get("/healthz")).status_code == 200
        assert (await h.client.get("/readyz")).status_code == 200


async def test_auth_bearer_and_x_api_key_work_on_every_protected_route():
    async with running_app(api_keys=["secret-key", "other-key"]) as h:
        bearer = await h.chat(user("hi"), headers={"Authorization": "Bearer secret-key"})
        assert bearer.status_code == 200
        lower = await h.chat(user("hi"), headers={"Authorization": "bearer other-key"})
        assert lower.status_code == 200
        header = await h.chat(user("hi"), headers={"X-Api-Key": "other-key"})
        assert header.status_code == 200

        for method, path in (
            ("GET", "/v1/models"),
            ("GET", "/scope/a/b/c/v1/models"),
            ("GET", BASE),
            ("GET", BASE + "/messages"),
            ("DELETE", BASE),
            ("POST", BASE + "/summarize"),
            ("POST", "/memory/v1/context/preview"),
            ("POST", "/scope/a/b/c/v1/chat/completions"),
        ):
            denied = await h.client.request(method, path)
            assert denied.status_code == 401, (method, path)
            allowed = await h.client.request(
                method, path, headers={"X-Api-Key": "secret-key"}, json={"messages": []}
            )
            assert allowed.status_code != 401, (method, path)


async def test_auth_disabled_when_no_keys_configured():
    async with running_app() as h:
        assert (await h.client.get("/v1/models")).status_code == 200


# spec 48
async def test_models_lists_aliases_and_default():
    aliases = {"claude": "bedrock/converse/a", "fast": "bedrock/converse/b"}
    async with running_app(model_aliases=aliases, default_model="claude") as h:
        for path in ("/v1/models", "/scope/ide/alice/auto/v1/models"):
            response = await h.client.get(path)
            assert response.status_code == 200
            body = response.json()
            assert body["object"] == "list"
            assert [m["id"] for m in body["data"]] == ["claude", "fast", "default"]
            assert body["data"][0] == {
                "id": "claude",
                "object": "model",
                "created": 0,
                "owned_by": "context-gateway",
            }


async def test_models_without_default_has_no_default_entry():
    async with running_app(model_aliases={"claude": "x/y"}) as h:
        ids = [m["id"] for m in (await h.client.get("/v1/models")).json()["data"]]
        assert ids == ["claude"]


# spec 49
async def test_admin_get_session_info():
    async with running_app() as h:
        await seed(h, turns=3)
        response = await h.client.get(BASE)
        assert response.status_code == 200
        info = response.json()
        assert info["app"] == "default"
        assert info["user"] == "anonymous"
        assert info["session"] == "default"
        assert info["id"]
        assert info["summary"] is None
        assert info["summarized_through_turn"] == 0
        assert info["turn_count"] == 3
        assert info["message_count"] == 6
        assert info["unsummarized_tokens"] > 0
        assert info["created_at"]
        assert info["updated_at"]
        assert set(info) == {
            "app", "user", "session", "id", "summary", "summarized_through_turn",
            "turn_count", "message_count", "unsummarized_tokens", "created_at", "updated_at",
        }  # fmt: skip


async def test_admin_list_messages_most_recent_ascending_with_limit():
    async with running_app() as h:
        await seed(h, turns=3)
        response = await h.client.get(BASE + "/messages", params={"limit": 4})
        data = response.json()["data"]
        assert [m["seq"] for m in data] == [3, 4, 5, 6]
        assert data[0]["role"] == "user"
        assert data[0]["message"] == user("question 2")
        assert {"seq", "turn_index", "role", "message", "token_count", "model"} <= set(data[0])
        default = await h.client.get(BASE + "/messages")
        assert len(default.json()["data"]) == 6


@pytest.mark.parametrize("limit", [0, 501, -1, "abc"])
async def test_admin_messages_limit_is_validated(limit):
    async with running_app() as h:
        await seed(h, turns=1)
        response = await h.client.get(BASE + "/messages", params={"limit": limit})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


async def test_admin_preview_has_no_side_effects_and_matches_a_real_request():
    async with running_app() as h:
        h.llm.queue_text("hello Shreyamsh")
        await h.chat(user("my name is Shreyamsh"))
        calls_before = len(h.llm.calls)
        session = await h.store.find_session(Scope("default", "anonymous", "default"))
        before = await h.store.list_messages(session.id, limit=100)

        body = {
            "model": "m",
            "messages": [system("be brief"), user("what is my name")],
        }
        response = await h.client.post("/memory/v1/context/preview", json=body)

        assert response.status_code == 200
        preview = response.json()
        assert preview["mode"] == "stateless"
        assert preview["session"] == "default"
        assert preview["model"] == "m"
        assert preview["messages"] == [
            system("be brief"),
            user("my name is Shreyamsh"),
            assistant("hello Shreyamsh"),
            user("what is my name"),
        ]
        assert preview["report"]["history_turns"] == 1
        assert preview["report"]["mode"] == "stateless"
        assert len(h.llm.calls) == calls_before
        assert await h.store.list_messages(session.id, limit=100) == before

        real = await h.chat(system("be brief"), user("what is my name"))
        assert h.llm.calls[-1].messages == preview["messages"]
        assert int(real.headers["X-Memory-Prompt-Tokens"]) == preview["report"]["prompt_tokens"]


async def test_admin_preview_for_unknown_session_does_not_create_it():
    async with running_app() as h:
        body = {
            "messages": [user("hi")],
            "scope": {"app": "ghost", "user": "nobody", "session": "none"},
            "memory_mode": "stateless",
        }
        response = await h.client.post("/memory/v1/context/preview", json=body)
        assert response.status_code == 200
        assert response.json()["messages"] == [user("hi")]
        assert response.json()["session"] == "none"
        assert await h.store.find_session(Scope("ghost", "nobody", "none")) is None
        assert h.llm.calls == []


async def test_admin_preview_header_scope_beats_body_scope_and_mode_off():
    async with running_app() as h:
        body = {"messages": [user("hi")], "scope": {"session": "from-body"}}
        response = await h.client.post(
            "/memory/v1/context/preview", json=body, headers={"X-Memory-Session": "from-header"}
        )
        assert response.json()["session"] == "from-header"

        off = await h.client.post(
            "/memory/v1/context/preview",
            json={"messages": [user("hi")], "memory_mode": "off"},
        )
        assert off.json()["mode"] == "off"
        assert off.json()["session"] is None
        assert off.json()["messages"] == [user("hi")]


async def test_admin_preview_validates_input():
    async with running_app() as h:
        for body in (
            {"messages": []},
            {"messages": [user("hi")], "scope": "text"},
            {"messages": [user("hi")], "memory_mode": 5},
            {"messages": [user("hi")], "memory_mode": "bogus"},
            {"messages": [user("hi")], "scope": {"user": "bad user"}},
        ):
            response = await h.client.post("/memory/v1/context/preview", json=body)
            assert response.status_code == 400, body


async def test_admin_summarize_returns_202():
    async with running_app() as h:
        await seed(h, turns=1)
        response = await h.client.post(BASE + "/summarize")
        assert response.status_code == 202
        assert response.json() == {"enqueued": True}


async def test_admin_delete_returns_204_then_404():
    async with running_app() as h:
        await seed(h, turns=2)
        deleted = await h.client.delete(BASE)
        assert deleted.status_code == 204
        assert deleted.content == b""
        again = await h.client.delete(BASE)
        assert again.status_code == 404
        assert again.json()["error"]["code"] == "session_not_found"
        assert again.json()["error"]["type"] == "not_found_error"
        assert (await h.client.get(BASE)).status_code == 404
        # A new request starts a fresh conversation.
        await h.chat(user("fresh"))
        assert h.llm.calls[-1].messages == [user("fresh")]


@pytest.mark.parametrize(
    ("method", "suffix"),
    [("GET", ""), ("GET", "/messages"), ("POST", "/summarize"), ("DELETE", "")],
)
async def test_admin_unknown_session_is_404(method, suffix):
    async with running_app() as h:
        response = await h.client.request(method, "/memory/v1/sessions/a/b/c" + suffix)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "session_not_found"


async def test_admin_invalid_scope_in_path_is_400():
    async with running_app() as h:
        response = await h.client.get("/memory/v1/sessions/app/bad%20user/default")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_scope"


async def test_admin_does_not_resolve_auto():
    async with running_app() as h:
        await seed(h, turns=1)
        assert (await h.client.get("/memory/v1/sessions/default/anonymous/auto")).status_code == 404


async def test_readyz_reports_store_failure_with_503():
    class DownStore:
        async def ping(self):
            raise ConnectionError("db is down")

        async def close(self):
            pass

    async with running_app(store=DownStore()) as h:
        response = await h.client.get("/readyz")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["checks"]["store"].startswith("error")
        assert body["checks"]["redis"] == "disabled"
        assert "db is down" not in response.text


async def test_readyz_ok_and_healthz():
    async with running_app() as h:
        ok = await h.client.get("/readyz")
        assert ok.status_code == 200
        assert ok.json() == {"status": "ok", "checks": {"store": "ok", "redis": "disabled"}}
        assert (await h.client.get("/healthz")).json() == {"status": "ok"}


async def test_cors_exposes_memory_headers_when_configured():
    async with running_app(cors_origins=["http://app.example"]) as h:
        response = await h.client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://app.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-memory-user",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://app.example"

        actual = await h.chat(user("hi"), headers={"Origin": "http://app.example"})
        exposed = actual.headers["access-control-expose-headers"]
        assert "X-Memory-Mode" in exposed
        assert "X-Request-Id" in exposed


async def test_no_cors_headers_when_not_configured():
    async with running_app() as h:
        response = await h.chat(user("hi"), headers={"Origin": "http://app.example"})
        assert "access-control-allow-origin" not in response.headers


async def test_fake_llm_is_not_called_by_admin_routes():
    llm = FakeLLM()
    async with running_app(llm=llm) as h:
        await seed(h, turns=1)
        calls = len(llm.calls)
        await h.client.get(BASE)
        await h.client.get(BASE + "/messages")
        await h.client.post("/memory/v1/context/preview", json={"messages": [user("x")]})
        assert len(llm.calls) == calls
