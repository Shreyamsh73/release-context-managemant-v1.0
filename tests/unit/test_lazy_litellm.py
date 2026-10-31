"""LiteLLM must stay out of unit and API paths (spec section 3, rule 3)."""

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import asyncio, sys
    import httpx
    import context_gateway
    from context_gateway.app import create_app
    from context_gateway.config import Settings
    import context_gateway.cli, context_gateway.worker.tasks, context_gateway.llm.litellm_client
    import context_gateway.jobs.procrastinate_queue, context_gateway.store.postgres
    import context_gateway.store.cache

    assert "litellm" not in sys.modules, "importing the package imported litellm"

    async def main():
        settings = Settings(_env_file=None, store="memory", llm_backend="echo",
                            token_counter="heuristic")
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
                r = await client.post("/v1/chat/completions", json=body)
                assert r.status_code == 200, r.text
                r = await client.post("/v1/chat/completions", json={**body, "stream": True})
                assert r.status_code == 200, r.text
    asyncio.run(main())
    assert "litellm" not in sys.modules, "serving a request imported litellm"
    import os
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    print("ok")
    """
)


def test_importing_and_serving_with_echo_never_imports_litellm():
    result = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


def test_cost_map_env_is_set_on_package_import():
    code = (
        "import os; os.environ.pop('LITELLM_LOCAL_MODEL_COST_MAP', None); "
        "import context_gateway; print(os.environ['LITELLM_LOCAL_MODEL_COST_MAP'])"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "True", result.stderr
