import httpx
import psycopg
import pytest
from procrastinate import PsycopgConnector

from context_gateway.app import create_app
from context_gateway.core.summary import Summarizer
from context_gateway.core.tokens import HeuristicTokenCounter
from context_gateway.core.types import Scope
from context_gateway.jobs.procrastinate_queue import ProcrastinateJobQueue
from context_gateway.llm.echo import EchoLLMClient
from context_gateway.store.postgres import PostgresSessionStore
from context_gateway.worker.app import app as worker_app
from context_gateway.worker.tasks import Runtime, set_runtime
from tests.fakes import FakeLLM
from tests.helpers import make_settings, user
from tests.store_contract import conversation

SCOPE = Scope("app", "alice", "default")


@pytest.fixture
async def queue(pg_url):
    """The module-level worker app, pointed at the test database."""
    with worker_app.replace_connector(PsycopgConnector(conninfo=pg_url)):
        job_queue = ProcrastinateJobQueue(worker_app)
        await job_queue.start()
        yield job_queue
        await job_queue.close()


@pytest.fixture
async def store(pg_async_url):
    postgres = PostgresSessionStore(pg_async_url)
    yield postgres
    await postgres.close()


def make_runtime(store, llm=None, **overrides):
    defaults = {
        "default_model": "echo",
        "summary_trigger_tokens": 50,
        "summary_keep_recent_tokens": 20,
    }
    settings = make_settings(**{**defaults, **overrides})
    llm = llm or EchoLLMClient()
    counter = HeuristicTokenCounter()
    return Runtime(settings, store, llm, counter, Summarizer(store, llm, counter, settings))


@pytest.fixture(autouse=True)
def _reset_runtime():
    yield
    set_runtime(None)


def jobs(pg_url, where="true"):
    with psycopg.connect(pg_url) as conn:
        return conn.execute(
            f"SELECT status, queue_name, task_name, lock, queueing_lock, args, attempts "
            f"FROM procrastinate_jobs WHERE {where} ORDER BY id"
        ).fetchall()


async def seed(store, turns=6, tokens=10):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(turns, tokens))
    return session


# spec 57
async def test_deferring_twice_for_one_session_queues_one_job(queue, store, pg_url):
    session = await store.get_or_create_session(SCOPE)

    await queue.enqueue_summarize(session.id)
    await queue.enqueue_summarize(session.id)  # AlreadyEnqueued is handled, no error

    rows = jobs(pg_url)
    assert len(rows) == 1
    status, queue_name, task_name, lock, queueing_lock, args, _ = rows[0]
    assert (status, queue_name, task_name) == ("todo", "summaries", "summarize_session")
    assert lock == queueing_lock == str(session.id)
    assert args == {"session_id": str(session.id)}


async def test_different_sessions_each_get_a_job(queue, store, pg_url):
    one = await store.get_or_create_session(SCOPE)
    two = await store.get_or_create_session(Scope("app", "bob", "default"))
    await queue.enqueue_summarize(one.id)
    await queue.enqueue_summarize(two.id)
    assert len(jobs(pg_url)) == 2


async def test_worker_runs_the_summary_against_the_real_store(queue, store, pg_url):
    session = await seed(store, turns=6, tokens=10)  # 120 tokens, trigger 50
    set_runtime(make_runtime(store))

    await queue.enqueue_summarize(session.id)
    await worker_app.run_worker_async(queues=["summaries"], wait=False)

    refreshed = await store.get_session(session.id)
    assert refreshed.summary is not None
    assert refreshed.summary.startswith("- summary of ")
    assert refreshed.summarized_through_turn >= 1
    assert [r[0] for r in jobs(pg_url)] == ["succeeded"]
    window = await store.load_window(session.id, max_turns=None)
    assert window.unsummarized_tokens <= 50 + 20


async def test_a_second_worker_run_without_new_jobs_does_nothing(queue, store, pg_url):
    session = await seed(store)
    set_runtime(make_runtime(store))
    await queue.enqueue_summarize(session.id)
    await worker_app.run_worker_async(queues=["summaries"], wait=False)
    before = await store.get_session(session.id)
    await worker_app.run_worker_async(queues=["summaries"], wait=False)
    assert await store.get_session(session.id) == before


async def test_partial_summary_defers_itself_until_caught_up(queue, store, pg_url):
    session = await seed(store, turns=10, tokens=10)  # 200 tokens
    set_runtime(make_runtime(store, summary_max_fold_tokens=45))

    await queue.enqueue_summarize(session.id)
    await worker_app.run_worker_async(queues=["summaries"], wait=False)

    statuses = [r[0] for r in jobs(pg_url)]
    assert len(statuses) >= 2, "a PARTIAL run should have deferred follow-up jobs"
    assert set(statuses) == {"succeeded"}
    refreshed = await store.get_session(session.id)
    window = await store.load_window(session.id, max_turns=None)
    assert refreshed.summarized_through_turn >= 4
    assert window.unsummarized_tokens <= 50


async def test_failing_summary_is_retried_then_marked_failed(queue, store, pg_url):
    session = await seed(store)
    broken = FakeLLM(summary_text="")  # empty output raises SummaryError
    set_runtime(make_runtime(store, llm=broken))

    await queue.enqueue_summarize(session.id)
    await worker_app.run_worker_async(queues=["summaries"], wait=False)

    rows = jobs(pg_url)
    assert [r[0] for r in rows] == ["failed"]
    attempts = rows[0][6]
    assert attempts > 1, "the job should have been retried"
    assert len(broken.summary_calls) == attempts
    assert (await store.get_session(session.id)).summary is None


async def test_gateway_defers_and_worker_summarizes_end_to_end(pg_url, pg_async_url):
    """The gateway with the procrastinate backend enqueues; the worker writes the summary."""
    settings = make_settings(
        store="postgres",
        database_url=pg_url,
        job_backend="procrastinate",
        default_model="echo",
        summary_trigger_tokens=60,
        summary_keep_recent_tokens=20,
    )
    with worker_app.replace_connector(PsycopgConnector(conninfo=pg_url)):
        app = create_app(settings, llm=EchoLLMClient(), counter=HeuristicTokenCounter())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                replies = []
                for i in range(1, 7):
                    response = await client.post(
                        "/v1/chat/completions",
                        json={"model": "m", "messages": [user(f"message number {i} padding")]},
                    )
                    assert response.status_code == 200
                    replies.append(response.json()["choices"][0]["message"]["content"])
                assert any(r[0] == "todo" for r in jobs(pg_url)), "gateway should have queued a job"
                assert replies[1].endswith("summary=no")

                # The worker runs in its own process in production. Here it runs in-process.
                runtime_store = PostgresSessionStore(pg_async_url)
                set_runtime(make_runtime(runtime_store, summary_trigger_tokens=60))
                try:
                    await worker_app.run_worker_async(queues=["summaries"], wait=False)
                finally:
                    await runtime_store.close()

                info = await client.get("/memory/v1/sessions/default/anonymous/default")
                assert info.json()["summary"].startswith("- summary of ")

                final = await client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [user("do you remember?")]},
                )
                assert final.headers["X-Memory-Summary"] == "used"
                assert final.json()["choices"][0]["message"]["content"].endswith("summary=yes")
