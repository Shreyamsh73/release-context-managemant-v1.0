import subprocess
import sys

import pytest

from context_gateway import cli
from context_gateway.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_parser_knows_the_three_commands():
    parser = cli.build_parser()
    assert parser.parse_args(["serve"]).command == "serve"
    assert parser.parse_args(["migrate"]).command == "migrate"
    assert parser.parse_args(["worker"]).command == "worker"
    args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "9000", "--reload"])
    assert (args.host, args.port, args.reload) == ("127.0.0.1", 9000, True)


def test_missing_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_serve_starts_uvicorn_with_the_app_factory(monkeypatch):
    calls = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.update(args=a, kwargs=kw))
    monkeypatch.setenv("CTX_PORT", "9123")
    assert cli.main(["serve", "--host", "127.0.0.1"]) == 0
    assert calls["args"] == ("context_gateway.app:create_app",)
    assert calls["kwargs"]["factory"] is True
    assert calls["kwargs"]["host"] == "127.0.0.1"
    assert calls["kwargs"]["port"] == 9123
    assert calls["kwargs"]["reload"] is False


def test_migrate_upgrades_with_the_sync_url(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "upgrade_database", lambda url: seen.append(url))
    monkeypatch.setenv("CTX_DATABASE_URL", "postgres://u:p@db:5432/x")
    assert cli.main(["migrate"]) == 0
    assert seen == ["postgresql+psycopg://u:p@db:5432/x"]
    assert "up to date" in capsys.readouterr().out


def test_worker_runs_the_procrastinate_worker(monkeypatch):
    ran = []

    async def fake_run():
        ran.append(True)

    monkeypatch.setattr(cli, "_run_worker", fake_run)
    assert cli.main(["worker"]) == 0
    assert ran == [True]


def test_invalid_config_fails_fast_with_a_clear_message(monkeypatch, capsys):
    monkeypatch.setenv("CTX_SUMMARY_KEEP_RECENT_TOKENS", "9000")
    assert cli.main(["migrate"]) == 2
    err = capsys.readouterr().err
    assert "Invalid configuration" in err
    assert "CTX_SUMMARY_KEEP_RECENT_TOKENS must be less than CTX_SUMMARY_TRIGGER_TOKENS" in err


def test_invalid_field_value_names_the_variable(monkeypatch, capsys):
    monkeypatch.setenv("CTX_STORE", "sqlite")
    assert cli.main(["serve"]) == 2
    assert "CTX_STORE" in capsys.readouterr().err


def test_console_script_runs():
    result = subprocess.run(
        [sys.executable, "-m", "context_gateway.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "migrate" in result.stdout
    assert "worker" in result.stdout
