#!/usr/bin/env bash
# Start, stop and inspect a local Postgres and Redis for development and tests.
# No Docker needed. Data lives in .devdata/ (gitignored).
#
#   scripts/dev-services.sh start|stop|status|env
#
# Optional overrides: PG_BIN, CG_PG_PORT, CG_REDIS_PORT, CG_DEVDATA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CG_DEVDATA:-$ROOT/.devdata}"
PG_PORT="${CG_PG_PORT:-55432}"
REDIS_PORT="${CG_REDIS_PORT:-56379}"
PG_HOME="$DATA/pg"
PG_DATA="$PG_HOME/data"
PG_LOG="$PG_HOME/server.log"
REDIS_DIR="$DATA/redis"
REDIS_PID="$DATA/redis.pid"
REDIS_LOG="$DATA/redis.log"

TEST_DATABASE_URL="postgresql://cg:cg@127.0.0.1:${PG_PORT}/cg_test"
DEV_DATABASE_URL="postgresql://cg:cg@127.0.0.1:${PG_PORT}/cg"
TEST_REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/0"

find_pg_bin() {
  if [ -n "${PG_BIN:-}" ]; then echo "$PG_BIN"; return; fi
  local candidate
  if candidate="$(command -v pg_ctl 2>/dev/null)"; then dirname "$candidate"; return; fi
  for candidate in \
    $(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V -r) \
    $(ls -d /opt/homebrew/opt/postgresql@*/bin 2>/dev/null | sort -V -r) \
    $(ls -d /usr/local/opt/postgresql@*/bin 2>/dev/null | sort -V -r); do
    if [ -x "$candidate/pg_ctl" ]; then echo "$candidate"; return; fi
  done
  echo "Postgres binaries not found. Install Postgres or set PG_BIN." >&2
  exit 1
}

PGBIN="$(find_pg_bin)"

# Postgres refuses to run as root. Switch to the postgres user in that case.
if [ "$(id -u)" -eq 0 ]; then
  as_pg() { runuser -u postgres -- "$@"; }
  prepare_dir() { install -d -o postgres -g postgres -m 700 "$1"; }
else
  as_pg() { "$@"; }
  prepare_dir() { install -d -m 700 "$1"; }
fi

pg_running() { as_pg "$PGBIN/pg_ctl" -D "$PG_DATA" status >/dev/null 2>&1; }
redis_running() { redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; }

start_postgres() {
  if pg_running; then echo "postgres: already running on port $PG_PORT"; return; fi
  mkdir -p "$DATA"
  prepare_dir "$PG_HOME"
  if [ ! -f "$PG_DATA/PG_VERSION" ]; then
    echo "postgres: initializing $PG_DATA"
    as_pg "$PGBIN/initdb" -D "$PG_DATA" -U postgres -E UTF8 --no-locale \
      --auth-local=trust --auth-host=scram-sha-256 >/dev/null
  fi
  # fsync is off because this is throwaway development data.
  as_pg "$PGBIN/pg_ctl" -D "$PG_DATA" -l "$PG_LOG" -w start \
    -o "-p $PG_PORT -c listen_addresses=127.0.0.1 -c unix_socket_directories=$PG_HOME -c fsync=off -c synchronous_commit=off -c full_page_writes=off" \
    >/dev/null
  as_pg "$PGBIN/psql" -h "$PG_HOME" -p "$PG_PORT" -U postgres -d postgres -v ON_ERROR_STOP=1 -q <<'SQL'
SELECT 'CREATE ROLE cg LOGIN PASSWORD ''cg'' CREATEDB'
  WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'cg') \gexec
SELECT 'CREATE DATABASE cg OWNER cg'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'cg') \gexec
SELECT 'CREATE DATABASE cg_test OWNER cg'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'cg_test') \gexec
SQL
  echo "postgres: running on 127.0.0.1:$PG_PORT (databases cg and cg_test, user cg, password cg)"
}

start_redis() {
  if redis_running; then echo "redis: already running on port $REDIS_PORT"; return; fi
  mkdir -p "$REDIS_DIR"
  redis-server --port "$REDIS_PORT" --bind 127.0.0.1 --dir "$REDIS_DIR" \
    --daemonize yes --pidfile "$REDIS_PID" --logfile "$REDIS_LOG" \
    --save "" --appendonly no >/dev/null
  for _ in $(seq 1 50); do
    if redis_running; then break; fi
    sleep 0.1
  done
  redis_running || { echo "redis: failed to start, see $REDIS_LOG" >&2; exit 1; }
  echo "redis: running on 127.0.0.1:$REDIS_PORT"
}

stop_postgres() {
  if pg_running; then
    as_pg "$PGBIN/pg_ctl" -D "$PG_DATA" -m fast -w stop >/dev/null
    echo "postgres: stopped"
  else
    echo "postgres: not running"
  fi
}

stop_redis() {
  if redis_running; then
    redis-cli -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
    echo "redis: stopped"
  else
    echo "redis: not running"
  fi
}

print_env() {
  cat <<EOT
export TEST_DATABASE_URL=$TEST_DATABASE_URL
export TEST_REDIS_URL=$TEST_REDIS_URL
export CTX_DATABASE_URL=$DEV_DATABASE_URL
export CTX_REDIS_URL=$TEST_REDIS_URL
EOT
}

case "${1:-}" in
  start)
    start_postgres
    start_redis
    echo
    echo "Run the tests with:"
    echo "  TEST_DATABASE_URL=$TEST_DATABASE_URL TEST_REDIS_URL=$TEST_REDIS_URL uv run pytest"
    ;;
  stop)
    stop_redis
    stop_postgres
    ;;
  status)
    if pg_running; then echo "postgres: running on port $PG_PORT"; else echo "postgres: stopped"; fi
    if redis_running; then echo "redis: running on port $REDIS_PORT"; else echo "redis: stopped"; fi
    ;;
  env)
    print_env
    ;;
  *)
    echo "usage: $0 start|stop|status|env" >&2
    exit 2
    ;;
esac
