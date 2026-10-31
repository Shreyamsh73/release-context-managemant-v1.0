# Decisions and deviations

Each entry records a place where the code differs from `CLAUDE.md`, or where the spec was
ambiguous and the simplest option was chosen. One line of reason per entry.

1. `pyproject.toml` sets `[tool.ruff.format] exclude = ["*.md"]`. Reason: Ruff 0.17 formats
   Python code blocks inside Markdown and rewrote `CLAUDE.md`, and `ruff format --check .` in
   CI would fail on it.
