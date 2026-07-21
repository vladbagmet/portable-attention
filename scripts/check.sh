#!/usr/bin/env bash
# Canonical local quality gate. Run this before every push; CI runs the same
# checks. Any non-zero step fails the whole gate (set -e + pipefail).
set -euo pipefail

uv run ruff check . \
  && uv run ruff format --check . \
  && uv run pyright \
  && uvx --from bandit bandit -c pyproject.toml -r src -x tests \
  && uv run pip-audit \
  && uv run coverage run -m pytest \
  && uv run coverage report
