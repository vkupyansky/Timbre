#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
uv sync --project ml --frozen
ml/.venv/bin/python -c 'from pathlib import Path; Path("ml/.venv/.gitignore").unlink(missing_ok=True)'
uv run --project ml --frozen ruff format --check ml
uv run --project ml --frozen ruff check ml
uv run --project ml --frozen mypy --config-file ml/pyproject.toml \
  ml/check.py ml/tts/request.py ml/tts/download.py ml/tts/run.py ml/tts/runtime.py \
  ml/tts/fish_runtime.py ml/tts/fish_codec.py ml/tts/style_context.py \
  ml/profile/run.py ml/profile/download.py ml/asr/run.py
uv run --project ml --frozen pytest -c ml/pyproject.toml \
  ml/tests ml/tts/tests ml/profile/tests ml/asr/tests -W error
uv pip check --python ml/.venv/bin/python
uv run --project ml --frozen pip-audit --cache-dir .shift/cache/audit-ml
