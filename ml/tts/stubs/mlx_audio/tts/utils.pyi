"""Narrow checked API for mlx-audio 0.5.3, which does not ship py.typed."""

from pathlib import Path

def load_model(model_path: Path, *, strict: bool = ...) -> object: ...
