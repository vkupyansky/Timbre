"""Developer checks; no production credentials or heavy model generation."""

import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = [
        ["uv", "run", "--frozen", "ruff", "format", "--no-cache", "--check", "."],
        ["uv", "run", "--frozen", "ruff", "check", "--no-cache", "."],
        ["uv", "run", "--frozen", "mypy"],
        ["uv", "run", "--frozen", "pytest", "-W", "error"],
        ["uv", "pip", "check"],
        [str(PROJECT / "ml/check.sh")],
        ["uv", "build", "--no-sources"],
        [str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts/verify_dist.py")],
        [
            "uv",
            "run",
            "--frozen",
            "pip-audit",
            "--local",
            "--skip-editable",
            "--cache-dir",
            ".shift/cache/audit-app",
        ],
        ["docker", "compose", "config", "--quiet"],
        ["docker", "compose", "build", "--quiet"],
        [str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts/smoke_container.py")],
    ]
    for command in commands:
        subprocess.run(
            command,
            cwd=PROJECT,
            check=True,
            env={key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
            | {"UV_PYTHON_DOWNLOADS": "never"},
        )
    print("Проверки пройдены; реальные ML/Telegram сценарии отмечены отдельно в docs/STATUS.md.")


if __name__ == "__main__":
    main()
