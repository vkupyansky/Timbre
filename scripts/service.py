"""Developer provisioning of the persistent per-user service; never reads .env."""

import os
import plistlib
import subprocess
from pathlib import Path

from shift.config import atomic_private, initialize

PROJECT = Path(__file__).resolve().parents[1]
LABEL = "ru.shift.worker"


def launch_config(state: Path, label: str) -> dict[str, object]:
    return {
        "Label": label,
        "ProgramArguments": [
            str(PROJECT / ".venv/bin/python"),
            "-W",
            "error",
            "-m",
            "shift",
            "--state",
            str(state),
            "worker",
        ],
        "WorkingDirectory": str(PROJECT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,
        # Every model job is an interactive request awaiting a Telegram reply.
        # Background applies CPU/I/O throttling even while the owner is waiting.
        "ProcessType": "Interactive",
        "Umask": 0o077,
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "StandardOutPath": str(state / "logs/worker.log"),
        "StandardErrorPath": str(state / "logs/worker.log"),
    }


def install() -> None:
    state = PROJECT / ".shift"
    initialize(state)
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"} | {
        "UV_PYTHON_DOWNLOADS": "never"
    }
    subprocess.run(
        ["uv", "sync", "--project", str(PROJECT / "ml"), "--frozen"], check=True, env=environment
    )
    for kind in ("tts", "profile"):
        destination = "asr" if kind == "profile" else kind
        subprocess.run(
            [
                str(PROJECT / "ml/.venv/bin/python"),
                str(PROJECT / "ml" / kind / "download.py"),
                "--model-dir",
                str(state / "models" / destination),
            ],
            check=True,
        )
    subprocess.run(
        [
            str(PROJECT / "ml/.venv/bin/python"),
            "-W",
            "error",
            str(PROJECT / "ml/check.py"),
            "--model-dir",
            str(state / "models"),
        ],
        check=True,
    )
    path = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = plistlib.loads(path.read_bytes())
        if old.get("WorkingDirectory") != str(PROJECT):
            raise RuntimeError("An existing shift service belongs to another directory")
    # The parent is a shared system directory; do not change its permissions.
    temp = state / "launch.plist"
    atomic_private(temp, plistlib.dumps(launch_config(state, LABEL)))
    os.replace(temp, path)
    service = f"gui/{os.getuid()}/{LABEL}"
    current = subprocess.run(["launchctl", "print", service], capture_output=True)
    if current.returncode == 0:
        subprocess.run(["launchctl", "bootout", service], check=True)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    print("Пользовательская служба готова и будет ждать запуска Docker Compose.")


if __name__ == "__main__":
    install()
