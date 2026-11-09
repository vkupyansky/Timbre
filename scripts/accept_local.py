"""Real hybrid acceptance. Only Telegram transport is replaced; no external account."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import plistlib
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp

from scripts.service import PROJECT, launch_config
from shift.audio import AudioError, to_voice, validate_enrollment_reference, validate_voice
from shift.config import atomic_private, initialize, read_worker_token


def run(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


async def exercise(state: Path) -> dict[str, object]:
    async with aiohttp.ClientSession(
        base_url="http://127.0.0.1:8765",
        headers={"Authorization": "Bearer " + read_worker_token(state)},
        timeout=aiohttp.ClientTimeout(total=10),
    ) as session:
        for _ in range(90):
            try:
                async with session.get("/health") as response:
                    response.raise_for_status()
                    if (await response.json())["worker_ready"] is True:
                        break
            except (aiohttp.ClientError, TimeoutError):
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Native launchd worker did not become ready")
        timings: dict[str, float] = {}
        body: dict[str, object] = {}
        for count, kind in enumerate(("profile", "tts", "vc"), 1):
            start = time.monotonic()
            endpoint = "/test/add_voice" if kind == "profile" else f"/test/input/{kind}"
            async with session.post(endpoint) as response:
                response.raise_for_status()
            for _ in range(450):
                async with session.get("/test/status") as response:
                    response.raise_for_status()
                    body = await response.json()
                jobs = body.get("jobs")
                deliveries = body.get("deliveries")
                if not isinstance(jobs, list) or not isinstance(deliveries, list):
                    raise RuntimeError("Malformed acceptance result")
                if any(job["status"] in ("failed", "uncertain") for job in jobs):
                    raise RuntimeError("Acceptance job failed")
                if len(deliveries) >= count:
                    timings[kind] = round(time.monotonic() - start, 3)
                    print(f"{kind}: готовое голосовое получено за {timings[kind]} с.", flush=True)
                    break
                await asyncio.sleep(2)
            else:
                raise RuntimeError("Acceptance job exceeded 15 minutes")
        return {"timings_seconds": timings, "state": body}


async def validate_inputs(reference: Path, source: Path) -> None:
    await validate_enrollment_reference(reference)
    await validate_voice(source)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Изолированная локальная проверка с вашими записями."
    )
    parser.add_argument("--reference", required=True, type=Path, help="Чистая речь 15–20 секунд")
    parser.add_argument(
        "--source", required=True, type=Path, help="Речь для распознавания и озвучки"
    )
    args = parser.parse_args(argv)
    reference: Path = args.reference
    source: Path = args.source
    try:
        asyncio.run(validate_inputs(reference, source))
    except (AudioError, OSError) as error:
        reason = str(error) if isinstance(error, AudioError) else "Не удалось прочитать аудио."
        parser.error(reason)
    os.umask(0o077)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8765))  # Fail before touching anything if shift is running.
    output = PROJECT / "artifacts/hybrid"
    output.mkdir(parents=True, exist_ok=True)
    identity = secrets.token_hex(4)
    container = "shift-accept-" + identity
    label = "ru.shift.acceptance." + identity
    domain = f"gui/{os.getuid()}"
    started = False
    booted = False
    with tempfile.TemporaryDirectory(prefix="shift-accept-") as directory:
        state = Path(directory).resolve()
        initialize(state)
        atomic_private(
            state / ".env",
            (
                f"TELEGRAM_BOT_TOKEN=123456789:{secrets.token_urlsafe(32)}\n"
                "TELEGRAM_ADMIN_USER_IDS=777\n"
            ).encode(),
        )
        (state / "models").symlink_to(PROJECT / ".shift/models", target_is_directory=True)
        asyncio.run(to_voice(reference, state / "reference.ogg"))
        asyncio.run(to_voice(source, state / "source.ogg"))
        try:
            run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-d",
                    "--name",
                    container,
                    "--init",
                    "--read-only",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges:true",
                    "--memory",
                    "768m",
                    "--pids-limit",
                    "96",
                    "--tmpfs",
                    "/tmp:size=64m,mode=1777",
                    "-p",
                    "127.0.0.1:8765:8765",
                    "-v",
                    f"{state}:/acceptance",
                    "-v",
                    f"{PROJECT / 'scripts/container_acceptance.py'}:/acceptance-server.py:ro",
                    "--entrypoint",
                    "python",
                    "shift:0.1.0",
                    "-W",
                    "error",
                    "/acceptance-server.py",
                ]
            )
            started = True
            dns = run(
                [
                    "docker",
                    "exec",
                    container,
                    "python",
                    "-c",
                    "import socket; print(socket.gethostbyname('host.docker.internal'))",
                ]
            )
            plist = state / "worker.plist"
            atomic_private(plist, plistlib.dumps(launch_config(state, label)))
            run(["launchctl", "bootstrap", domain, str(plist)])
            booted = True
            report = asyncio.run(exercise(state))
            report.update(
                launchd=True,
                container_loopback=True,
                host_docker_internal=dns,
                telegram="isolated fake transport; no polling or external account",
                models="real local Whisper ASR and MLX Fish TTS for both input modes",
            )
            for result in (state / "results").glob("*.ogg"):
                shutil.copyfile(result, output / result.name)
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print("Гибридная локальная приёмка пройдена; отчёт artifacts/hybrid/report.json.")
        finally:
            if booted:
                run(["launchctl", "bootout", f"{domain}/{label}"])
            if started:
                logs = run(["docker", "logs", container])
                (output / "container.log").write_text(logs)
                run(["docker", "stop", "-t", "20", container])
            log = state / "logs/worker.log"
            if log.exists():
                shutil.copyfile(log, output / "worker.log")


if __name__ == "__main__":
    main()
