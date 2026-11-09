"""Reproducible isolated container smoke; no Telegram polling, profiles, or ML runs."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from shift.config import atomic_private, initialize, read_worker_token
from shift.protocol import ClaimResponse

PROJECT = Path(__file__).resolve().parents[1]
IMAGE = "shift:0.1.0"
REPORT = PROJECT / "artifacts/container-smoke.json"


class SmokeFailure(RuntimeError):
    pass


class PortBinding(BaseModel):
    model_config = ConfigDict(strict=True)
    host_ip: str = Field(alias="HostIp")
    host_port: str = Field(alias="HostPort")


class HostConfiguration(BaseModel):
    model_config = ConfigDict(strict=True)
    read_only: bool = Field(alias="ReadonlyRootfs")
    cap_drop: list[str] = Field(alias="CapDrop")
    security_options: list[str] = Field(alias="SecurityOpt")


class ContainerConfiguration(BaseModel):
    model_config = ConfigDict(strict=True)
    user: str = Field(alias="User")


class NetworkConfiguration(BaseModel):
    model_config = ConfigDict(strict=True)
    ports: dict[str, list[PortBinding] | None] = Field(alias="Ports")


class Inspection(BaseModel):
    model_config = ConfigDict(strict=True)
    host: HostConfiguration = Field(alias="HostConfig")
    config: ContainerConfiguration = Field(alias="Config")
    network: NetworkConfiguration = Field(alias="NetworkSettings")
    image: str = Field(alias="Image")


def run(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def inspect_container(name: str) -> tuple[int, dict[str, object]]:
    values = TypeAdapter(list[Inspection]).validate_json(run(["docker", "inspect", name]))
    if len(values) != 1:
        raise SmokeFailure("unexpected_container_count")
    item = values[0]
    bindings = item.network.ports.get("8765/tcp")
    published = [port for port, entries in item.network.ports.items() if entries]
    if published != ["8765/tcp"] or bindings is None or len(bindings) != 1:
        raise SmokeFailure("unexpected_port_bindings")
    binding = bindings[0]
    if binding.host_ip != "127.0.0.1" or not binding.host_port.isdecimal():
        raise SmokeFailure("port_not_loopback")
    port = int(binding.host_port)
    if port == 8765 or not 1024 <= port <= 65535:
        raise SmokeFailure("port_not_isolated")
    expected_user = f"{os.getuid()}:{os.getgid()}"
    if os.getuid() == 0 or item.config.user != expected_user:
        raise SmokeFailure("container_user_not_unprivileged")
    if not item.host.read_only or "ALL" not in item.host.cap_drop:
        raise SmokeFailure("container_filesystem_or_capabilities")
    if not any(
        value in {"no-new-privileges", "no-new-privileges:true"}
        for value in item.host.security_options
    ):
        raise SmokeFailure("container_privilege_escalation_allowed")
    return port, {
        "image": item.image,
        "user": item.config.user,
        "read_only_rootfs": item.host.read_only,
        "cap_drop": item.host.cap_drop,
        "no_new_privileges": True,
        "published_host": binding.host_ip,
        "published_port": port,
    }


async def exercise(port: int, token: str) -> dict[str, object]:
    address = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession(
        base_url=address,
        timeout=aiohttp.ClientTimeout(total=3),
    ) as session:
        initial: object = None
        async with asyncio.timeout(45):
            while True:
                try:
                    async with session.get("/health") as response:
                        if response.status == 200:
                            initial = await response.json()
                            break
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(0.5)
        if initial != {"status": "ok", "profile_ready": False, "worker_ready": False}:
            raise SmokeFailure("unexpected_initial_readiness")
        async with session.post("/worker/claim") as response:
            if response.status != 401:
                raise SmokeFailure("missing_authorization_not_rejected")
        async with session.post(
            "/worker/claim",
            headers={"Authorization": "Bearer invalid"},
        ) as response:
            if response.status != 401:
                raise SmokeFailure("invalid_authorization_not_rejected")
        async with session.post(
            "/worker/claim",
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            if response.status != 200:
                raise SmokeFailure("internal_authorization_not_accepted")
            result = ClaimResponse.model_validate_json(await response.read())
            if result.job is not None:
                raise SmokeFailure("isolated_queue_not_empty")
        return {
            "initial_health": initial,
            "missing_authorization": 401,
            "invalid_authorization": 401,
            "internal_authorization": 200,
            "claim_job": None,
        }


def remove_container(name: str) -> bool:
    outcome = subprocess.run(
        ["docker", "rm", "--force", name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if outcome.returncode == 0:
        return True
    remaining = subprocess.run(
        ["docker", "ps", "--all", "--filter", f"name=^/{name}$", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return remaining.returncode == 0 and not remaining.stdout.strip()


def main() -> int:
    os.umask(0o077)
    name = "shift-smoke-" + secrets.token_hex(6)
    report: dict[str, object] = {
        "passed": False,
        "checked_at": datetime.now(UTC).isoformat(),
        "telegram": "isolated in-process transport; no external account or polling",
        "models_executed": False,
        "production_data_used": False,
    }
    started = time.monotonic()
    stage = "prepare"
    attempted_container = False
    try:
        if os.getuid() == 0:
            raise SmokeFailure("run_as_regular_user")
        with tempfile.TemporaryDirectory(prefix="shift-smoke-") as directory:
            state = Path(directory).resolve()
            initialize(state)
            atomic_private(
                state / ".env",
                (
                    f"TELEGRAM_BOT_TOKEN=123456789:{secrets.token_urlsafe(32)}\n"
                    "TELEGRAM_ADMIN_USER_IDS=777\n"
                ).encode(),
            )
            token = read_worker_token(state)
            try:
                stage = "container_start"
                attempted_container = True
                run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--detach",
                        "--name",
                        name,
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
                        "--cpus",
                        "2",
                        "--tmpfs",
                        "/tmp:size=64m,mode=1777",
                        "--publish",
                        "127.0.0.1::8765",
                        "--mount",
                        f"type=bind,source={state},target=/acceptance",
                        "--mount",
                        "type=bind,source="
                        f"{PROJECT / 'scripts/container_acceptance.py'}"
                        ",target=/smoke-server.py,readonly",
                        "--entrypoint",
                        "python",
                        IMAGE,
                        "-W",
                        "error",
                        "/smoke-server.py",
                    ]
                )
                stage = "container_security"
                port, details = inspect_container(name)
                report["container"] = details
                stage = "http_readiness_and_authorization"
                report["http"] = asyncio.run(exercise(port, token))
                stage = "docker_desktop_dns"
                dns = run(
                    [
                        "docker",
                        "exec",
                        name,
                        "python",
                        "-c",
                        "import socket; print(socket.gethostbyname('host.docker.internal'))",
                    ]
                )
                # Store an address only, never arbitrary command output.
                report["host_docker_internal"] = str(ipaddress.ip_address(dns))
                report["passed"] = True
            finally:
                if attempted_container:
                    report["container_removed"] = remove_container(name)
                    if not report["container_removed"]:
                        report["passed"] = False
                        report["cleanup_error"] = "container_removal_failed"
    except Exception as error:
        # CLI boundary: raw transport exceptions may contain credentials in request details.
        report.update(passed=False, failed_stage=stage, error_type=type(error).__name__)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["passed"] is not True:
        print(
            "Проверка контейнера не пройдена; отчёт artifacts/container-smoke.json.",
            file=sys.stderr,
        )
        return 1
    print("Проверка контейнера пройдена; отчёт artifacts/container-smoke.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
