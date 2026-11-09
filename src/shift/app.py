"""Loopback-published worker API and the Telegram long-polling service."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import math
import secrets
import shutil
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BotCommand
from aiohttp import BodyPartReader, web
from pydantic import ValidationError

from .audio import (
    MAX_RESULT_BYTES,
    AudioError,
    validate_enrollment_reference,
    validate_reference,
    validate_result,
)
from .bot import BotController
from .protocol import (
    MAX_PROFILE_EXAMPLE_BYTES,
    MAX_PROFILE_REFERENCE_BYTES,
    MAX_TEXT,
    ClaimResponse,
    FailedRequest,
    HeartbeatResponse,
    TranscriptRequest,
)
from .store import InsufficientBalance, NoProfile, StaleLease, Store
from .whisper_cache import WhisperContext

logger = logging.getLogger("shift.app")


class SafeExternalLogs(logging.Filter):
    """External transport exception strings can contain credential-bearing request URLs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.__dict__.get("shift_sanitized") is True:
            return True
        if record.name.split(".", 1)[0] in {"aiogram", "aiohttp", "asyncio"}:
            reason = (
                type(record.exc_info[1]).__name__
                if record.exc_info is not None and record.exc_info[1] is not None
                else "external_event"
            )
            record.msg = f"external_event reason={reason}"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            record.__dict__["shift_sanitized"] = True
        return True


def configure_safe_logging() -> None:
    for handler in logging.getLogger().handlers:
        if not any(isinstance(item, SafeExternalLogs) for item in handler.filters):
            handler.addFilter(SafeExternalLogs())


@dataclass(frozen=True)
class AppConfig:
    owner_id: int
    data_dir: Path
    bind_host: str = "127.0.0.1"
    port: int = 8765
    admin_user_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if type(self.owner_id) is not int or not 0 < self.owner_id < 2**63:
            raise ValueError("Owner must be explicitly configured")
        if any(
            type(user_id) is not int or not 0 < user_id < 2**63 for user_id in self.admin_user_ids
        ):
            raise ValueError("Administrator IDs must be positive Telegram IDs")


@dataclass
class Runtime:
    store: Store
    worker_token: str
    worker_seen: float = 0
    lock: BinaryIO | None = None


RUNTIME = web.AppKey("runtime", Runtime)
CONTROLLER = web.AppKey("controller", BotController)


@web.middleware
async def authenticate(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    if request.path != "/health":
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {request.app[RUNTIME].worker_token}"
        if not secrets.compare_digest(supplied.encode(), expected.encode()):
            raise web.HTTPUnauthorized()
    try:
        return await handler(request)
    except StaleLease:
        raise web.HTTPConflict(text="inactive lease") from None
    except NoProfile:
        raise web.HTTPConflict(text="voice no longer available") from None
    except (ValidationError, ValueError):
        raise web.HTTPBadRequest(text="invalid request") from None
    except OSError:
        logger.error("worker_request_failed reason=storage_error")
        raise web.HTTPInsufficientStorage(text="storage unavailable") from None
    except sqlite3.Error:
        logger.error("worker_request_failed reason=database_error")
        raise web.HTTPServiceUnavailable(text="database unavailable") from None


async def health(request: web.Request) -> web.Response:
    runtime = request.app[RUNTIME]
    profiles = await asyncio.to_thread(runtime.store.profiles, runtime.store.owner_id)
    await asyncio.to_thread(runtime.store.list_jobs, 0, 1)
    return web.json_response(
        {
            "status": "ok",
            "worker_ready": runtime.worker_seen > 0 and time.monotonic() - runtime.worker_seen < 45,
            "profile_ready": bool(profiles),
        }
    )


async def claim(request: web.Request) -> web.Response:
    runtime = request.app[RUNTIME]
    runtime.worker_seen = time.monotonic()
    await asyncio.to_thread(runtime.store.cancel_unauthorized_jobs)
    job = await asyncio.to_thread(runtime.store.claim)
    return web.json_response(ClaimResponse(job=job).model_dump(mode="json"))


def job_arguments(request: web.Request) -> tuple[Store, str, str]:
    return (
        request.app[RUNTIME].store,
        request.match_info["job_id"],
        request.headers.get("X-Shift-Lease", ""),
    )


async def input_audio(request: web.Request) -> web.FileResponse:
    store, job_id, lease = job_arguments(request)
    await asyncio.to_thread(store.check_lease, job_id, lease)
    path = store.job_dir(job_id) / "input.audio"
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Content-Type": "application/octet-stream"})


async def heartbeat(request: web.Request) -> web.Response:
    store, job_id, lease = job_arguments(request)
    cancel = await asyncio.to_thread(store.heartbeat, job_id, lease)
    request.app[RUNTIME].worker_seen = time.monotonic()
    return web.json_response(HeartbeatResponse(cancel=cancel).model_dump())


async def result_audio(request: web.Request) -> web.Response:
    store, job_id, lease = job_arguments(request)
    await asyncio.to_thread(store.check_lease, job_id, lease)
    if request.content_length is not None and request.content_length > MAX_RESULT_BYTES:
        raise web.HTTPRequestEntityTooLarge(
            max_size=MAX_RESULT_BYTES, actual_size=request.content_length
        )
    path = store.job_dir(job_id) / f"upload-{secrets.token_hex(8)}.ogg"
    size = 0
    try:
        with path.open("xb") as output:
            async for chunk in request.content.iter_chunked(64 * 1024):
                size += len(chunk)
                if size > MAX_RESULT_BYTES:
                    raise web.HTTPRequestEntityTooLarge(max_size=MAX_RESULT_BYTES, actual_size=size)
                await asyncio.to_thread(output.write, chunk)
        try:
            info = await validate_result(path)
        except AudioError:
            raise web.HTTPUnprocessableEntity(text="invalid audio") from None
        try:
            await asyncio.to_thread(
                store.finish, job_id, lease, path, duration_ms=math.ceil(info.duration * 1000)
            )
        except InsufficientBalance:
            return web.json_response({"status": "failed", "reason": "insufficient_balance"})
    finally:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    return web.json_response({"status": "ready"})


async def failed(request: web.Request) -> web.Response:
    store, job_id, lease = job_arguments(request)
    body = FailedRequest.model_validate_json(await request.read())
    await asyncio.to_thread(store.fail, job_id, lease, body.reason)
    return web.json_response({"status": "failed"})


async def transcript(request: web.Request) -> web.Response:
    store, job_id, lease = job_arguments(request)
    await asyncio.to_thread(store.check_lease, job_id, lease)
    # JSON may encode one Unicode character as a pair of six-byte surrogate escapes.
    limit = MAX_TEXT * 12 + 512
    payload = bytearray()
    async for chunk in request.content.iter_chunked(4096):
        payload.extend(chunk)
        if len(payload) > limit:
            raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=len(payload))
    body = TranscriptRequest.model_validate_json(payload)
    await asyncio.to_thread(store.record_transcript, job_id, lease, body.text)
    return web.json_response({"status": "saved"})


async def profile_result(request: web.Request) -> web.Response:
    store, job_id, lease = job_arguments(request)
    await asyncio.to_thread(store.check_lease, job_id, lease)
    directory = store.job_dir(job_id) / f"profile-upload-{secrets.token_hex(8)}"
    await asyncio.to_thread(directory.mkdir, mode=0o700)
    sizes = {
        "reference": MAX_PROFILE_REFERENCE_BYTES,
        "example": MAX_PROFILE_EXAMPLE_BYTES,
        "transcript": MAX_TEXT * 4,
    }
    filenames = {
        "reference": "reference.wav",
        "example": "example.ogg",
        "transcript": "transcript.txt",
    }
    received: set[str] = set()
    try:
        reader = await request.multipart()
        async with asyncio.timeout(60):
            while part := await reader.next():
                if not isinstance(part, BodyPartReader) or part.name not in sizes:
                    raise web.HTTPBadRequest(text="unexpected profile field")
                name = part.name
                if name in received:
                    raise web.HTTPBadRequest(text="duplicate profile field")
                received.add(name)
                size = 0
                with (directory / filenames[name]).open("xb") as output:
                    while chunk := await part.read_chunk(64 * 1024):
                        size += len(chunk)
                        if size > sizes[name]:
                            raise web.HTTPRequestEntityTooLarge(
                                max_size=sizes[name], actual_size=size
                            )
                        await asyncio.to_thread(output.write, chunk)
        if received != set(sizes):
            raise web.HTTPBadRequest(text="profile field missing")
        transcript = (await asyncio.to_thread((directory / "transcript.txt").read_text)).strip()
        if not transcript or len(transcript) > MAX_TEXT:
            raise web.HTTPUnprocessableEntity(text="invalid reference transcript")
        try:
            reference = await validate_reference(directory / "reference.wav")
            example = await validate_result(directory / "example.ogg")
            if (
                reference.sample_rate not in (24000, 44100)
                or reference.channels != 1
                or reference.codec != "pcm_s16le"
                or example.duration > 20
            ):
                raise AudioError("Invalid normalized profile")
        except AudioError:
            raise web.HTTPUnprocessableEntity(text="invalid profile audio") from None
        await asyncio.to_thread((directory / "transcript.txt").unlink)
        profile = await asyncio.to_thread(
            store.finish_profile, job_id, lease, directory, transcript
        )
    finally:
        if await asyncio.to_thread(directory.exists):
            await asyncio.to_thread(shutil.rmtree, directory)
    return web.json_response({"status": "ready", "profile_id": profile.id})


async def reference_result(request: web.Request) -> web.Response:
    store, job_id, lease = job_arguments(request)
    snapshot, adding = await asyncio.to_thread(store.reference_job, job_id, lease)
    count = len(snapshot.reference_samples) + int(adding)
    if not 1 <= count <= 3:
        raise web.HTTPBadRequest(text="invalid reference count")
    directory = store.job_dir(job_id) / f"reference-upload-{secrets.token_hex(8)}"
    await asyncio.to_thread(directory.mkdir, mode=0o700)
    sizes = {
        "example": MAX_PROFILE_EXAMPLE_BYTES,
        "reference_texts": 32 * 1024,
        "whisper_context": 32 * 1024,
        **{f"reference_{index}": MAX_PROFILE_REFERENCE_BYTES for index in range(count)},
    }
    filenames = {
        "example": "example.ogg",
        "reference_texts": "transcripts.json",
        "whisper_context": "whisper-context.json",
        **{f"reference_{index}": Store._reference_filename(index) for index in range(count)},
    }
    required = set(sizes) - {"whisper_context"}
    received: set[str] = set()
    try:
        reader = await request.multipart()
        async with asyncio.timeout(60):
            while part := await reader.next():
                if not isinstance(part, BodyPartReader) or part.name not in sizes:
                    raise web.HTTPBadRequest(text="unexpected reference field")
                name = part.name
                if name in received:
                    raise web.HTTPBadRequest(text="duplicate reference field")
                received.add(name)
                size = 0
                with (directory / filenames[name]).open("xb") as output:
                    output_path = directory / filenames[name]
                    output_path.chmod(0o600)
                    while chunk := await part.read_chunk(64 * 1024):
                        size += len(chunk)
                        if size > sizes[name]:
                            raise web.HTTPRequestEntityTooLarge(
                                max_size=sizes[name], actual_size=size
                            )
                        await asyncio.to_thread(output.write, chunk)
        if not required <= received:
            raise web.HTTPBadRequest(text="reference field missing")
        values: object = json.loads(
            await asyncio.to_thread((directory / "transcripts.json").read_text)
        )
        if (
            not isinstance(values, list)
            or len(values) != count
            or any(
                not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT
                for value in values
            )
        ):
            raise web.HTTPBadRequest(text="invalid reference transcripts")
        texts = [str(value) for value in values]
        await asyncio.to_thread((directory / "transcripts.json").unlink)
        if "whisper_context" in received:
            WhisperContext.model_validate_json(
                await asyncio.to_thread((directory / "whisper-context.json").read_text)
            )
        try:
            for index in range(count):
                validator = (
                    validate_enrollment_reference
                    if adding and index == count - 1
                    else validate_reference
                )
                info = await validator(directory / Store._reference_filename(index))
                if (
                    info.sample_rate not in (24000, 44100)
                    or info.channels != 1
                    or info.codec != "pcm_s16le"
                ):
                    raise AudioError("Invalid normalized reference")
            example = await validate_result(directory / "example.ogg")
            if example.duration > 20:
                raise AudioError("Reference preview is too long")
        except AudioError:
            raise web.HTTPUnprocessableEntity(text="invalid reference audio") from None
        candidate = await asyncio.to_thread(store.finish_reference, job_id, lease, directory, texts)
    finally:
        if await asyncio.to_thread(directory.exists):
            await asyncio.to_thread(shutil.rmtree, directory)
    return web.json_response({"status": "ready", "profile_id": candidate.id})


async def cancelled(request: web.Request) -> web.Response:
    store, job_id, lease = job_arguments(request)
    await asyncio.to_thread(store.confirm_cancel, job_id, lease)
    return web.json_response({"status": "cancelled"})


async def maintenance(runtime: Runtime) -> None:
    while True:
        try:
            await asyncio.to_thread(runtime.store.maintain)
        except (OSError, sqlite3.Error):
            logger.error("maintenance_failed reason=storage_error")
        await asyncio.sleep(15)


async def create_app(
    config: AppConfig,
    telegram_token: str,
    worker_token: str,
    *,
    bot: Bot | None = None,
    background: bool = True,
) -> web.Application:
    if len(worker_token) < 32:
        raise ValueError("Worker token is missing or invalid")
    store = Store(
        config.data_dir / "shift.sqlite3",
        config.data_dir,
        config.owner_id,
        admin_user_ids=config.admin_user_ids,
    )
    runtime = Runtime(store=store, worker_token=worker_token)
    telegram = bot if bot is not None else Bot(token=telegram_token)
    controller = BotController(
        telegram,
        store,
        worker_available=lambda: (
            runtime.worker_seen > 0 and time.monotonic() - runtime.worker_seen < 45
        ),
    )
    app = web.Application(client_max_size=MAX_RESULT_BYTES, middlewares=[authenticate])
    app[RUNTIME] = runtime
    app[CONTROLLER] = controller
    app.add_routes(
        [
            web.get("/health", health),
            web.post("/worker/claim", claim),
            web.get("/worker/jobs/{job_id}/input", input_audio),
            web.post("/worker/jobs/{job_id}/heartbeat", heartbeat),
            web.post("/worker/jobs/{job_id}/result", result_audio),
            web.post("/worker/jobs/{job_id}/failed", failed),
            web.post("/worker/jobs/{job_id}/transcript", transcript),
            web.post("/worker/jobs/{job_id}/profile_result", profile_result),
            web.post("/worker/jobs/{job_id}/reference_result", reference_result),
            web.post("/worker/jobs/{job_id}/cancelled", cancelled),
        ]
    )

    async def lifecycle(application: web.Application) -> AsyncIterator[None]:
        await asyncio.to_thread(config.data_dir.mkdir, parents=True, exist_ok=True, mode=0o700)
        lock = (config.data_dir / "polling.lock").open("a+b")
        runtime.lock = lock
        tasks: list[asyncio.Task[None]] = []
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("shift is already running for this data directory") from None
            await asyncio.to_thread(store.initialize)
            await asyncio.to_thread(store.cancel_unauthorized_jobs)
            await asyncio.to_thread(store.recover)
            if background:
                tasks = [
                    asyncio.create_task(maintenance(runtime)),
                    asyncio.create_task(controller.deliver_loop()),
                ]
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await telegram.session.close()
            lock.close()
            runtime.lock = None

    app.cleanup_ctx.append(lifecycle)
    return app


async def run_app(config: AppConfig, telegram_token: str, worker_token: str) -> None:
    configure_safe_logging()
    app = await create_app(config, telegram_token, worker_token)
    runner = web.AppRunner(app, access_log=None)
    dispatcher = Dispatcher()
    controller = app[CONTROLLER]
    dispatcher.message.register(controller.on_message)
    dispatcher.callback_query.register(controller.on_callback)
    try:
        await runner.setup()
        site = web.TCPSite(runner, config.bind_host, config.port)
        await site.start()
        try:
            await controller.bot.set_my_commands(
                [BotCommand(command="start", description="Открыть панель голосов")]
            )
        except (TelegramAPIError, TimeoutError):
            logger.warning("command_menu_update_failed reason=network_or_api")
        await dispatcher.start_polling(
            controller.bot,
            allowed_updates=["message", "callback_query"],
            handle_as_tasks=False,
            close_bot_session=False,
        )
    finally:
        await runner.cleanup()
