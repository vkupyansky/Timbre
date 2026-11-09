"""Single native executor. No Telegram credential and no network listener."""

import asyncio
import fcntl
import json
import logging
import os
import random
import shutil
import signal
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, Self

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from shift.audio import AudioError, to_voice, to_wav
from shift.config import atomic_private, initialize, private_dir, read_worker_token
from shift.protocol import PREVIEW_TEXT, Profile, TranscriptRequest
from shift.whisper_cache import (
    CACHE_NAME,
    MAX_CONTEXT_BYTES,
    MAX_REQUEST_BYTES,
    WhisperContext,
    load_whisper_context,
    save_whisper_context,
    wants_whisper,
)

LOG = logging.getLogger(__name__)
API = "http://127.0.0.1:8765"
MAX_JOB_SECONDS = 900


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    kind: Literal["tts", "vc", "profile", "reference"]
    text: str | None
    profile: Profile | None
    lease: str

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        if self.kind != "profile" and self.profile is None:
            raise ValueError("generation_requires_profile")
        if self.kind == "profile" and (self.profile is not None or not self.text):
            raise ValueError("profile_preparation_requires_name")
        if self.kind == "reference" and self.text not in ("add", "remove"):
            raise ValueError("reference_requires_operation")
        return self


class Transcript(BaseModel):
    reference_text: str = Field(min_length=1, max_length=2000)


class SpeechTranscript(TranscriptRequest):
    language: Literal["ru"]


async def terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def reference_path(state: Path, relative: str) -> Path:
    root = (state / "data/jobs").resolve()
    path = (state / "data" / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise AudioError("Профиль задания недоступен.")
    return path


def _tts_payload(
    state: Path,
    profile: Profile,
    text: str,
    output: Path,
    references: list[tuple[Path, str]],
) -> dict[str, object]:
    if not text.strip() or "\x00" in text:
        raise AudioError("Не удалось разобрать речь.")
    if not 1 <= len(references) <= 3:
        raise AudioError("Для голоса нужно от одного до трёх образцов.")
    reference, transcript = references[0]
    request: dict[str, object] = {
        "mode": "tts",
        "text": text,
        "reference_audio": str(reference),
        "reference_text": transcript,
        "delivery": profile.delivery,
        "native_tags": list(profile.native_tags),
        "style_instruction": profile.style_instruction,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "seed": profile.seed,
        "output": str(output),
        "model_dir": str(state / "models/tts/base"),
    }
    if len(references) > 1:
        request["extra_references"] = [
            {"reference_audio": str(path), "reference_text": transcript}
            for path, transcript in references[1:]
        ]
    return request


def snapshot_references(state: Path, profile: Profile) -> list[tuple[Path, str]]:
    return [
        (reference_path(state, sample.reference_audio), sample.reference_text)
        for sample in profile.reference_samples
    ]


def tts_request(state: Path, profile: Profile, text: str, output: Path) -> dict[str, object]:
    """Typed messages and recognized speech use exactly the same synthesis path."""
    references = snapshot_references(state, profile)
    request = _tts_payload(state, profile, text, output, references)
    context = load_whisper_context(
        state,
        profile,
        references[0][0],
        extra_references=[path for path, _ in references[1:]],
    )
    if context is not None:
        request["whisper_context"] = context
    return request


def encode_request(request: dict[str, object]) -> bytes:
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    if len(payload) > MAX_REQUEST_BYTES:
        raise AudioError("Запрос озвучки превышает допустимый размер.")
    return payload


async def _watch(
    session: aiohttp.ClientSession,
    job: Job,
    cancel: asyncio.Event,
) -> None:
    while not cancel.is_set():
        async with session.post(
            f"{API}/worker/jobs/{job.id}/heartbeat", headers={"X-Shift-Lease": job.lease}
        ) as response:
            response.raise_for_status()
            body = await response.json()
            if body.get("cancel") is True:
                cancel.set()
                return
        await asyncio.sleep(3)


async def run_model(
    project: Path,
    kind: str,
    request_path: Path,
    cancel: asyncio.Event,
    watcher: asyncio.Task[None],
) -> int | None:
    # The developer's audio comparisons share this lock with live bot jobs.
    with (project / ".shift/gpu.lock").open("a") as lock:
        while True:
            if cancel.is_set():
                return None
            if watcher.done():
                await watcher
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancel.is_set():
                    return None
                if watcher.done():
                    await watcher
                await asyncio.sleep(0.25)
        return await _run_model(project, kind, request_path, cancel, watcher)


async def _run_model(
    project: Path,
    kind: str,
    request_path: Path,
    cancel: asyncio.Event,
    watcher: asyncio.Task[None],
) -> int | None:
    process = await asyncio.create_subprocess_exec(
        str(project / "ml/.venv/bin/python"),
        "-W",
        "error",
        str(project / "ml" / kind / "run.py"),
        "--request",
        str(request_path),
        cwd=project,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        while process.returncode is None:
            if cancel.is_set():
                return None
            if watcher.done():
                await watcher
            await asyncio.sleep(0.25)
        return process.returncode
    finally:
        await terminate(process)


async def execute(
    project: Path,
    state: Path,
    session: aiohttp.ClientSession,
    job: Job,
) -> None:
    folder = state / "work/jobs" / job.id
    await asyncio.to_thread(private_dir, folder)
    headers = {"X-Shift-Lease": job.lease}
    cancel = asyncio.Event()
    watcher = asyncio.create_task(_watch(session, job, cancel))
    reason: str | None = None
    cancelled = False
    try:
        async with asyncio.timeout(MAX_JOB_SECONDS):
            wav = folder / "output.wav"
            candidate_references: list[tuple[Path, str]] | None = None
            request: dict[str, object] = {"mode": job.kind}
            if job.kind == "tts":
                if job.profile is None or job.text is None:
                    raise AudioError("Не выбран голос или отсутствует текст.")
                request = await asyncio.to_thread(tts_request, state, job.profile, job.text, wav)
            elif job.kind == "reference" and job.text == "remove":
                if job.profile is None:
                    raise AudioError("Не выбран голос.")
                candidate_references = await asyncio.to_thread(
                    snapshot_references, state, job.profile
                )
                request = _tts_payload(state, job.profile, PREVIEW_TEXT, wav, candidate_references)
            else:
                source = folder / "source.audio"
                async with session.get(
                    f"{API}/worker/jobs/{job.id}/input", headers=headers
                ) as response:
                    response.raise_for_status()
                    size = 0
                    with source.open("wb") as file:
                        async for chunk in response.content.iter_chunked(65536):
                            size += len(chunk)
                            if size > 10 * 1024 * 1024:
                                raise AudioError("Превышен размер исходного аудио.")
                            await asyncio.to_thread(file.write, chunk)
                source_wav = folder / "source.wav"
                await to_wav(
                    source,
                    source_wav,
                    max_duration=20 if job.kind in ("profile", "reference") else 120,
                    sample_rate=44100 if job.kind in ("profile", "reference") else 24000,
                )
                request["source_audio"] = str(source_wav)
                if job.kind in ("profile", "reference"):
                    request["mode"] = "profile"
                    prepared = folder / "prepared"
                    await asyncio.to_thread(private_dir, prepared)
                    request.update(output_dir=str(prepared), model_dir=str(state / "models/asr"))
                else:
                    # Keep the existing SQLite kind for old job history. Voice input
                    # now means local transcription followed by the same TTS as text.
                    request = {
                        "source_audio": str(source_wav),
                        "output": str(folder / "speech.json"),
                        "model_dir": str(state / "models/asr"),
                    }
            request_path = folder / "request.json"
            await asyncio.to_thread(atomic_private, request_path, encode_request(request))
            started = time.monotonic()
            model_kind = (
                "asr"
                if job.kind == "vc"
                else "profile"
                if job.kind == "reference" and job.text == "add"
                else "tts"
                if job.kind == "reference"
                else job.kind
            )
            returncode = await run_model(project, model_kind, request_path, cancel, watcher)
            cancelled = returncode is None
            if job.kind == "vc" and not cancelled:
                if returncode == 0:
                    speech = SpeechTranscript.model_validate_json(
                        await asyncio.to_thread((folder / "speech.json").read_text)
                    )
                    if job.profile is None:
                        raise AudioError("Не выбран голос.")
                    async with session.post(
                        f"{API}/worker/jobs/{job.id}/transcript",
                        headers=headers,
                        json={"text": speech.text},
                    ) as response:
                        response.raise_for_status()
                    request = await asyncio.to_thread(
                        tts_request, state, job.profile, speech.text, wav
                    )
                    await asyncio.to_thread(
                        atomic_private,
                        request_path,
                        encode_request(request),
                    )
                    returncode = await run_model(project, "tts", request_path, cancel, watcher)
                    cancelled = returncode is None
                elif returncode == 3:
                    reason = "speech_unrecognized"
                elif returncode == 4:
                    reason = "speech_too_long"
                elif returncode == 2:
                    reason = "invalid_audio"
            if returncode == 0 and (
                job.kind == "profile" or (job.kind == "reference" and job.text == "add")
            ):
                transcript = Transcript.model_validate_json(
                    await asyncio.to_thread((folder / "prepared/transcript.json").read_text)
                ).reference_text
                preview_request: dict[str, object] = {
                    "mode": "tts",
                    "text": PREVIEW_TEXT,
                    "reference_audio": str(folder / "prepared/reference.wav"),
                    "reference_text": transcript,
                    "output": str(wav),
                    "model_dir": str(state / "models/tts/base"),
                }
                if job.kind == "reference":
                    if job.profile is None:
                        raise AudioError("Не выбран голос.")
                    candidate_references = await asyncio.to_thread(
                        snapshot_references, state, job.profile
                    )
                    candidate_references.append((folder / "prepared/reference.wav", transcript))
                    preview_request = _tts_payload(
                        state, job.profile, PREVIEW_TEXT, wav, candidate_references
                    )
                await asyncio.to_thread(
                    atomic_private,
                    request_path,
                    encode_request(preview_request),
                )
                returncode = await run_model(project, "tts", request_path, cancel, watcher)
                cancelled = returncode is None
            if not cancelled:
                if returncode != 0:
                    reason = reason or (
                        "out_of_memory"
                        if returncode == 75
                        else "whisper_preparation_failed"
                        if returncode == 76
                        else "profile_invalid"
                        if job.kind in ("profile", "reference") and returncode == 2
                        else "model_error"
                    )
                else:
                    if job.kind == "profile":
                        reference = folder / "prepared/reference.wav"
                        transcript = Transcript.model_validate_json(
                            await asyncio.to_thread((folder / "prepared/transcript.json").read_text)
                        ).reference_text
                        output = folder / "example.ogg"
                        await to_voice(wav, output)
                        if cancel.is_set():
                            cancelled = True
                        else:
                            with reference.open("rb") as ref, output.open("rb") as example:
                                form = aiohttp.FormData()
                                form.add_field(
                                    "reference",
                                    ref,
                                    filename="reference.wav",
                                    content_type="audio/wav",
                                )
                                form.add_field(
                                    "example",
                                    example,
                                    filename="example.ogg",
                                    content_type="audio/ogg",
                                )
                                form.add_field("transcript", transcript)
                                async with session.post(
                                    f"{API}/worker/jobs/{job.id}/profile_result",
                                    data=form,
                                    headers=headers,
                                ) as response:
                                    response.raise_for_status()
                    elif job.kind == "reference":
                        if job.profile is None or candidate_references is None:
                            raise AudioError("Не подготовлены образцы голоса.")
                        output = folder / "example.ogg"
                        await to_voice(wav, output, speed=job.profile.speed)
                        if cancel.is_set():
                            cancelled = True
                        else:
                            with ExitStack() as files:
                                form = aiohttp.FormData()
                                form.add_field(
                                    "reference_texts",
                                    json.dumps(
                                        [text for _, text in candidate_references],
                                        ensure_ascii=False,
                                    ),
                                )
                                for index, (path, _) in enumerate(candidate_references):
                                    form.add_field(
                                        f"reference_{index}",
                                        files.enter_context(path.open("rb")),
                                        filename=f"reference_{index}.wav",
                                        content_type="audio/wav",
                                    )
                                form.add_field(
                                    "example",
                                    files.enter_context(output.open("rb")),
                                    filename="example.ogg",
                                    content_type="audio/ogg",
                                )
                                exported = folder / CACHE_NAME
                                if exported.is_file() and wants_whisper(job.profile):
                                    with exported.open("rb") as context_source:
                                        data = context_source.read(MAX_CONTEXT_BYTES + 1)
                                    if len(data) > MAX_CONTEXT_BYTES:
                                        raise AudioError("Контекст голоса слишком большой.")
                                    context = WhisperContext.model_validate_json(data)
                                    form.add_field("whisper_context", context.model_dump_json())
                                async with session.post(
                                    f"{API}/worker/jobs/{job.id}/reference_result",
                                    data=form,
                                    headers=headers,
                                ) as response:
                                    response.raise_for_status()
                    else:
                        output = folder / "result.ogg"
                        if job.profile is None:
                            raise AudioError("Не выбран голос.")
                        await to_voice(wav, output, speed=job.profile.speed)
                        if cancel.is_set():
                            cancelled = True
                        else:
                            if wants_whisper(job.profile):
                                await asyncio.to_thread(
                                    save_whisper_context,
                                    state,
                                    job.profile,
                                    reference_path(state, job.profile.reference_audio),
                                    folder / CACHE_NAME,
                                    extra_references=[
                                        reference_path(state, sample.reference_audio)
                                        for sample in job.profile.extra_references
                                    ],
                                )
                            with output.open("rb") as audio:
                                async with session.post(
                                    f"{API}/worker/jobs/{job.id}/result",
                                    data=audio,
                                    headers={**headers, "Content-Type": "audio/ogg"},
                                ) as response:
                                    response.raise_for_status()
                    LOG.info("job=%s elapsed=%.1f", job.id, time.monotonic() - started)
    except TimeoutError:
        reason = "timeout"
    except (AudioError, ValidationError):
        reason = "profile_invalid" if job.kind in ("profile", "reference") else "invalid_audio"
    except OSError:
        reason = "storage_error"
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await asyncio.to_thread(shutil.rmtree, folder)
    if cancelled or reason:
        endpoint = "cancelled" if cancelled else "failed"
        async with session.post(
            f"{API}/worker/jobs/{job.id}/{endpoint}",
            headers=headers,
            json={} if cancelled else {"reason": reason},
        ) as response:
            response.raise_for_status()


async def run_worker(project: Path, state: Path) -> None:
    await asyncio.to_thread(initialize, state)
    token = read_worker_token(state)
    check = await asyncio.create_subprocess_exec(
        str(project / "ml/.venv/bin/python"),
        "-W",
        "error",
        str(project / "ml/check.py"),
        "--model-dir",
        str((state / "models").resolve()),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(120):
            if await check.wait() != 0:
                raise RuntimeError("Проверка моделей и Apple GPU не пройдена.")
    except (TimeoutError, asyncio.CancelledError):
        if check.returncode is None:
            check.kill()
        await check.wait()
        raise
    private_dir(state / "work/jobs")
    with (state / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Обработчик уже запущен.") from exc
        for folder in (state / "work/jobs").iterdir():
            if folder.is_dir() and not folder.is_symlink():
                await asyncio.to_thread(shutil.rmtree, folder)
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {token}"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            failures = 0
            while True:
                try:
                    async with session.post(f"{API}/worker/claim") as response:
                        response.raise_for_status()
                        body = await response.json()
                    if body["job"] is not None:
                        await execute(project, state, session, Job.model_validate(body["job"]))
                    else:
                        await asyncio.sleep(2)
                    failures = 0
                except (aiohttp.ClientError, TimeoutError, OSError):
                    failures += 1
                    if failures == 1:
                        LOG.warning("worker_connection_error")
                    await asyncio.sleep(min(30, 2 ** min(failures, 5)) + random.random())
