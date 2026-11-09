import asyncio
import json
import secrets
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from shift import worker
from shift.app import RUNTIME, AppConfig, create_app
from shift.config import atomic_private, initialize
from shift.protocol import ClaimResponse
from shift.whisper_cache import CACHE_NAME
from tests.test_api import AppTestClient, headers
from tests.test_audio import make_pcm
from tests.test_store import add_profile
from tests.test_whisper_cache import REVISION, context_for


@pytest.fixture
async def speech_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AppTestClient]:
    state = tmp_path / ".shift"
    initialize(state)
    add_profile(state / "data", "anna")
    metadata = state / "data/profiles/anna/profile.json"
    profile = json.loads(metadata.read_text())
    profile.update(
        delivery="angry",
        speed=0.9,
        native_tags=["soft tone"],
        style_instruction="Speak warmly.",
        temperature=0.6,
        top_p=0.8,
        top_k=40,
        seed=123,
    )
    metadata.write_text(json.dumps(profile))
    app = await create_app(
        AppConfig(owner_id=7, data_dir=state / "data"),
        "123:" + secrets.token_urlsafe(30),
        secrets.token_urlsafe(32),
        background=False,
    )
    async with TestClient(TestServer(app)) as client:
        monkeypatch.setattr(worker, "API", str(client.make_url("/")).rstrip("/"))
        yield client


async def execute_claim(client: AppTestClient, project: Path) -> str:
    response = await client.post("/worker/claim", headers=headers(client))
    claimed = ClaimResponse.model_validate_json(await response.read()).job
    assert claimed is not None
    job = worker.Job.model_validate(claimed.model_dump())
    async with aiohttp.ClientSession(headers=headers(client)) as session:
        await worker.execute(project, project / ".shift", session, job)
    return job.id


async def test_speech_and_typed_text_use_identical_tts_settings_and_clean_transcript(
    speech_client: AppTestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def model(
        project: Path, kind: str, request: Path, cancel: asyncio.Event, watcher: asyncio.Task[None]
    ) -> int:
        payload = json.loads(await asyncio.to_thread(request.read_text))
        calls.append((kind, payload))
        if kind == "asr":
            await asyncio.to_thread(
                Path(payload["output"]).write_text,
                json.dumps({"text": "Это проверка.", "language": "ru"}),
            )
        else:
            assert kind == "tts"  # No direct voice conversion may run.
            make_pcm(Path(payload["output"]), 2)
        return 0

    monkeypatch.setattr(worker, "run_model", model)
    store = speech_client.app[RUNTIME].store
    source = tmp_path / "input.wav"
    make_pcm(source, 2)
    store.enqueue(1, 7, "vc", source=source)
    first = await execute_claim(speech_client, tmp_path)
    assert [kind for kind, _ in calls] == ["asr", "tts"]
    assert store.list_jobs(7)[0].status == "ready"
    assert list(store.job_dir(first).iterdir()) == [store.job_dir(first) / "result.ogg"]
    assert not (tmp_path / ".shift/work/jobs" / first).exists()
    with sqlite3.connect(store.db) as connection:
        assert connection.execute(
            "SELECT transcript FROM generation_history WHERE job_id=?", (first,)
        ).fetchone() == ("Это проверка.",)
    store.enqueue(2, 7, "tts", text="Это проверка.")
    await execute_claim(speech_client, tmp_path)
    assert [kind for kind, _ in calls] == ["asr", "tts", "tts"]
    for field in (
        "mode",
        "text",
        "reference_text",
        "delivery",
        "model_dir",
        "native_tags",
        "style_instruction",
        "temperature",
        "top_p",
        "top_k",
        "seed",
    ):
        assert calls[1][1][field] == calls[2][1][field]
    assert calls[1][1]["delivery"] == "angry"
    assert calls[1][1]["native_tags"] == ["soft tone"]
    assert calls[1][1]["style_instruction"] == "Speak warmly."
    assert calls[1][1]["temperature"] == 0.6
    assert calls[1][1]["top_p"] == 0.8
    assert calls[1][1]["top_k"] == 40
    assert calls[1][1]["seed"] == 123
    from shift.audio import validate_result

    assert (await validate_result(store.job_dir(first) / "result.ogg")).duration > 2.1


@pytest.mark.parametrize("code,reason", [(3, "speech_unrecognized"), (4, "speech_too_long")])
async def test_failed_transcription_never_synthesizes_or_returns_original(
    speech_client: AppTestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    reason: str,
) -> None:
    calls: list[str] = []

    async def model(
        project: Path, kind: str, request: Path, cancel: asyncio.Event, watcher: asyncio.Task[None]
    ) -> int:
        calls.append(kind)
        return code

    monkeypatch.setattr(worker, "run_model", model)
    store = speech_client.app[RUNTIME].store
    source = tmp_path / "input.wav"
    make_pcm(source, 2)
    store.enqueue(1, 7, "vc", source=source)
    identifier = await execute_claim(speech_client, tmp_path)
    assert calls == ["asr"]
    job = store.list_jobs(7)[0]
    assert job.status == "failed" and job.failure == reason
    assert not list(store.job_dir(identifier).iterdir())
    assert not (tmp_path / ".shift/work/jobs" / identifier).exists()


@pytest.mark.parametrize("delete_during_generation", [False, True])
async def test_successful_whisper_is_reused_for_both_inputs_without_restoring_deleted_profile(
    speech_client: AppTestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delete_during_generation: bool,
) -> None:
    store = speech_client.app[RUNTIME].store
    store.update_voice_settings(7, "anna", changes={"native_tags": ["whisper"]})
    state = tmp_path / ".shift"
    atomic_private(
        state / "models/tts/base/verified.json", json.dumps({"revision": REVISION}).encode()
    )
    tts_calls: list[dict[str, object]] = []

    async def model(
        project: Path, kind: str, request: Path, cancel: asyncio.Event, watcher: asyncio.Task[None]
    ) -> int:
        payload = json.loads(await asyncio.to_thread(request.read_text))
        if kind == "asr":
            await asyncio.to_thread(
                Path(payload["output"]).write_text,
                json.dumps({"text": "Это проверка.", "language": "ru"}),
            )
            return 0
        assert kind == "tts"
        tts_calls.append(payload)
        output = Path(payload["output"])
        reference = Path(payload["reference_audio"])
        context = context_for(reference, payload["reference_text"])
        if len(tts_calls) == 1:
            assert "whisper_context" not in payload
        else:
            assert payload["whisper_context"] == context.model_dump()
        if delete_during_generation:
            draft = store.begin_profile_draft(7, "delete", "anna")
            assert store.delete_profile(7, draft.id)
            assert await asyncio.to_thread(reference.exists)
        atomic_private(output.parent / CACHE_NAME, context.model_dump_json().encode())
        make_pcm(output, 2)
        return 0

    monkeypatch.setattr(worker, "run_model", model)
    source = tmp_path / "input.wav"
    make_pcm(source, 2)
    store.enqueue(1, 7, "vc", source=source)
    first = await execute_claim(speech_client, tmp_path)
    assert store.list_jobs(7)[0].status == "ready"
    assert (store.job_dir(first) / "result.ogg").exists()
    folder = store.data_dir / "profiles/anna"
    if delete_during_generation:
        assert not folder.exists()
    else:
        assert (folder / CACHE_NAME).stat().st_mode & 0o777 == 0o600
        store.enqueue(2, 7, "tts", text="Это проверка.")
        await execute_claim(speech_client, tmp_path)
        assert len(tts_calls) == 2
        assert store.list_jobs(7)[0].status == "ready"
    assert not (state / "work/jobs" / first).exists()


@pytest.mark.parametrize("kind", ["tts", "vc"])
async def test_whisper_failure_is_specific_and_does_not_persist_bootstrap(
    speech_client: AppTestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    store = speech_client.app[RUNTIME].store
    store.update_voice_settings(7, "anna", changes={"native_tags": ["whisper"]})
    state = tmp_path / ".shift"
    atomic_private(
        state / "models/tts/base/verified.json", json.dumps({"revision": REVISION}).encode()
    )

    async def model(
        project: Path, mode: str, request: Path, cancel: asyncio.Event, watcher: asyncio.Task[None]
    ) -> int:
        payload = json.loads(await asyncio.to_thread(request.read_text))
        if mode == "asr":
            await asyncio.to_thread(
                Path(payload["output"]).write_text,
                json.dumps({"text": "Это проверка.", "language": "ru"}),
            )
            return 0
        context = context_for(Path(payload["reference_audio"]), payload["reference_text"])
        atomic_private(
            Path(payload["output"]).parent / CACHE_NAME, context.model_dump_json().encode()
        )
        return 76

    monkeypatch.setattr(worker, "run_model", model)
    if kind == "vc":
        source = tmp_path / "input.wav"
        make_pcm(source, 2)
        store.enqueue(1, 7, "vc", source=source)
    else:
        store.enqueue(1, 7, "tts", text="Это проверка.")
    identifier = await execute_claim(speech_client, tmp_path)
    job = store.list_jobs(7)[0]
    assert job.status == "failed" and job.failure == "whisper_preparation_failed"
    assert not (store.data_dir / "profiles/anna" / CACHE_NAME).exists()
    assert not list(store.job_dir(identifier).iterdir())
    assert not (state / "work/jobs" / identifier).exists()
