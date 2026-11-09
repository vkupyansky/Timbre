"""Reference candidates cross the real HTTP/codec boundary without neural models."""

import asyncio
import hashlib
import io
import json
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from shift import worker
from shift.audio import AudioError, validate_result
from shift.config import atomic_private, private_dir
from shift.protocol import PREVIEW_TEXT, Profile, ReferenceSample
from shift.whisper_cache import CACHE_NAME, WhisperContext, whisper_fingerprint
from tests.test_audio import make_pcm

JOB_ID = "a" * 32
LEASE = "unit-test-lease"
REVISION = "b" * 40
THIRD_TEXT = "Третий образец для проверки транспорта."
SPEECH_TEXT = "Текст, одинаковый для обоих входов."


@dataclass
class ReferenceAPI:
    project: Path
    state: Path
    profile: Profile
    source: Path
    snapshots: tuple[Path, ...]
    original_bytes: tuple[bytes, ...]
    live_cache: Path
    routes: list[tuple[str, str]] = field(default_factory=list)
    candidates: list[dict[str, bytes]] = field(default_factory=list)
    profiles: list[dict[str, bytes]] = field(default_factory=list)
    prepared_references: list[bytes] = field(default_factory=list)
    notifications: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    results: list[bytes] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    cancel_ready: asyncio.Event | None = None

    async def handle(self, request: web.Request) -> web.Response:
        assert request.match_info["id"] == JOB_ID
        assert request.headers.get("X-Shift-Lease") == LEASE
        action = request.match_info["action"]
        self.routes.append((request.method, action))
        if action == "heartbeat":
            if self.cancel_ready is not None:
                await self.cancel_ready.wait()
            return web.json_response({"cancel": self.cancel_ready is not None})
        if action == "input":
            return web.Response(body=self.source.read_bytes(), content_type="audio/wav")
        if action in ("reference_result", "profile_result"):
            fields: dict[str, bytes] = {}
            reader = await request.multipart()
            while (part := await reader.next()) is not None:
                assert isinstance(part, aiohttp.BodyPartReader)
                assert part.name is not None and part.name not in fields
                fields[part.name] = bytes(await part.read())
            (self.candidates if action == "reference_result" else self.profiles).append(fields)
            return web.json_response({"ok": True})
        if action in ("failed", "cancelled"):
            self.notifications.append((action, await request.json()))
            return web.json_response({"ok": True})
        if action == "result":
            self.results.append(await request.read())
            return web.json_response({"ok": True})
        if action == "transcript":
            self.transcripts.append((await request.json())["text"])
            return web.json_response({"status": "saved"})
        raise web.HTTPNotFound()

    async def execute(self, kind: str = "reference", text: str = "add") -> None:
        job = worker.Job.model_validate(
            {
                "id": JOB_ID,
                "kind": kind,
                "text": text,
                "profile": None if kind == "profile" else self.profile,
                "lease": LEASE,
            }
        )
        async with aiohttp.ClientSession() as session:
            await worker.execute(self.project, self.state, session, job)

    def assert_clean(self) -> None:
        assert not (self.state / "work/jobs" / JOB_ID).exists()
        assert tuple(path.read_bytes() for path in self.snapshots) == self.original_bytes
        assert self.live_cache.read_bytes() == b"live cache must remain unchanged"


@pytest.fixture
async def reference_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[ReferenceAPI]:
    state = tmp_path / ".shift"
    folder = state / "data/jobs" / JOB_ID
    private_dir(folder)
    snapshots = (folder / "reference.wav", folder / "reference-2.wav")
    for index, path in enumerate(snapshots):
        make_pcm(path, 16, amplitude=9000 - index * 1000)
    source = tmp_path / "input.wav"
    make_pcm(source, 16, amplitude=7000)
    live_cache = state / "data/profiles/anna" / CACHE_NAME
    atomic_private(live_cache, b"live cache must remain unchanged")
    atomic_private(
        state / "models/tts/base/verified.json", json.dumps({"revision": REVISION}).encode()
    )
    profile = Profile(
        id="anna",
        name="Тест транспорта",
        reference_audio=f"jobs/{JOB_ID}/reference.wav",
        reference_text="Первый образец.",
        extra_references=(
            ReferenceSample(
                reference_audio=f"jobs/{JOB_ID}/reference-2.wav", reference_text="Второй образец."
            ),
        ),
        delivery="whisper",
        speed=0.9,
        native_tags=("soft tone",),
        style_instruction="Спокойная подача.",
        temperature=0.6,
        top_p=0.8,
        top_k=40,
        seed=123,
    )
    api = ReferenceAPI(
        tmp_path,
        state,
        profile,
        source,
        snapshots,
        tuple(path.read_bytes() for path in snapshots),
        live_cache,
    )
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.router.add_route("*", "/worker/jobs/{id}/{action}", api.handle)
    async with TestServer(app) as server:
        monkeypatch.setattr(worker, "API", str(server.make_url("/")).rstrip("/"))
        yield api


def references_in(payload: dict[str, object]) -> list[tuple[Path, str]]:
    references = [(Path(str(payload["reference_audio"])), str(payload["reference_text"]))]
    extras = payload.get("extra_references", [])
    assert isinstance(extras, list)
    for sample in extras:
        assert isinstance(sample, dict)
        references.append((Path(sample["reference_audio"]), sample["reference_text"]))
    return references


def assert_pcm_reference(audio: bytes, sample_rate: int) -> None:
    with wave.open(io.BytesIO(audio), "rb") as reference:
        assert reference.getframerate() == sample_rate
        assert reference.getnframes() == 16 * sample_rate
        assert reference.getnchannels() == 1
        assert reference.getsampwidth() == 2


def unit_context(references: list[tuple[Path, str]]) -> WhisperContext:
    fingerprints = [
        (hashlib.sha256(path.read_bytes()).hexdigest(), text) for path, text in references
    ]
    return WhisperContext(
        version=1,
        fingerprint=whisper_fingerprint(
            reference_sha256=fingerprints[0][0],
            reference_text=fingerprints[0][1],
            model_revision=REVISION,
            extra_references=fingerprints[1:],
        ),
        text="[whisper] Контекст только для проверки формата.",
        codes=[[0, 1] for _ in range(10)],
    )


def install_model_boundary(
    api: ReferenceAPI,
    monkeypatch: pytest.MonkeyPatch,
    *,
    export_context: bool = False,
    invalid_profile: bool = False,
    cancel_preview: bool = False,
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    async def model(
        project: Path, kind: str, request: Path, cancel: asyncio.Event, watcher: asyncio.Task[None]
    ) -> int | None:
        payload = json.loads(await asyncio.to_thread(request.read_text))
        calls.append((kind, payload))
        if kind == "profile":
            assert payload["mode"] == "profile"
            normalized = await asyncio.to_thread(Path(payload["source_audio"]).read_bytes)
            assert_pcm_reference(normalized, 44100)
            if invalid_profile:
                return 2
            prepared = Path(payload["output_dir"])
            atomic_private(prepared / "reference.wav", normalized)
            api.prepared_references.append(normalized)
            atomic_private(
                prepared / "transcript.json", json.dumps({"reference_text": THIRD_TEXT}).encode()
            )
        elif kind == "asr":
            normalized = await asyncio.to_thread(Path(payload["source_audio"]).read_bytes)
            assert_pcm_reference(normalized, 24000)
            atomic_private(
                Path(payload["output"]),
                json.dumps({"text": SPEECH_TEXT, "language": "ru"}).encode(),
            )
        else:
            assert kind == "tts"
            if cancel_preview:
                assert api.cancel_ready is not None
                api.cancel_ready.set()
                await asyncio.wait_for(cancel.wait(), 3)
                return None
            output = Path(payload["output"])
            make_pcm(output, 2)
            if export_context:
                context = unit_context(references_in(payload))
                atomic_private(output.parent / CACHE_NAME, context.model_dump_json().encode())
        return 0

    monkeypatch.setattr(worker, "run_model", model)
    return calls


def forbid_live_cache_save(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> bool:
        raise AssertionError("A candidate must not update the live profile cache")

    monkeypatch.setattr(worker, "save_whisper_context", fail)


def assert_voice_settings(payload: dict[str, object], profile: Profile) -> None:
    for name in ("delivery", "style_instruction", "temperature", "top_p", "top_k", "seed"):
        assert payload[name] == getattr(profile, name)
    assert payload["native_tags"] == list(profile.native_tags)


@pytest.mark.parametrize("operation", ["add", "remove"])
@pytest.mark.parametrize("export_context", [False, True])
async def test_reference_candidate_preserves_snapshot_and_uploads_only_preview(
    reference_api: ReferenceAPI,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    export_context: bool,
) -> None:
    api = reference_api
    forbid_live_cache_save(monkeypatch)
    calls = install_model_boundary(api, monkeypatch, export_context=export_context)
    await api.execute(text=operation)
    assert [kind for kind, _ in calls] == (["profile", "tts"] if operation == "add" else ["tts"])
    assert api.routes.count(("GET", "input")) == int(operation == "add")
    assert not api.notifications and not api.results
    assert {action for _, action in api.routes} <= {"heartbeat", "input", "reference_result"}
    assert len(api.candidates) == 1
    fields = api.candidates[0]
    texts = [sample.reference_text for sample in api.profile.reference_samples]
    if operation == "add":
        texts.append(THIRD_TEXT)
    assert json.loads(fields["reference_texts"]) == texts
    expected_fields = {"reference_texts", "example", *(f"reference_{i}" for i in range(len(texts)))}
    assert set(fields) == expected_fields | ({"whisper_context"} if export_context else set())
    for index, original in enumerate(api.original_bytes):
        assert fields[f"reference_{index}"] == original
        assert_pcm_reference(fields[f"reference_{index}"], 24000)
    if operation == "add":
        assert fields["reference_2"] == api.prepared_references[0]
        assert_pcm_reference(fields["reference_2"], 44100)
    payload = calls[-1][1]
    assert payload["text"] == PREVIEW_TEXT
    assert_voice_settings(payload, api.profile)
    references = references_in(payload)
    assert [text for _, text in references] == texts
    assert [path for path, _ in references[:2]] == list(api.snapshots)
    assert "whisper_context" not in payload
    if export_context:
        received = WhisperContext.model_validate_json(fields["whisper_context"])
        digest_texts = [
            (hashlib.sha256(fields[f"reference_{i}"]).hexdigest(), text)
            for i, text in enumerate(texts)
        ]
        assert received.fingerprint == whisper_fingerprint(
            reference_sha256=digest_texts[0][0],
            reference_text=digest_texts[0][1],
            model_revision=REVISION,
            extra_references=digest_texts[1:],
        )
    preview = api.project / "received-preview.ogg"
    preview.write_bytes(fields["example"])
    info = await validate_result(preview)
    assert info.codec == "opus" and 2.1 < info.duration < 2.3
    api.assert_clean()


async def test_new_profile_preserves_44100_reference_through_preview_upload(
    reference_api: ReferenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = reference_api
    forbid_live_cache_save(monkeypatch)
    calls = install_model_boundary(api, monkeypatch)
    await api.execute(kind="profile", text="Новый голос")
    assert [kind for kind, _ in calls] == ["profile", "tts"]
    assert api.routes.count(("GET", "input")) == 1
    assert {action for _, action in api.routes} <= {"heartbeat", "input", "profile_result"}
    assert not api.notifications and not api.results and not api.candidates
    assert len(api.profiles) == len(api.prepared_references) == 1
    fields = api.profiles[0]
    assert set(fields) == {"reference", "example", "transcript"}
    assert fields["reference"] == api.prepared_references[0]
    assert_pcm_reference(fields["reference"], 44100)
    assert fields["transcript"].decode() == THIRD_TEXT
    payload = calls[-1][1]
    assert payload["reference_text"] == THIRD_TEXT
    assert payload["text"] == PREVIEW_TEXT
    assert "extra_references" not in payload
    preview = api.project / "new-profile-preview.ogg"
    preview.write_bytes(fields["example"])
    info = await validate_result(preview)
    assert info.codec == "opus" and 1.95 < info.duration < 2.05
    api.assert_clean()


async def test_invalid_reference_never_generates_or_uploads_preview(
    reference_api: ReferenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = reference_api
    forbid_live_cache_save(monkeypatch)
    calls = install_model_boundary(api, monkeypatch, invalid_profile=True)
    await api.execute()
    assert [kind for kind, _ in calls] == ["profile"]
    assert api.notifications == [("failed", {"reason": "profile_invalid"})]
    assert not api.candidates and not api.results
    assert not any(action.endswith("result") for _, action in api.routes)
    api.assert_clean()


async def test_cancelled_candidate_cleans_work_and_reports_cancellation(
    reference_api: ReferenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = reference_api
    api.cancel_ready = asyncio.Event()
    forbid_live_cache_save(monkeypatch)
    calls = install_model_boundary(api, monkeypatch, cancel_preview=True)
    await api.execute()
    assert [kind for kind, _ in calls] == ["profile", "tts"]
    assert ("POST", "heartbeat") in api.routes
    assert api.notifications == [("cancelled", {})]
    assert not api.candidates and not api.results
    assert not any(action.endswith("result") for _, action in api.routes)
    api.assert_clean()


async def test_typed_and_recognized_text_pass_same_complete_reference_set(
    reference_api: ReferenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = reference_api
    calls = install_model_boundary(api, monkeypatch)
    saved: list[list[Path]] = []

    def save(
        state: Path,
        profile: Profile,
        reference: Path,
        exported: Path,
        *,
        extra_references: list[Path],
    ) -> bool:
        saved.append([reference, *extra_references])
        return False

    monkeypatch.setattr(worker, "save_whisper_context", save)
    await api.execute(kind="tts", text=SPEECH_TEXT)
    make_pcm(api.source, 16, amplitude=7000, rate=44100)
    await api.execute(kind="vc", text=SPEECH_TEXT)
    assert [kind for kind, _ in calls] == ["tts", "asr", "tts"]
    assert calls[0][1] == calls[-1][1]
    assert calls[0][1]["text"] == SPEECH_TEXT
    assert_voice_settings(calls[0][1], api.profile)
    assert references_in(calls[0][1]) == [
        (path, sample.reference_text)
        for path, sample in zip(api.snapshots, api.profile.reference_samples, strict=True)
    ]
    assert saved == [list(api.snapshots), list(api.snapshots)]
    assert len(api.results) == 2 and not api.notifications and not api.candidates
    assert api.transcripts == [SPEECH_TEXT]
    api.assert_clean()


@pytest.mark.parametrize("symlink", [False, True])
async def test_extra_snapshot_cannot_escape_job_data(
    reference_api: ReferenceAPI, symlink: bool
) -> None:
    api = reference_api
    outside = api.project / "outside.wav"
    outside.write_bytes(api.original_bytes[1])
    if symlink:
        extra = api.snapshots[-1]
        extra.unlink()
        extra.symlink_to(outside)
        relative = api.profile.extra_references[0].reference_audio
    else:
        relative = "../../outside.wav"
    profile = api.profile.model_copy(
        update={
            "extra_references": (
                ReferenceSample(reference_audio=relative, reference_text="Второй образец."),
            )
        }
    )
    with pytest.raises(AudioError, match="недоступен"):
        worker.tts_request(api.state, profile, SPEECH_TEXT, api.project / "output.wav")
    assert outside.read_bytes() == api.original_bytes[1]
