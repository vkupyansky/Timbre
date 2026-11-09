import hashlib
import json
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from shift import whisper_cache
from shift.audio import AudioError
from shift.config import atomic_private, initialize
from shift.protocol import Profile, ReferenceSample
from shift.store import Store
from shift.whisper_cache import (
    CACHE_NAME,
    MAX_CONTEXT_BYTES,
    MAX_REQUEST_BYTES,
    WhisperContext,
    load_whisper_context,
    save_whisper_context,
    whisper_fingerprint,
)
from shift.worker import encode_request, tts_request
from tests.test_store import add_profile

REVISION = "a" * 40


def context_for(
    reference: Path, text: str, *, extra_references: Sequence[tuple[Path, str]] = ()
) -> WhisperContext:
    return WhisperContext(
        version=1,
        fingerprint=whisper_fingerprint(
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            reference_text=text,
            model_revision=REVISION,
            extra_references=tuple(
                (hashlib.sha256(path.read_bytes()).hexdigest(), transcript)
                for path, transcript in extra_references
            ),
        ),
        text="[whisper] Тихий голос, сохранённый без изменения текста.",
        codes=[[1, 4095] for _ in range(10)],
    )


@dataclass(frozen=True)
class CacheVoice:
    state: Path
    store: Store
    profile: Profile
    reference: Path
    export: Path
    context: WhisperContext

    @property
    def folder(self) -> Path:
        return self.state / "data/profiles" / self.profile.id


@pytest.fixture
def cache_voice(tmp_path: Path) -> CacheVoice:
    state = tmp_path / ".shift"
    initialize(state)
    store = Store(state / "queue.sqlite3", state / "data", owner_id=7)
    store.initialize()
    add_profile(store.data_dir, "anna", db=store.db)
    store.update_voice_settings(7, "anna", delivery="whisper")
    job_id = store.enqueue(1, 7, "tts", text="Проверка.")
    assert job_id is not None
    job = store.claim()
    assert job is not None and job.profile is not None
    reference = state / "data" / job.profile.reference_audio
    atomic_private(
        state / "models/tts/base/verified.json", json.dumps({"revision": REVISION}).encode()
    )
    context = context_for(reference, job.profile.reference_text)
    export = state / "work/export" / CACHE_NAME
    atomic_private(export, context.model_dump_json().encode())
    return CacheVoice(state, store, job.profile, reference, export, context)


def test_successful_context_roundtrip_is_private_and_bound_to_job(cache_voice: CacheVoice) -> None:
    voice = cache_voice
    assert load_whisper_context(voice.state, voice.profile, voice.reference) is None
    assert save_whisper_context(voice.state, voice.profile, voice.reference, voice.export)
    saved = voice.folder / CACHE_NAME
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert (
        load_whisper_context(voice.state, voice.profile, voice.reference)
        == voice.context.model_dump()
    )
    request = tts_request(voice.state, voice.profile, "ЗамОк.", voice.export.parent / "out.wav")
    assert request["whisper_context"] == voice.context.model_dump()
    assert request["text"] == "ЗамОк."
    assert len(encode_request(request)) < MAX_REQUEST_BYTES
    assert (voice.folder / "reference.wav").read_bytes() == b"anna"


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("version", "1"),
        ("fingerprint", "A" * 64),
        ("fingerprint", "a" * 63),
        ("text", ""),
        ("text", "   "),
        ("text", "а" * 301),
        ("text", "[whisper] привет\n"),
        ("text", "<|im_end|>"),
        ("text", "[whisper"),
        ("codes", [[1]] * 9),
        ("codes", [[1]] * 9 + [[1, 2]]),
        ("codes", [[]] * 10),
        ("codes", [[0] * 257] * 10),
        ("codes", [[-1]] * 10),
        ("codes", [[4096]] * 10),
        ("codes", [[True]] * 10),
        ("codes", [[1.0]] * 10),
        ("codes", [["1"]] * 10),
        ("path", "/private/reference"),
    ],
)
def test_invalid_context_is_rejected_and_treated_as_cache_miss(
    cache_voice: CacheVoice, field: str, value: object
) -> None:
    voice = cache_voice
    data = voice.context.model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        WhisperContext.model_validate(data)
    atomic_private(voice.folder / CACHE_NAME, json.dumps(data).encode())
    assert load_whisper_context(voice.state, voice.profile, voice.reference) is None


@pytest.mark.parametrize("changed", ["reference", "transcript", "revision", "other_voice"])
def test_cache_cannot_cross_reference_text_model_or_voice(
    cache_voice: CacheVoice, changed: str
) -> None:
    voice = cache_voice
    assert save_whisper_context(voice.state, voice.profile, voice.reference, voice.export)
    profile = voice.profile
    if changed == "reference":
        voice.reference.write_bytes(b"different recording")
    elif changed == "transcript":
        profile = profile.model_copy(update={"reference_text": "Другие слова."})
    elif changed == "revision":
        atomic_private(
            voice.state / "models/tts/base/verified.json",
            json.dumps({"revision": "b" * 40}).encode(),
        )
    else:
        add_profile(voice.store.data_dir, "maria")
        atomic_private(
            voice.store.data_dir / "profiles/maria" / CACHE_NAME, voice.export.read_bytes()
        )
        profile = profile.model_copy(update={"id": "maria"})
        voice.reference.write_bytes(b"maria")
    assert load_whisper_context(voice.state, profile, voice.reference) is None
    assert not save_whisper_context(voice.state, profile, voice.reference, voice.export)


@pytest.mark.parametrize("kind", ["malformed", "oversized", "public"])
def test_untrusted_cache_file_is_ignored(cache_voice: CacheVoice, kind: str) -> None:
    voice = cache_voice
    target = voice.folder / CACHE_NAME
    payload = voice.export.read_bytes()
    if kind == "malformed":
        payload = b"not json"
    elif kind == "oversized":
        payload = b" " * (MAX_CONTEXT_BYTES + 1)
    atomic_private(target, payload)
    if kind == "public":
        target.chmod(0o644)
    assert load_whisper_context(voice.state, voice.profile, voice.reference) is None
    assert not save_whisper_context(voice.state, voice.profile, voice.reference, target)


def test_deleted_profile_cache_is_removed_and_never_recreated(cache_voice: CacheVoice) -> None:
    voice = cache_voice
    assert save_whisper_context(voice.state, voice.profile, voice.reference, voice.export)
    draft = voice.store.begin_profile_draft(7, "delete", voice.profile.id)
    assert voice.store.delete_profile(7, draft.id)
    assert voice.reference.read_bytes() == b"anna"
    assert load_whisper_context(voice.state, voice.profile, voice.reference) is None
    assert not save_whisper_context(voice.state, voice.profile, voice.reference, voice.export)
    assert not voice.folder.exists()


def test_profile_removed_during_atomic_write_cannot_be_resurrected(
    cache_voice: CacheVoice, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = cache_voice
    original = whisper_cache._still_present
    moved = voice.folder.with_name("removed-anna")
    calls = 0

    def remove_before_commit(path: Path, descriptor: int) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            path.rename(moved)
        return original(path, descriptor)

    monkeypatch.setattr(whisper_cache, "_still_present", remove_before_commit)
    assert not save_whisper_context(voice.state, voice.profile, voice.reference, voice.export)
    assert not voice.folder.exists()
    assert not (moved / CACHE_NAME).exists()
    assert not list(moved.glob(".whisper-context-*.tmp"))


@pytest.mark.parametrize("kind", ["profile", "cache", "reference", "export"])
def test_cache_never_traverses_symlinks(cache_voice: CacheVoice, kind: str) -> None:
    voice = cache_voice
    outside = voice.state.parent / "outside"
    outside.mkdir()
    sentinel = outside / CACHE_NAME
    atomic_private(sentinel, voice.export.read_bytes())
    if kind == "profile":
        moved = outside / "anna"
        voice.folder.rename(moved)
        voice.folder.symlink_to(moved, target_is_directory=True)
    elif kind == "cache":
        (voice.folder / CACHE_NAME).symlink_to(sentinel)
    elif kind == "reference":
        reference = voice.folder / "reference.wav"
        reference.unlink()
        (outside / "reference.wav").write_bytes(b"anna")
        reference.symlink_to(outside / "reference.wav")
    else:
        voice.export.unlink()
        voice.export.symlink_to(sentinel)
    assert load_whisper_context(voice.state, voice.profile, voice.reference) is None
    assert not save_whisper_context(voice.state, voice.profile, voice.reference, voice.export)
    assert sentinel.read_bytes() == voice.context.model_dump_json().encode()


def test_neutral_mode_does_not_consult_cache_or_require_model_marker(
    cache_voice: CacheVoice,
) -> None:
    voice = cache_voice
    neutral = voice.profile.model_copy(update={"delivery": "neutral", "native_tags": ()})
    (voice.state / "models/tts/base/verified.json").unlink()
    assert load_whisper_context(voice.state, neutral, voice.reference) is None
    assert not save_whisper_context(voice.state, neutral, voice.reference, voice.export)
    assert "whisper_context" not in tts_request(voice.state, neutral, "Привет.", voice.export)


@pytest.mark.parametrize("tag", ["whisper", " Whispering ", "WHISPER IN SMALL VOICE"])
def test_native_whisper_aliases_use_the_cache(cache_voice: CacheVoice, tag: str) -> None:
    voice = cache_voice
    profile = voice.profile.model_copy(update={"delivery": "neutral", "native_tags": (tag,)})
    assert save_whisper_context(voice.state, profile, voice.reference, voice.export)
    assert load_whisper_context(voice.state, profile, voice.reference) == voice.context.model_dump()


def test_maximum_context_fits_request_and_limit_counts_utf8_bytes(cache_voice: CacheVoice) -> None:
    voice = cache_voice
    context = voice.context.model_copy(update={"text": "я" * 300, "codes": [[4095] * 256] * 10})
    atomic_private(voice.export, context.model_dump_json().encode())
    assert save_whisper_context(voice.state, voice.profile, voice.reference, voice.export)
    request = tts_request(voice.state, voice.profile, "я" * 2000, voice.export)
    assert len(encode_request(request)) < MAX_REQUEST_BYTES
    with pytest.raises(AudioError, match="размер"):
        encode_request({"text": "я" * 40000})


def test_fingerprint_uses_exact_utf8_text_and_canonical_json() -> None:
    text = "[whisper] ЗамОк. "
    payload = {
        "model_revision": REVISION,
        "recipe": "fish-whisper-context-v1",
        "reference_sha256": "b" * 64,
        "reference_text": text,
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    assert (
        whisper_fingerprint(reference_sha256="b" * 64, reference_text=text, model_revision=REVISION)
        == expected
    )
    assert (
        whisper_fingerprint(
            reference_sha256="b" * 64,
            reference_text=text,
            model_revision=REVISION,
            extra_references=(),
        )
        == expected
    )


def test_multiple_reference_fingerprint_matches_ordered_v2_recipe() -> None:
    references = (("b" * 64, "Первый замОк. "), ("c" * 64, "Второй зАмок."))
    expected = hashlib.sha256(
        json.dumps(
            {
                "recipe": "fish-whisper-context-v2",
                "model_revision": REVISION,
                "references": [
                    {"reference_sha256": "a" * 64, "reference_text": "Основной образец."},
                    *[
                        {"reference_sha256": digest, "reference_text": text}
                        for digest, text in references
                    ],
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    assert (
        whisper_fingerprint(
            reference_sha256="a" * 64,
            reference_text="Основной образец.",
            model_revision=REVISION,
            extra_references=references,
        )
        == expected
    )


@pytest.mark.parametrize("changed", ["primary", "audio", "text", "order", "count", "revision"])
def test_multiple_reference_fingerprint_tracks_every_ordered_input(changed: str) -> None:
    extras: tuple[tuple[str, str], ...] = (("b" * 64, "Первый."), ("c" * 64, "Второй."))
    primary, revision = "a" * 64, REVISION
    original = whisper_fingerprint(
        reference_sha256=primary,
        reference_text="Основной.",
        model_revision=revision,
        extra_references=extras,
    )
    if changed == "primary":
        primary = "d" * 64
    elif changed == "audio":
        extras = (extras[0], ("d" * 64, extras[1][1]))
    elif changed == "text":
        extras = (extras[0], (extras[1][0], extras[1][1] + " "))
    elif changed == "order":
        extras = tuple(reversed(extras))
    elif changed == "count":
        extras = extras[:1]
    else:
        revision = "d" * 40
    assert (
        whisper_fingerprint(
            reference_sha256=primary,
            reference_text="Основной.",
            model_revision=revision,
            extra_references=extras,
        )
        != original
    )


@pytest.mark.parametrize("count", [0, 1, 2])
def test_fingerprint_matches_ml_contract(count: int) -> None:
    from ml.tts.request import whisper_fingerprint as ml_fingerprint

    extras = (("b" * 64, "Первый."), ("c" * 64, "Второй."))[:count]
    assert whisper_fingerprint(
        reference_sha256="a" * 64,
        reference_text="Основной.",
        model_revision=REVISION,
        extra_references=extras,
    ) == ml_fingerprint(
        reference_sha256="a" * 64,
        reference_text="Основной.",
        model_revision=REVISION,
        extra_references=extras,
    )


@dataclass(frozen=True)
class MultiCacheVoice(CacheVoice):
    extra_references: tuple[Path, ...]


@pytest.fixture
def multi_voice(cache_voice: CacheVoice) -> MultiCacheVoice:
    voice = cache_voice
    live_samples = []
    snapshot_samples = []
    paths = []
    for index in range(2):
        relative = f"extras/reference-{index + 1}.wav"
        text = f"Дополнительный образец {index + 1}."
        live = voice.folder / relative
        snapshot = voice.reference.parent / relative
        atomic_private(live, f"additional recording {index + 1}".encode())
        atomic_private(snapshot, live.read_bytes())
        live_samples.append(ReferenceSample(reference_audio=relative, reference_text=text))
        snapshot_samples.append(
            ReferenceSample(
                reference_audio=str(snapshot.relative_to(voice.store.data_dir)),
                reference_text=text,
            )
        )
        paths.append(snapshot)
    live_profile = Profile.model_validate_json((voice.folder / "profile.json").read_bytes())
    live_profile = live_profile.model_copy(update={"extra_references": tuple(live_samples)})
    atomic_private(voice.folder / "profile.json", live_profile.model_dump_json().encode())
    profile = voice.profile.model_copy(update={"extra_references": tuple(snapshot_samples)})
    context = context_for(
        voice.reference,
        profile.reference_text,
        extra_references=tuple(
            (path, sample.reference_text)
            for path, sample in zip(paths, snapshot_samples, strict=True)
        ),
    )
    atomic_private(voice.export, context.model_dump_json().encode())
    return MultiCacheVoice(
        voice.state, voice.store, profile, voice.reference, voice.export, context, tuple(paths)
    )


def test_multiple_reference_snapshot_roundtrips_and_uses_all_bytes(
    multi_voice: MultiCacheVoice,
) -> None:
    voice = multi_voice
    assert save_whisper_context(
        voice.state,
        voice.profile,
        voice.reference,
        voice.export,
        extra_references=voice.extra_references,
    )
    assert (
        load_whisper_context(
            voice.state, voice.profile, voice.reference, extra_references=voice.extra_references
        )
        == voice.context.model_dump()
    )
    voice.extra_references[-1].write_bytes(b"changed last snapshot")
    assert (
        load_whisper_context(
            voice.state, voice.profile, voice.reference, extra_references=voice.extra_references
        )
        is None
    )
    assert not save_whisper_context(
        voice.state,
        voice.profile,
        voice.reference,
        voice.export,
        extra_references=voice.extra_references,
    )


@pytest.mark.parametrize("count", [0, 1, 3])
def test_cache_requires_complete_reference_snapshot(
    multi_voice: MultiCacheVoice, count: int
) -> None:
    voice = multi_voice
    paths = (voice.extra_references * 2)[:count]
    atomic_private(voice.folder / CACHE_NAME, voice.export.read_bytes())
    assert (
        load_whisper_context(voice.state, voice.profile, voice.reference, extra_references=paths)
        is None
    )
    assert not save_whisper_context(
        voice.state, voice.profile, voice.reference, voice.export, extra_references=paths
    )
    assert (voice.folder / CACHE_NAME).read_bytes() == voice.export.read_bytes()


@pytest.mark.parametrize("changed", ["audio", "text", "order", "removed", "path"])
def test_old_snapshot_cannot_overwrite_cache_after_live_reference_change(
    multi_voice: MultiCacheVoice, changed: str
) -> None:
    voice = multi_voice
    current = Profile.model_validate_json((voice.folder / "profile.json").read_bytes())
    samples = current.extra_references
    if changed == "audio":
        (voice.folder / samples[-1].reference_audio).write_bytes(b"new live recording")
    elif changed == "text":
        samples = (samples[0], samples[1].model_copy(update={"reference_text": "Другие слова."}))
    elif changed == "order":
        samples = tuple(reversed(samples))
    elif changed == "removed":
        samples = samples[:1]
    else:
        atomic_private(voice.folder / "replacement.wav", b"new live recording")
        samples = (
            samples[0],
            samples[1].model_copy(update={"reference_audio": "replacement.wav"}),
        )
    current = current.model_copy(update={"extra_references": samples})
    atomic_private(voice.folder / "profile.json", current.model_dump_json().encode())
    newer = voice.context.model_copy(update={"fingerprint": "d" * 64})
    atomic_private(voice.folder / CACHE_NAME, newer.model_dump_json().encode())
    assert not save_whisper_context(
        voice.state,
        voice.profile,
        voice.reference,
        voice.export,
        extra_references=voice.extra_references,
    )
    assert (voice.folder / CACHE_NAME).read_bytes() == newer.model_dump_json().encode()


@pytest.mark.parametrize(
    "location", ["snapshot_file", "snapshot_parent", "live_file", "live_parent"]
)
def test_additional_reference_symlinks_are_not_followed(
    multi_voice: MultiCacheVoice, location: str
) -> None:
    voice = multi_voice
    outside = voice.state.parent / "outside"
    outside.mkdir()
    target = (
        voice.extra_references[-1]
        if location.startswith("snapshot")
        else voice.folder / "extras/reference-2.wav"
    )
    if location.endswith("parent"):
        target = target.parent
    moved = outside / target.name
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    atomic_private(voice.folder / CACHE_NAME, voice.export.read_bytes())
    if location.startswith("snapshot"):
        assert (
            load_whisper_context(
                voice.state,
                voice.profile,
                voice.reference,
                extra_references=voice.extra_references,
            )
            is None
        )
    assert not save_whisper_context(
        voice.state,
        voice.profile,
        voice.reference,
        voice.export,
        extra_references=voice.extra_references,
    )
    assert (voice.folder / CACHE_NAME).read_bytes() == voice.export.read_bytes()


@pytest.mark.parametrize("absolute", [False, True])
def test_live_additional_reference_cannot_escape_profile(
    multi_voice: MultiCacheVoice, absolute: bool
) -> None:
    voice = multi_voice
    outside = voice.folder.parent / "outside.wav"
    atomic_private(outside, voice.extra_references[-1].read_bytes())
    current = Profile.model_validate_json((voice.folder / "profile.json").read_bytes())
    changed = current.extra_references[-1].model_copy(
        update={"reference_audio": str(outside) if absolute else "../outside.wav"}
    )
    current = current.model_copy(
        update={"extra_references": (current.extra_references[0], changed)}
    )
    atomic_private(voice.folder / "profile.json", current.model_dump_json().encode())
    assert not save_whisper_context(
        voice.state,
        voice.profile,
        voice.reference,
        voice.export,
        extra_references=voice.extra_references,
    )
    assert outside.read_bytes() == voice.extra_references[-1].read_bytes()
    assert not (voice.folder / CACHE_NAME).exists()


def test_reference_change_during_write_does_not_commit_stale_context(
    multi_voice: MultiCacheVoice, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = multi_voice
    original = whisper_cache._still_present
    calls = 0

    def change_before_commit(path: Path, descriptor: int) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            (voice.folder / "extras/reference-2.wav").write_bytes(b"new live recording")
        return original(path, descriptor)

    monkeypatch.setattr(whisper_cache, "_still_present", change_before_commit)
    assert not save_whisper_context(
        voice.state,
        voice.profile,
        voice.reference,
        voice.export,
        extra_references=voice.extra_references,
    )
    assert not (voice.folder / CACHE_NAME).exists()
    assert not list(voice.folder.glob(".whisper-context-*.tmp"))
