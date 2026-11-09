"""Private whisper history validation and process boundary, without loading ML."""

import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from ml.tts.request import Request, Summary, WhisperContext, whisper_fingerprint
from ml.tts.run import main
from ml.tts.style_context import WhisperUnavailable
from pydantic import ValidationError


def context_payload() -> dict[str, object]:
    return {
        "version": 1,
        "fingerprint": "a" * 64,
        "text": "[whisper] Приватный шёпотный контекст.",
        "codes": [[0, 4095] for _ in range(10)],
    }


@pytest.mark.parametrize("frames", [1, 256])
def test_context_accepts_rectangular_code_bounds(frames: int) -> None:
    codes = [[0 if row % 2 else 4095] * frames for row in range(10)]
    payload = context_payload() | {"codes": codes}
    context = WhisperContext.model_validate_json(json.dumps(payload))
    assert context.codes == codes
    assert context.text == payload["text"]
    assert context.version == 1


@pytest.mark.parametrize(
    "codes",
    [
        [],
        [[0]] * 9,
        [[0]] * 11,
        [[]] * 10,
        [[0] * 257] * 10,
        [[0]] * 9 + [[0, 1]],
        [[True]] * 10,
        [[1.0]] * 10,
        [["1"]] * 10,
        [[-1]] * 10,
        [[4096]] * 10,
        [[None]] * 10,
    ],
)
def test_context_rejects_invalid_codec_history(codes: object) -> None:
    with pytest.raises(ValidationError):
        WhisperContext.model_validate_json(
            json.dumps(context_payload() | {"codes": codes})
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("version", 1.0),
        ("version", "1"),
        ("version", 2),
        ("fingerprint", "a" * 63),
        ("fingerprint", "a" * 65),
        ("fingerprint", "A" * 64),
        ("fingerprint", "g" * 64),
        ("fingerprint", "a" * 64 + "\n"),
        ("text", ""),
        ("text", "   "),
        ("text", "а" * 301),
        ("text", "<|im_end|>"),
        ("text", "приватное|>"),
        ("text", "приватное\x00"),
        ("text", "приватное\nсообщение"),
        ("text", "приватное\u202e"),
        ("text", "приватное\ue000"),
        ("text", "[whisper"),
        ("unexpected", "private"),
    ],
)
def test_context_rejects_unsafe_metadata(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        WhisperContext.model_validate_json(
            json.dumps(context_payload() | {field: value})
        )


@pytest.fixture
def request_payload(tmp_path: Path) -> dict[str, object]:
    reference = tmp_path / "reference.wav"
    reference.touch()
    return {
        "mode": "tts",
        "text": "Приватный текст нового сообщения.",
        "reference_audio": str(reference),
        "reference_text": "Приватная расшифровка образца.",
        "output": str(tmp_path / "output.wav"),
        "model_dir": str(tmp_path),
    }


def test_legacy_request_has_no_context(request_payload: dict[str, object]) -> None:
    request = Request.model_validate_json(json.dumps(request_payload))
    assert request.whisper_context is None
    assert not request.uses_whisper
    request_payload["delivery"] = "whisper"
    assert Request.model_validate_json(json.dumps(request_payload)).uses_whisper


@pytest.mark.parametrize(
    "tag", ["  WHISPER  ", "  Whispering  ", "  Whisper In Small Voice  "]
)
def test_whisper_aliases_ignore_outer_spaces_and_case(
    request_payload: dict[str, object], tag: str
) -> None:
    request_payload["native_tags"] = [tag]
    assert Request.model_validate_json(json.dumps(request_payload)).uses_whisper


def test_whisper_in_plain_text_is_not_a_delivery_setting(
    request_payload: dict[str, object],
) -> None:
    request_payload.update(text="Произнеси whisper.", native_tags=["soft tone"])
    assert not Request.model_validate_json(json.dumps(request_payload)).uses_whisper


def test_fingerprint_matches_fixed_canonical_recipe() -> None:
    assert (
        whisper_fingerprint(
            reference_sha256="a" * 64,
            reference_text="Образец голоса.",
            model_revision="revision-1",
        )
        == "5d39a10f0c66051b16cb39e2a190a934716d16609d3fc48679b02ff9129d0b97"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_sha256", "b" * 64),
        ("reference_text", "Образец голоса. "),
        ("model_revision", "revision-2"),
    ],
)
def test_fingerprint_invalidates_changed_voice_or_model(field: str, value: str) -> None:
    identity = {
        "reference_sha256": "a" * 64,
        "reference_text": "Образец голоса.",
        "model_revision": "revision-1",
    }
    assert whisper_fingerprint(**identity) != whisper_fingerprint(
        **(identity | {field: value})
    )


@pytest.fixture
def boundary_process(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name in (
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(name, raising=False)
    previous_umask = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous_umask)


def install_runtime(
    monkeypatch: pytest.MonkeyPatch, generate: Callable[[Request], Summary]
) -> None:
    runtime = ModuleType("ml.tts.runtime")
    monkeypatch.setattr(runtime, "generate", generate, raising=False)
    monkeypatch.setitem(sys.modules, "ml.tts.runtime", runtime)


@pytest.mark.usefixtures("boundary_process")
def test_whisper_failure_has_private_distinct_exit(
    tmp_path: Path,
    request_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_payload["whisper_context"] = context_payload()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request_payload))
    detail = "PRIVATE exception with out of memory and private prompt"

    def unavailable(request: Request) -> Summary:
        assert request.whisper_context is not None
        raise WhisperUnavailable(f"{detail}: {request.text}")

    install_runtime(monkeypatch, unavailable)
    monkeypatch.setattr(sys, "argv", ["run.py", "--request", str(path)])
    assert main() == 76
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "error": "WhisperUnavailable",
        "stage": "model_generation",
    }
    assert captured.err == ""
    assert detail not in captured.out
    assert "Приватн" not in captured.out


@pytest.mark.usefixtures("boundary_process")
@pytest.mark.parametrize("oversized", [False, True])
def test_context_request_size_boundary(
    tmp_path: Path,
    request_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    oversized: bool,
) -> None:
    request_payload["whisper_context"] = context_payload() | {
        "codes": [[4095] * 256 for _ in range(10)]
    }
    encoded = json.dumps(request_payload, indent=4)
    assert 32 * 1024 < len(encoded.encode()) < 64 * 1024
    if oversized:
        encoded += " " * (64 * 1024 + 1 - len(encoded.encode()))
    path = tmp_path / "request.json"
    path.write_text(encoded)
    received: list[Request] = []

    def generate(request: Request) -> Summary:
        received.append(request)
        return Summary(
            mode="tts",
            duration_seconds=1,
            elapsed_seconds=0,
            generation_seconds=0,
            sample_rate=44100,
            peak_mlx_bytes=0,
            peak_rss_bytes=0,
            model_revision="boundary-unit-test",
            decoder_mode="bounded_causal",
        )

    install_runtime(monkeypatch, generate)
    monkeypatch.setattr(sys, "argv", ["run.py", "--request", str(path)])
    assert main() == (1 if oversized else 0)
    captured = capsys.readouterr()
    assert captured.err == ""
    if oversized:
        assert received == []
        assert json.loads(captured.out) == {
            "error": "ValueError",
            "stage": "validate_request",
        }
    else:
        assert len(received) == 1
        assert received[0].whisper_context == WhisperContext.model_validate(
            request_payload["whisper_context"]
        )
        assert Summary.model_validate_json(captured.out).sample_rate == 44100
    assert "Приватн" not in captured.out
