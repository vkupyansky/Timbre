import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from ml.tts.download import ModelFile, valid_file
from ml.tts.request import Request
from ml.tts.run import main
from pydantic import ValidationError


def payload(tmp_path: Path) -> dict[str, object]:
    reference = tmp_path / "reference.wav"
    reference.touch()
    return {
        "mode": "tts",
        "text": "Анна, вы будете завтра в пять?",
        "reference_audio": str(reference),
        "reference_text": "Образец голоса.",
        "output": str(tmp_path / "out.wav"),
        "model_dir": str(tmp_path),
    }


def test_text_is_preserved_exactly(tmp_path: Path) -> None:
    body = payload(tmp_path)
    body["text"] = " Анна, 25 рублей?\nЯ уже звонил. "
    request = Request.model_validate_json(json.dumps(body))
    assert request.text == body["text"]
    assert request.delivery == "neutral"
    assert request.native_tags == []
    assert request.style_instruction is None
    assert (request.temperature, request.top_p, request.top_k, request.seed) == (
        0.7,
        0.7,
        30,
        42,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_tags", ["happy", "sad", "calm", "angry"]),
        ("native_tags", ["happy", "happy"]),
        ("native_tags", ["[happy]"]),
        ("native_tags", [""]),
        ("native_tags", ["x" * 101]),
        ("native_tags", ["a\nb"]),
        ("native_tags", ["<|speaker:0|>"]),
        ("native_tags", ["a|>b"]),
        ("native_tags", ["a\u202eb"]),
        ("style_instruction", "x" * 501),
        ("style_instruction", "line\nbreak"),
        ("style_instruction", "<|im_end|>"),
        ("style_instruction", "[unbalanced"),
        ("temperature", -0.1),
        ("temperature", 1.1),
        ("temperature", True),
        ("temperature", float("nan")),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("top_p", "0.7"),
        ("top_k", -1),
        ("top_k", 4097),
        ("top_k", 30.1),
        ("seed", -1),
        ("seed", 4294967296),
        ("seed", True),
    ],
)
def test_native_controls_reject_unsafe_or_unsupported_values(
    tmp_path: Path, field: str, value: object
) -> None:
    body = payload(tmp_path) | {field: value}
    with pytest.raises(ValidationError):
        Request.model_validate_json(json.dumps(body))


def test_legacy_and_native_tags_share_three_cue_limit(tmp_path: Path) -> None:
    body = payload(tmp_path) | {
        "delivery": "happy",
        "native_tags": ["calm", "sad", "whisper"],
    }
    with pytest.raises(ValidationError, match="at_most_three_effective_native_cues"):
        Request.model_validate_json(json.dumps(body))
    body["native_tags"] = ["delight", "whisper", "soft tone"]
    request = Request.model_validate_json(json.dumps(body))
    assert request.effective_native_tags == ("delight", "whisper", "soft tone")


def test_native_controls_accept_edges_and_keep_case(tmp_path: Path) -> None:
    body = payload(tmp_path) | {
        "text": "Старинный зАмок и дверной замОк.",
        "native_tags": ["  calm  "],
        "style_instruction": "  Спокойно, с теплом; [soft tone].  ",
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "seed": 4294967295,
    }
    request = Request.model_validate_json(json.dumps(body))
    assert request.text == body["text"]
    assert request.native_tags == ["calm"]
    assert request.style_instruction == "Спокойно, с теплом; [soft tone]."
    body.update(temperature=1, top_p=0, top_k=4096, seed=0)
    assert Request.model_validate_json(json.dumps(body)).top_k == 4096


@pytest.mark.parametrize("invalid", ["excited", "calm", "Neutral", 1, True, None])
def test_delivery_requires_a_supported_exact_value(
    tmp_path: Path, invalid: object
) -> None:
    body = payload(tmp_path)
    body["delivery"] = invalid
    with pytest.raises(ValidationError):
        Request.model_validate_json(json.dumps(body))


@pytest.mark.parametrize("mode", ["design", "vc", "unknown"])
def test_only_supported_tts_mode_is_accepted(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ValidationError):
        Request.model_validate_json(json.dumps(payload(tmp_path) | {"mode": mode}))


def test_unknown_model_does_not_fall_back_to_another_engine(tmp_path: Path) -> None:
    from ml.tts.runtime import generate

    (tmp_path / "config.json").write_text('{"model_type":"unsupported"}')
    request = Request.model_validate_json(json.dumps(payload(tmp_path)))
    with pytest.raises(ValueError, match="fish_model_required"):
        generate(request)
    assert not request.output.exists()


@pytest.mark.parametrize("invalid", ["relative.wav", "https://example.test/audio.wav"])
def test_external_or_relative_output_rejected(tmp_path: Path, invalid: str) -> None:
    body = payload(tmp_path)
    body["output"] = invalid
    with pytest.raises(ValidationError):
        Request.model_validate_json(json.dumps(body))


@pytest.mark.parametrize(
    ("field", "value"),
    [("reference_audio", None), ("reference_text", None), ("reference_text", " ")],
)
def test_clone_requires_reference_and_exact_transcript(
    tmp_path: Path, field: str, value: object
) -> None:
    body = payload(tmp_path)
    body[field] = value
    with pytest.raises(ValidationError):
        Request.model_validate_json(json.dumps(body))


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "out.wav").write_bytes(b"existing")
    with pytest.raises(ValidationError):
        Request.model_validate_json(json.dumps(payload(tmp_path)))
    assert (tmp_path / "out.wav").read_bytes() == b"existing"


def test_corrupt_model_file_is_not_reused(tmp_path: Path) -> None:
    file = tmp_path / "weights"
    file.write_bytes(b"corrupted")
    entry = ModelFile(
        path="weights", bytes=9, sha256=hashlib.sha256(b"expected!").hexdigest()
    )
    assert not valid_file(file, entry)
    file.write_bytes(b"expected!")
    assert valid_file(file, entry)


def test_validation_does_not_leak_private_text(tmp_path: Path) -> None:
    private = "Приватный текст, который нельзя логировать"
    body = payload(tmp_path)
    body["text"] = private
    body["mode"] = "unknown"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(body))
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "run.py"),
            "--request",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 1
    assert private not in result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "error": "ValidationError",
        "stage": "validate_request",
    }


def test_real_memory_error_uses_distinct_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload(tmp_path)))
    runtime = ModuleType("ml.tts.runtime")

    def fail(_request: Request) -> None:
        raise MemoryError("private model error detail")

    monkeypatch.setattr(runtime, "generate", fail, raising=False)
    monkeypatch.setitem(sys.modules, "ml.tts.runtime", runtime)
    monkeypatch.setattr(sys, "argv", ["run.py", "--request", str(path)])
    assert main() == 75
    captured = capsys.readouterr()
    assert "private model error detail" not in captured.out + captured.err


def test_invalid_prompt_cannot_impersonate_memory_failure(tmp_path: Path) -> None:
    body = payload(tmp_path)
    body["mode"] = "invalid"
    body["text"] = "out of memory"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(body))
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "run.py"),
            "--request",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 1
