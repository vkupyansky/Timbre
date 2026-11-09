import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from ml.profile.run import (
    ProfileInvalid,
    ProfileRequest,
    measure_audio,
    prepare_profile,
)
from pydantic import ValidationError


def body(tmp_path: Path) -> dict[str, str]:
    source = tmp_path / "source.wav"
    source.touch()
    return {
        "mode": "profile",
        "source_audio": str(source),
        "model_dir": str(tmp_path),
        "output_dir": str(tmp_path),
    }


def test_profile_never_overwrites_existing_reference(tmp_path: Path) -> None:
    (tmp_path / "reference.wav").write_bytes(b"existing")
    with pytest.raises(ValidationError):
        ProfileRequest.model_validate_json(json.dumps(body(tmp_path)))
    assert (tmp_path / "reference.wav").read_bytes() == b"existing"


def test_remote_source_is_rejected(tmp_path: Path) -> None:
    request = body(tmp_path)
    request["source_audio"] = "https://example.test/private.wav"
    with pytest.raises(ValidationError):
        ProfileRequest.model_validate_json(json.dumps(request))


@pytest.mark.parametrize("duration", [14.99, 15.0, 20.0, 20.01])
@pytest.mark.parametrize("sample_rate", [24000, 44100])
def test_actual_duration_bounds(
    tmp_path: Path, duration: float, sample_rate: int
) -> None:
    path = tmp_path / "source.wav"
    speech_fixture = np.sin(
        np.arange(int(sample_rate * duration), dtype=np.float32) * 0.1
    )
    sf.write(path, speech_fixture * 0.2, sample_rate)
    if 15 <= duration <= 20:
        _, quality = measure_audio(path)
        assert quality.duration_seconds == pytest.approx(duration, abs=1 / sample_rate)
        assert quality.sample_rate == sample_rate
    else:
        with pytest.raises(ProfileInvalid, match="15_to_20_seconds"):
            measure_audio(path)


def test_silence_is_not_a_reference(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(24000 * 15, dtype=np.float32), 24000)
    with pytest.raises(ProfileInvalid, match="too_quiet"):
        measure_audio(path)


def test_clipped_audio_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    sf.write(path, np.ones(24000 * 15, dtype=np.float32), 24000)
    with pytest.raises(ProfileInvalid, match="clipped"):
        measure_audio(path)


@pytest.mark.parametrize("sample_rate", [24000, 44100])
def test_mostly_silent_audio_is_rejected(tmp_path: Path, sample_rate: int) -> None:
    path = tmp_path / "source.wav"
    audio = np.zeros(sample_rate * 20, dtype=np.float32)
    audio[:sample_rate] = np.sin(np.arange(sample_rate, dtype=np.float32) * 0.1) * 0.2
    sf.write(path, audio, sample_rate)
    with pytest.raises(ProfileInvalid, match="insufficient_speech"):
        measure_audio(path)


def test_stereo_must_be_normalized_before_model(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros((24000 * 15, 2), dtype=np.float32), 24000)
    with pytest.raises(ProfileInvalid, match="mono_24_or_44khz"):
        measure_audio(path)


def test_invalid_request_does_not_log_private_filename(tmp_path: Path) -> None:
    request = body(tmp_path)
    request["mode"] = "private-recording-name"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
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
    assert result.returncode == 2
    assert "private-recording-name" not in result.stdout + result.stderr
    assert json.loads(result.stdout)["error"] == "profile_request_invalid"


@pytest.mark.parametrize("sample_rate", [24000, 44100])
def test_profile_keeps_reference_rate_and_pcm_while_asr_receives_16khz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    """Only the ASR model boundary is stubbed; file and resampler work is real CPU."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    source = tmp_path / "source.wav"
    timeline = np.arange(sample_rate * 15, dtype=np.float32) / sample_rate
    audio = 0.15 * np.sin(2 * np.pi * 200 * timeline)
    if sample_rate == 44100:
        audio += 0.1 * np.sin(2 * np.pi * 14000 * timeline)
    sf.write(source, audio, sample_rate, subtype="PCM_16")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (tmp_path / "verified.json").write_text(
        json.dumps({"revision": "a" * 40, "manifest_sha256": "b" * 64})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"lang_to_id": {"<|ru|>": 50259}})
    )
    processor = MagicMock()
    processor.return_value = SimpleNamespace(
        input_features=MagicMock(), attention_mask=MagicMock()
    )
    processor.batch_decode.return_value = ["Это проверка сохранения частоты образца."]
    model = MagicMock()
    model.to.return_value = model
    model.generation_config = SimpleNamespace()
    model.detect_language.return_value = [50259]
    model.compute_transition_scores.return_value = torch.tensor([-0.2])
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(WhisperProcessor, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(
        WhisperForConditionalGeneration, "from_pretrained", lambda *a, **k: model
    )
    request = ProfileRequest.model_validate_json(
        json.dumps(
            {
                "mode": "profile",
                "source_audio": str(source),
                "output_dir": str(prepared),
                "model_dir": str(tmp_path),
            }
        )
    )
    summary = prepare_profile(request)
    (captured,) = processor.call_args.args
    assert len(captured) == 16000 * 15
    assert processor.call_args.kwargs["sampling_rate"] == 16000
    before, before_rate = sf.read(source, dtype="float32")
    after, after_rate = sf.read(prepared / "reference.wav", dtype="float32")
    assert before_rate == after_rate == sample_rate == summary["sample_rate"]
    np.testing.assert_array_equal(before, after)
    assert sf.info(prepared / "reference.wav").subtype == "PCM_16"
