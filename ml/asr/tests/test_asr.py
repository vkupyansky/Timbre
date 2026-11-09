import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf
import torch
from ml.asr.run import (
    ASRRequest,
    SpeechUnclear,
    TranscriptTooLong,
    configure_longform_compatibility,
    decode_audio,
    read_audio,
    validate_segments,
    validate_text,
    write_transcript,
)
from pydantic import ValidationError


def request_body(tmp_path: Path) -> dict[str, str]:
    source = tmp_path / "source.wav"
    source.touch()
    return {
        "source_audio": str(source),
        "output": str(tmp_path / "transcript.json"),
        "model_dir": str(tmp_path),
    }


@pytest.mark.parametrize("invalid", ["relative.wav", "https://example.test/a.wav"])
def test_nonlocal_source_rejected(tmp_path: Path, invalid: str) -> None:
    body = request_body(tmp_path)
    body["source_audio"] = invalid
    with pytest.raises(ValidationError):
        ASRRequest.model_validate_json(json.dumps(body))


def test_message_content_cannot_enter_request_as_instructions(tmp_path: Path) -> None:
    body = request_body(tmp_path) | {"prompt": "invent different words"}
    with pytest.raises(ValidationError):
        ASRRequest.model_validate_json(json.dumps(body))


def test_short_one_word_duration_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    samples = np.sin(np.arange(4800, dtype=np.float32) * 0.1) * 0.1
    sf.write(path, samples, 24000, subtype="PCM_16")
    assert len(read_audio(path)) == 4800
    assert validate_text(" Да. ") == "Да."


def test_silence_rejected_before_model(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(24000 * 32), 24000, subtype="PCM_16")
    with pytest.raises(SpeechUnclear, match="no_audible"):
        read_audio(path)


def test_long_pauses_do_not_invalidate_short_speech(tmp_path: Path) -> None:
    path = tmp_path / "pauses.wav"
    samples = np.zeros(24000 * 40, dtype=np.float32)
    samples[24000 * 35 : 24000 * 36] = (
        np.sin(np.arange(24000, dtype=np.float32) * 0.1) * 0.1
    )
    sf.write(path, samples, 24000, subtype="PCM_16")
    assert len(read_audio(path)) == 24000 * 40


def decoded_window(tokens: list[int], confident: bool = True) -> dict[str, Any]:
    scores = []
    for token in tokens:
        logits = torch.zeros(12)
        if confident:
            logits[token] = 8
        scores.append(logits)
    return {
        "idxs": (1, 1 + len(tokens)),
        "result": {"sequences": torch.tensor([10, *tokens]), "scores": scores},
    }


def test_uncertain_later_window_is_not_hidden_by_confident_beginning() -> None:
    result = {"segments": [[decoded_window([1] * 50), decoded_window([2], False)]]}
    with pytest.raises(SpeechUnclear, match="recognition_uncertain"):
        validate_segments(result, eos_token_id=10)


def test_no_speech_decoding_is_not_published() -> None:
    with pytest.raises(SpeechUnclear, match="no_recognized"):
        validate_segments({"segments": [[]]}, eos_token_id=10)


def test_whisper_longform_sees_audio_and_text_after_30_seconds() -> None:
    class Processor:
        def __call__(
            self, audio: npt.NDArray[np.float32], **kwargs: Any
        ) -> SimpleNamespace:
            assert len(audio) == 16000 * 40
            assert kwargs["truncation"] is False
            assert kwargs["padding"] == "longest"
            assert kwargs["return_attention_mask"] is True
            return SimpleNamespace(
                input_features=torch.zeros(1, 80, 4000),
                attention_mask=torch.ones(1, 4000, dtype=torch.long),
            )

        def batch_decode(self, sequences: torch.Tensor, **kwargs: Any) -> list[str]:
            assert sequences.tolist() == [[1, 2]]
            return ["Первая часть. После тридцатой секунды — вторая часть."]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=10)

        def generate(self, features: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
            assert features.shape[-1] == 4000
            assert kwargs["return_segments"] is True
            assert kwargs["return_timestamps"] is True
            assert kwargs["return_dict_in_generate"] is False
            assert kwargs["logprob_threshold"] == -1.0
            assert kwargs["temperature"] == (0.0,)
            configuration = kwargs["generation_config"]
            assert configuration.language == "russian"
            assert configuration.task == "transcribe"
            assert configuration.no_speech_threshold == 0.6
            assert configuration.condition_on_prev_tokens is False
            return {
                "sequences": torch.tensor([[1, 2]]),
                "segments": [[decoded_window([1]), decoded_window([2])]],
            }

    text, windows, _ = decode_audio(
        Model(), Processor(), np.zeros(16000 * 40, dtype=np.float32), "cpu"
    )
    assert text.endswith("вторая часть.")
    assert windows == 2


def test_processor_cannot_silently_truncate_long_audio() -> None:
    class Processor:
        def __call__(self, audio: object, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))

    with pytest.raises(RuntimeError, match="features_truncated"):
        decode_audio(
            object(), Processor(), np.zeros(16000 * 40, dtype=np.float32), "cpu"
        )


def test_internal_suppression_settings_are_preserved_without_duplicate_kwargs() -> None:
    class Model:
        def _prepare_generation_config(
            self, configuration: Any, **kwargs: Any
        ) -> tuple[Any, dict[str, Any]]:
            return configuration, kwargs

    model = Model()
    configuration = SimpleNamespace(suppress_tokens=[5], begin_suppress_tokens=[6])
    configure_longform_compatibility(model)
    prepared, kwargs = model._prepare_generation_config(
        configuration,
        suppress_tokens=None,
        begin_suppress_tokens=[2],
        attention_mask="preserved",
    )
    assert prepared.suppress_tokens == []
    assert prepared.begin_suppress_tokens == [2]
    assert kwargs == {"attention_mask": "preserved"}
    assert configuration.suppress_tokens == [5]


def test_long_transcript_is_rejected_without_truncation() -> None:
    with pytest.raises(TranscriptTooLong):
        validate_text("а" * 2001)
    assert len(validate_text("а" * 2000)) == 2000


def test_private_atomic_output_never_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"
    write_transcript(path, "Привет.")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"text": "Привет.", "language": "ru"}
    with pytest.raises(FileExistsError):
        write_transcript(path, "Замена.")
    assert json.loads(path.read_text())["text"] == "Привет."
    assert not list(tmp_path.glob(".transcript-*"))


def test_invalid_request_does_not_log_private_data(tmp_path: Path) -> None:
    body = request_body(tmp_path) | {"private": "sensitive sentence"}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(body))
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            str(Path(__file__).parents[1] / "run.py"),
            "--request",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2
    assert "sensitive sentence" not in result.stdout + result.stderr
    assert json.loads(result.stdout)["error"] == "asr_invalid"
