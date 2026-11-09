"""Native context integration exercised with CPU tokens, without model weights."""

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from ml.tts import fish_runtime as runtime
from ml.tts.request import Request, WhisperContext, whisper_fingerprint
from ml.tts.style_context import WhisperUnavailable
from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
    Conversation,
    Message,
    TextPart,
    VQPart,
)


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class RecordingFish:
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()
        self.codec = SimpleNamespace(
            encoder=object(),
            quantizer=SimpleNamespace(pre_module=object(), downsample=object()),
        )
        self.calls: list[dict[str, object]] = []
        self.instructions: list[str | None] = []
        self.reference_texts: list[str] = []

    def _prepare_reference_prompt(
        self, audio: mx.array, text: str
    ) -> tuple[list[str], list[mx.array]]:
        self.reference_texts.append(text)
        return [text], [mx.zeros((10, 2), dtype=mx.int32)]

    def _build_conversation(
        self,
        texts: list[str],
        tokens: list[mx.array],
        instruct: str | None = None,
    ) -> Conversation:
        self.instructions.append(instruct)
        return Conversation([Message(role="system", parts=[TextPart(texts[0])])])

    def _generate_codes_for_batch(
        self,
        *,
        conversation: Conversation,
        batch_text: str,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> mx.array:
        self.calls.append(
            {
                "messages": list(conversation.messages),
                "text": batch_text,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
            }
        )
        return mx.full((10, 8), len(self.calls), dtype=mx.int32)


def case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> tuple[Request, RecordingFish, list[int], list[int]]:
    (tmp_path / "reference.wav").touch()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "verified.json").write_text('{"revision":"model-revision","files":0}')
    fingerprint = whisper_fingerprint(
        reference_sha256=hashlib.sha256(b"").hexdigest(),
        reference_text="Референс выбранного голоса.",
        model_revision="model-revision",
    )
    context = WhisperContext(
        fingerprint=fingerprint,
        text="[whisper] Предыдущая короткая реплика.",
        codes=[[7] * 6 for _ in range(10)],
    )
    body = {
        "mode": "tts",
        "text": "  Вот замОк. И ещё одна фраза. ",
        "reference_audio": str(tmp_path / "reference.wav"),
        "reference_text": "Референс выбранного голоса.",
        "output": str(tmp_path / "output.wav"),
        "model_dir": str(tmp_path),
        "native_tags": ["whisper"],
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 11,
        "seed": 99,
        "style_instruction": "С мягким акцентом.",
        "whisper_context": context.model_dump() if cached else None,
    }
    request = Request.model_validate_json(json.dumps(body))
    model = RecordingFish()
    seeds: list[int] = []
    gates: list[int] = []
    monkeypatch.setattr(runtime, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(runtime, "validate_reference", lambda path: None)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    for name in ("set_default_device", "set_memory_limit", "set_cache_limit"):
        monkeypatch.setattr(mx, name, lambda value: None)
    for name in ("reset_peak_memory", "clear_cache", "get_peak_memory"):
        monkeypatch.setattr(mx, name, lambda: 0)
    monkeypatch.setattr(mx.random, "seed", seeds.append)
    monkeypatch.setattr(
        runtime.sf, "read", lambda *args, **kwargs: (np.ones(2400, np.float32), 24000)
    )
    monkeypatch.setattr(runtime, "resample", lambda audio, *args: audio)
    monkeypatch.setattr(
        runtime,
        "decode_bounded",
        lambda codec, codes: (
            np.full(codes.shape[1] * runtime.FRAME_LENGTH, 0.05, np.float32),
            1,
        ),
    )
    monkeypatch.setattr(
        runtime, "require_whisper", lambda audio, rate: gates.append(len(audio))
    )
    return request, model, seeds, gates


def messages(call: dict[str, object]) -> list[Message]:
    value = call["messages"]
    assert isinstance(value, list)
    assert all(isinstance(message, Message) for message in value)
    return value


def test_cached_context_reuses_exact_history_and_keeps_original_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, model, seeds, gates = case(tmp_path, monkeypatch, cached=True)
    with mx.stream(mx.cpu):
        runtime.generate(request)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["text"] == "<|speaker:0|>[whisper in small voice] " + request.text
    turns = messages(call)
    assert [message.role for message in turns] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    reference = turns[0].parts[0]
    previous_user = turns[1].parts[0]
    previous_audio = turns[2].parts[0]
    assert isinstance(reference, TextPart) and reference.text == request.reference_text
    assert isinstance(previous_user, TextPart)
    assert previous_user.text == "[whisper] Предыдущая короткая реплика."
    assert isinstance(previous_audio, VQPart)
    np.testing.assert_array_equal(np.asarray(previous_audio.codes), np.full((10, 6), 7))
    assert (
        turns[2].modality == "voice" and turns[2].add_im_end and turns[2].add_im_start
    )
    assert seeds == [99, 99]
    assert model.instructions == [request.style_instruction]
    assert model.reference_texts == [request.reference_text]
    assert (call["temperature"], call["top_p"], call["top_k"]) == (0.4, 0.9, 11)
    assert len(gates) == 1
    exported = tmp_path / "whisper-context.json"
    assert (
        WhisperContext.model_validate_json(exported.read_text())
        == request.whisper_context
    )
    assert stat.S_IMODE(exported.stat().st_mode) == 0o600
    assert request.output.exists()


def test_cache_miss_prepares_once_then_resets_target_seed_and_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, model, seeds, gates = case(tmp_path, monkeypatch, cached=False)
    with mx.stream(mx.cpu):
        runtime.generate(request)
    assert len(model.calls) == 2
    guide, target = model.calls
    assert guide["text"] == runtime.WHISPER_GUIDE_TEXT
    assert (guide["temperature"], guide["top_p"], guide["top_k"]) == (0.7, 0.7, 30)
    assert len(messages(guide)) == 2
    assert len(messages(target)) == 4
    assert model.instructions == [None, request.style_instruction]
    assert seeds == [99, 42, 99]
    assert len(gates) == 2
    context = WhisperContext.model_validate_json(
        (tmp_path / "whisper-context.json").read_text()
    )
    assert context.text == runtime.WHISPER_GUIDE_TEXT
    assert context.codes == [[1] * 8 for _ in range(10)]
    assert target["text"] == "<|speaker:0|>[whisper in small voice] " + request.text


def test_voiced_guide_fails_once_without_publishing_context_or_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, model, _, _ = case(tmp_path, monkeypatch, cached=False)

    def reject(audio: object, sample_rate: int) -> None:
        raise WhisperUnavailable("whisper_context_voiced")

    monkeypatch.setattr(runtime, "require_whisper", reject)
    with mx.stream(mx.cpu), pytest.raises(WhisperUnavailable):
        runtime.generate(request)
    assert len(model.calls) == 1
    assert not request.output.exists()
    assert not (tmp_path / "whisper-context.json").exists()


def test_voiced_target_is_not_written_as_a_successful_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, model, _, _ = case(tmp_path, monkeypatch, cached=True)

    def reject(audio: object, sample_rate: int) -> None:
        raise WhisperUnavailable("whisper_context_voiced")

    monkeypatch.setattr(runtime, "require_whisper", reject)
    with mx.stream(mx.cpu), pytest.raises(WhisperUnavailable):
        runtime.generate(request)
    assert len(model.calls) == 1
    assert not request.output.exists()
    assert (tmp_path / "whisper-context.json").exists()


def test_context_for_a_different_reference_fails_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, model, seeds, _ = case(tmp_path, monkeypatch, cached=True)
    assert request.whisper_context is not None
    request.whisper_context.fingerprint = "f" * 64
    with pytest.raises(ValueError, match="whisper_context_fingerprint_mismatch"):
        runtime.generate(request)
    assert not model.calls and not seeds
    assert not request.output.exists()


def test_nonwhisper_ignores_context_and_preserves_existing_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, model, seeds, gates = case(tmp_path, monkeypatch, cached=True)
    request.native_tags = ["happy", "soft tone"]
    with mx.stream(mx.cpu):
        runtime.generate(request)
    assert len(model.calls) == 1
    assert len(messages(model.calls[0])) == 2
    assert model.calls[0]["text"] == "[happy] [soft tone] " + request.text
    assert seeds == [99] and not gates
    assert not (tmp_path / "whisper-context.json").exists()


def test_whisper_target_normalizes_aliases_and_keeps_other_native_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _, _, _ = case(tmp_path, monkeypatch, cached=True)
    request.native_tags = ["WHISPERING", "soft tone", "sad"]
    assert request.uses_whisper
    assert runtime.whisper_target_text(request) == (
        "<|speaker:0|>[whisper in small voice] [soft tone] [sad] " + request.text
    )
