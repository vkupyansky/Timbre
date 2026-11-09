"""Native same-speaker prompts with CPU arrays; no model weights or GPU."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from ml.tts import fish_runtime as runtime
from ml.tts.fish_runtime import validate_additional_reference
from ml.tts.request import Request, whisper_fingerprint
from ml.tts.tests.test_whisper_runtime import RecordingFish, Tokenizer, case, messages
from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model
from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from pydantic import ValidationError


class NativeTokenizer(Tokenizer):
    semantic_begin_id = 1000


class MultipleFish(RecordingFish):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = NativeTokenizer()
        self.prepared: list[list[str]] = []

    def _prepare_reference_prompt(
        self, audio: mx.array, text: str
    ) -> tuple[list[str], list[mx.array]]:
        assert hasattr(self.codec, "encoder")
        self.reference_texts.append(text)
        return [text], [mx.full((10, 2), len(self.reference_texts), dtype=mx.int32)]

    def _build_conversation(
        self,
        texts: list[str],
        tokens: list[mx.array],
        instruct: str | None = None,
    ) -> Conversation:
        self.prepared.append(texts.copy())
        self.instructions.append(instruct)
        return Model._build_conversation(self, texts, tokens, instruct=instruct)


def add_references(request: Request, count: int) -> Request:
    payload = request.model_dump(mode="json")
    extras = []
    for index in range(count):
        path = request.output.parent / f"sample-{index}.wav"
        path.write_bytes(f"sample-{index}".encode())
        extras.append(
            {"reference_audio": str(path), "reference_text": f"Образец {index}."}
        )
    payload["extra_references"] = extras
    return Request.model_validate_json(json.dumps(payload))


@pytest.fixture(autouse=True)
def avoid_reading_placeholder_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "validate_additional_reference", lambda path: None)


@pytest.mark.parametrize("count", [1, 2])
def test_all_clips_use_one_native_speaker_and_separate_encoder_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    request, _, seeds, _ = case(tmp_path, monkeypatch, cached=False)
    request = add_references(request, count).model_copy(update={"native_tags": []})
    model = MultipleFish()
    monkeypatch.setattr(runtime, "load_model", lambda *args, **kwargs: model)
    validated = []
    monkeypatch.setattr(runtime, "validate_reference", validated.append)
    monkeypatch.setattr(runtime, "validate_additional_reference", validated.append)
    with mx.stream(mx.cpu):
        runtime.generate(request)
    raw_texts = [request.reference_text] + [
        sample.reference_text for sample in request.extra_references
    ]
    assert model.reference_texts == raw_texts
    assert model.prepared == [["<|speaker:0|>" + text for text in raw_texts]]
    assert seeds == [request.seed]
    assert validated == [request.reference_audio] + [
        sample.reference_audio for sample in request.extra_references
    ]
    assert not hasattr(model.codec, "encoder")
    system = messages(model.calls[0])[0]
    spoken_refs = system.parts[1]
    assert isinstance(spoken_refs, TextPart)
    assert spoken_refs.text.count("<|speaker:0|>") == count + 1
    assert "<|speaker:1|>" not in spoken_refs.text
    native_audio = system.parts[3]
    assert isinstance(native_audio, VQPart)
    expected = np.concatenate(
        [np.full((10, 2), index + 1) for index in range(count + 1)], axis=1
    )
    np.testing.assert_array_equal(np.asarray(native_audio.codes), expected)
    assert not (tmp_path / "whisper-context.json").exists()


def test_multiple_reference_whisper_keeps_history_and_binds_every_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _, seeds, _ = case(tmp_path, monkeypatch, cached=True)
    request = add_references(request, 2)
    assert request.whisper_context is not None
    fingerprint = whisper_fingerprint(
        reference_sha256=hashlib.sha256(
            request.reference_audio.read_bytes()
        ).hexdigest(),
        reference_text=request.reference_text,
        model_revision="model-revision",
        extra_references=[
            (
                hashlib.sha256(sample.reference_audio.read_bytes()).hexdigest(),
                sample.reference_text,
            )
            for sample in request.extra_references
        ],
    )
    request = request.model_copy(
        update={
            "whisper_context": request.whisper_context.model_copy(
                update={"fingerprint": fingerprint}
            )
        }
    )
    model = MultipleFish()
    monkeypatch.setattr(runtime, "load_model", lambda *args, **kwargs: model)
    with mx.stream(mx.cpu):
        runtime.generate(request)
    assert len(model.calls) == 1
    assert len(messages(model.calls[0])) == 4
    assert seeds == [request.seed, request.seed]
    assert (
        request.whisper_context.fingerprint
        == json.loads((tmp_path / "whisper-context.json").read_text())["fingerprint"]
    )


def test_context_limit_rejects_before_native_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _, _, _ = case(tmp_path, monkeypatch, cached=False)
    request = add_references(request, 2).model_copy(update={"native_tags": []})
    model = MultipleFish()
    monkeypatch.setattr(runtime, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(runtime, "MAX_CONTEXT_TOKENS", 20)
    with mx.stream(mx.cpu), pytest.raises(ValueError, match="reference_context_limit"):
        runtime.generate(request)
    assert not model.calls and not request.output.exists()


@pytest.mark.parametrize("remaining", [0, -1])
def test_three_maximum_clips_reserve_full_180_second_output_budget(
    monkeypatch: pytest.MonkeyPatch, remaining: int
) -> None:
    model = MultipleFish()
    text = "Русский текст. " * 100
    assert (
        runtime.semantic_budget(len(model.tokenizer.encode(text))) == runtime.MAX_TOKENS
    )
    with mx.stream(mx.cpu):
        codes = [mx.zeros((10, 431), dtype=mx.int32) for _ in range(3)]
        conversation = model._build_conversation(
            [f"<|speaker:0|>Образец {index}." for index in range(3)], codes
        )
        full_prompt = Conversation(list(conversation.messages))
        full_prompt.append(Message(role="user", parts=[TextPart(text)]))
        full_prompt.append(
            Message(role="assistant", parts=[], modality="voice", add_im_end=False)
        )
        positions = full_prompt.encode_for_inference(
            model.tokenizer, num_codebooks=10
        ).shape[1]
        assert positions + runtime.MAX_TOKENS < 32768
        monkeypatch.setattr(
            runtime, "MAX_CONTEXT_TOKENS", positions + runtime.MAX_TOKENS + remaining
        )
        if remaining < 0:
            with pytest.raises(ValueError, match="reference_context_limit"):
                runtime.generate_codes(model, conversation, text, check_context=True)
            assert not model.calls
        else:
            runtime.generate_codes(model, conversation, text, check_context=True)
            assert len(model.calls) == 1


@pytest.mark.parametrize(
    "change", ["too_many", "relative", "missing", "blank", "too_long", "unexpected"]
)
def test_extra_reference_contract_rejects_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    request, _, _, _ = case(tmp_path, monkeypatch, cached=False)
    payload = add_references(request, 1).model_dump(mode="json")
    sample = payload["extra_references"][0]
    if change == "too_many":
        payload["extra_references"] *= 3
    elif change == "relative":
        sample["reference_audio"] = "sample.wav"
    elif change == "missing":
        sample["reference_audio"] = str(tmp_path / "missing.wav")
    elif change == "blank":
        sample["reference_text"] = " "
    elif change == "too_long":
        sample["reference_text"] = "а" * 2001
    else:
        sample["instruction"] = "invented"
    with pytest.raises(ValidationError):
        Request.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("duration", [3, 4.99, 5, 20])
def test_additional_reference_minimum_keeps_legacy_primary_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: float
) -> None:
    calls = []
    monkeypatch.setattr(runtime, "validate_reference", calls.append)
    monkeypatch.setattr(
        runtime.sf, "info", lambda path: SimpleNamespace(duration=duration)
    )
    path = tmp_path / "reference.wav"
    if duration < 5:
        with pytest.raises(ValueError, match="5_to_20"):
            validate_additional_reference(path)
    else:
        validate_additional_reference(path)
    assert calls == [path]
