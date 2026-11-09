"""Check causal stitching and failure boundaries without loading neural weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from ml.tts.fish_codec import decode_bounded
from ml.tts.fish_runtime import (
    MAX_TOKENS,
    delivery_text,
    generate_codes,
    semantic_budget,
    validate_codes,
)
from ml.tts.request import Delivery, Request
from mlx_audio.tts.models.fish_qwen3_omni.prompt import Conversation


@pytest.mark.parametrize(
    ("delivery", "prefix"),
    [
        ("neutral", ""),
        ("angry", "[angry] "),
        ("happy", "[delight] "),
        ("sad", "[sad] "),
        ("whisper", "[whisper] "),
    ],
)
def test_delivery_preserves_every_spoken_character(
    delivery: Delivery, prefix: str
) -> None:
    text = " Анна, 25 рублей?\nЯ уже звонил. "
    assert delivery_text(text, delivery) == prefix + text


def test_legacy_tag_is_merged_once_and_preserves_case() -> None:
    assert (
        delivery_text("Вот замОк.", "happy", ["delight", "soft tone"])
        == "[delight] [soft tone] Вот замОк."
    )


class RecordingTokenizer:
    def __init__(self) -> None:
        self.received: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.received.append(text)
        return list(range(len(text)))


class CapturingFish:
    def __init__(self, frames: int) -> None:
        self.tokenizer = RecordingTokenizer()
        self.frames = frames
        self.received: dict[str, object] = {}

    def _generate_codes_for_batch(self, **kwargs: object) -> mx.array:
        self.received = kwargs
        return mx.zeros((10, self.frames), dtype=mx.int32)


def test_real_runtime_passes_all_native_controls_without_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ml.tts.fish_runtime as runtime

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "verified.json").write_text('{"revision":"test","files":0}')
    reference = tmp_path / "reference.wav"
    reference.touch()
    request = Request.model_validate_json(
        json.dumps(
            {
                "mode": "tts",
                "text": "Вот замОк.",
                "reference_audio": str(reference),
                "reference_text": "Образец.",
                "output": str(tmp_path / "output.wav"),
                "model_dir": str(tmp_path),
                "delivery": "happy",
                "native_tags": ["delight", "soft tone"],
                "style_instruction": "Спокойно, но с улыбкой.",
                "temperature": 0.4,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 123456,
            }
        )
    )
    captured: dict[str, object] = {}

    class Model(CapturingFish):
        def __init__(self) -> None:
            super().__init__(3)
            self.codec = SimpleNamespace(
                encoder=object(),
                quantizer=SimpleNamespace(pre_module=object(), downsample=object()),
            )

        def _prepare_reference_prompt(
            self, ref_audio: mx.array, ref_text: str
        ) -> tuple[list[str], list[mx.array]]:
            captured["reference_text"] = ref_text
            return [ref_text], [mx.zeros((10, 2), dtype=mx.int32)]

        def _build_conversation(
            self, texts: list[str], tokens: list[mx.array], instruct: str | None = None
        ) -> Conversation:
            captured["instruct"] = instruct
            return Conversation()

    model = Model()
    monkeypatch.setattr(runtime, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(runtime, "validate_reference", lambda path: None)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    for name in ("set_default_device", "set_memory_limit", "set_cache_limit"):
        monkeypatch.setattr(mx, name, lambda value: None)
    for name in ("reset_peak_memory", "clear_cache", "get_peak_memory"):
        monkeypatch.setattr(mx, name, lambda: 0)
    monkeypatch.setattr(mx.random, "seed", lambda value: captured.update(seed=value))
    monkeypatch.setattr(
        runtime.sf, "read", lambda *args, **kwargs: (np.ones(2400, np.float32), 24000)
    )
    monkeypatch.setattr(runtime, "resample", lambda audio, *args: audio)
    monkeypatch.setattr(
        runtime,
        "decode_bounded",
        lambda codec, codes: (np.full(6144, 0.1, np.float32), 1),
    )
    with mx.stream(mx.cpu):
        summary = runtime.generate(request)
    assert captured == {
        "seed": 123456,
        "reference_text": "Образец.",
        "instruct": "Спокойно, но с улыбкой.",
    }
    assert model.received["temperature"] == 0.4
    assert model.received["top_p"] == 1.0
    assert model.received["top_k"] == 0
    assert model.received["batch_text"] == "[delight] [soft tone] Вот замОк."
    assert request.output.exists()
    for field in (
        "model_load_seconds",
        "reference_encode_seconds",
        "generate_codes_seconds",
        "decode_seconds",
        "write_seconds",
    ):
        assert getattr(summary, field) >= 0


@pytest.mark.parametrize("reached_limit", [False, True])
def test_tagged_prompt_drives_model_and_incomplete_output_boundary(
    reached_limit: bool,
) -> None:
    text = delivery_text("Да.", "whisper")
    limit = len(text) * 12
    model = CapturingFish(limit if reached_limit else limit - 1)
    conversation = Conversation()
    with mx.stream(mx.cpu):
        if reached_limit:
            with pytest.raises(RuntimeError, match="generation_token_limit_reached"):
                generate_codes(model, conversation, text)
        else:
            codes = generate_codes(model, conversation, text)
            assert codes.shape[1] == limit - 1
    assert model.tokenizer.received == [text]
    assert model.received["batch_text"] == text
    assert model.received["max_new_tokens"] == MAX_TOKENS
    assert model.received["conversation"] is conversation
    assert conversation.messages[-1].parts[0].text == text


class FakeQuantizer:
    def decode(self, indices: mx.array) -> mx.array:
        frames = indices.shape[-1] * 4
        values = mx.arange(frames, dtype=mx.float32)[None, None, :]
        return mx.broadcast_to(values, (1, 1024, frames))


class CausalDecoder:
    """Nontrivial finite history catches missing overlap or incorrect cropping."""

    def __call__(self, latent: mx.array) -> mx.array:
        values = latent[:, :1, :]
        padded = mx.pad(values, [(0, 0), (0, 0), (19, 0)])
        result = mx.zeros_like(values)
        for delay in range(20):
            start = 19 - delay
            result = result + padded[..., start : start + values.shape[-1]]
        return mx.repeat(result / 20, 512, axis=-1)


class FakeCodec:
    quantizer = FakeQuantizer()
    decoder = CausalDecoder()
    sample_rate = 44100
    frame_length = 2048
    decoder_rates = [8, 8, 4, 2]
    latent_dim = 1024


def test_causal_chunks_preserve_samples_and_boundaries() -> None:
    with mx.stream(mx.cpu):
        codes = mx.zeros((10, 150), dtype=mx.int32)
        codec = FakeCodec()
        expected = np.asarray(codec.decoder(codec.quantizer.decode(codes[None]))[0, 0])
        actual, chunks = decode_bounded(codec, codes)
        assert chunks == 3
        assert len(actual) == 150 * 2048
        np.testing.assert_array_equal(actual, expected)


def test_changed_codec_architecture_is_rejected() -> None:
    codec = FakeCodec()
    codec.frame_length = 1024
    with mx.stream(mx.cpu):
        with pytest.raises(ValueError, match="unsupported_fish_codec_architecture"):
            decode_bounded(codec, mx.zeros((10, 2), dtype=mx.int32))


@pytest.mark.parametrize("text_tokens", [1, 10, 2000])
def test_reaching_either_upstream_budget_is_incomplete(text_tokens: int) -> None:
    budget = semantic_budget(text_tokens)
    with mx.stream(mx.cpu):
        validate_codes(mx.zeros((10, budget - 1), dtype=mx.int32), budget)
        with pytest.raises(RuntimeError, match="generation_token_limit_reached"):
            validate_codes(mx.zeros((10, budget), dtype=mx.int32), budget)


def test_duration_limit_fits_180_seconds_without_rounding_up() -> None:
    assert MAX_TOKENS * 2048 <= 180 * 44100
    assert (MAX_TOKENS + 1) * 2048 > 180 * 44100
