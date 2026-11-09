"""Pinned Fish S2 Pro inference with bounded waveform decoding and no text edits."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from mlx_audio.tts.utils import load_model
from torchaudio.functional import resample

from ml.tts.fish_codec import FishCodec
from ml.tts.fish_codec import decode_bounded as decode_bounded
from ml.tts.request import (
    LEGACY_DELIVERY_TAGS,
    WHISPER_TAGS,
    Delivery,
    Request,
    Summary,
    WhisperContext,
    whisper_fingerprint,
)
from ml.tts.runtime import Verification, validate_reference
from ml.tts.style_context import WhisperUnavailable, require_whisper

SAMPLE_RATE = 44100
FRAME_LENGTH = 2048
MAX_SECONDS = 180
MAX_TOKENS = MAX_SECONDS * SAMPLE_RATE // FRAME_LENGTH
MAX_CONTEXT_TOKENS = 32768
WHISPER_GUIDE_TEXT = (
    "[whisper] Я хочу тебе кое-что сказать. "
    "Подойди поближе и послушай меня внимательно."
)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class FishModel(Protocol):
    codec: FishCodec | None
    tokenizer: Tokenizer | None

    def _prepare_reference_prompt(
        self, ref_audio: mx.array, ref_text: str
    ) -> tuple[list[str], list[mx.array]]: ...

    def _build_conversation(
        self,
        prompt_texts: list[str],
        prompt_tokens: list[mx.array],
        instruct: str | None = None,
    ) -> Conversation: ...

    def _generate_codes_for_batch(
        self,
        *,
        conversation: Conversation,
        batch_text: str,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> mx.array: ...


def semantic_budget(text_token_count: int, max_tokens: int = MAX_TOKENS) -> int:
    """Match the pinned upstream budget; reaching either bound is incomplete."""
    return min(max_tokens, max(32, text_token_count * 12))


def delivery_text(
    text: str, delivery: Delivery, native_tags: list[str] | None = None
) -> str:
    """Prefix Fish's native instruction; keep the spoken text unchanged."""
    tags = dict.fromkeys((*LEGACY_DELIVERY_TAGS[delivery], *(native_tags or [])))
    prefix = " ".join(f"[{tag}]" for tag in tags)
    return f"{prefix} {text}" if prefix else text


def whisper_target_text(request: Request) -> str:
    remaining = [
        tag
        for tag in request.effective_native_tags
        if tag.strip().casefold() not in WHISPER_TAGS
    ]
    text = delivery_text(request.text, "neutral", remaining)
    return "<|speaker:0|>[whisper in small voice] " + text


def append_whisper_history(conversation: Conversation, context: WhisperContext) -> None:
    # This is the exact native multi-turn format approved by listening.
    # Context text is already the original user turn; do not edit its cues.
    conversation.append(
        Message(
            role="user",
            parts=[TextPart(context.text)],
            add_im_start=True,
            add_im_end=True,
        )
    )
    conversation.append(
        Message(
            role="assistant",
            parts=[VQPart(mx.array(context.codes, dtype=mx.int32))],
            modality="voice",
            add_im_start=True,
            add_im_end=True,
        )
    )


def write_whisper_context(path: Path, context: WhisperContext) -> None:
    if path.is_symlink():
        raise ValueError("whisper_context_symlink")
    descriptor, name = tempfile.mkstemp(prefix=".whisper-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(context.model_dump_json().encode())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_whisper_context(
    model: FishModel,
    prompt_texts: list[str],
    prompt_tokens: list[mx.array],
    fingerprint: str,
    *,
    check_context: bool = False,
) -> WhisperContext:
    """One fixed native guide attempt; no seed search or ordinary-voice fallback."""
    if model.codec is None:
        raise RuntimeError("fish_codec_missing")
    mx.random.seed(42)
    conversation = model._build_conversation(prompt_texts, prompt_tokens, instruct=None)
    guide = generate_codes(
        model,
        conversation,
        WHISPER_GUIDE_TEXT,
        temperature=0.7,
        top_p=0.7,
        top_k=30,
        check_context=check_context,
    )
    if guide.shape[1] > 256:
        raise WhisperUnavailable("whisper_guide_too_long")
    audio, _ = decode_bounded(model.codec, guide)
    require_whisper(audio, SAMPLE_RATE)
    return WhisperContext(
        fingerprint=fingerprint,
        text=WHISPER_GUIDE_TEXT,
        codes=np.asarray(guide, dtype=np.int32).tolist(),
    )


def generate_codes(
    model: FishModel,
    conversation: Conversation,
    text: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 30,
    check_context: bool = False,
) -> mx.array:
    if model.tokenizer is None:
        raise RuntimeError("fish_tokenizer_missing")
    conversation.append(
        Message(
            role="user",
            parts=[TextPart(text)],
            add_im_start=True,
            add_im_end=True,
        )
    )
    budget = semantic_budget(len(model.tokenizer.encode(text)))
    if check_context:
        prompt = Conversation(list(conversation.messages))
        prompt.append(
            Message(
                role="assistant",
                parts=[],
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )
        encoded = prompt.encode_for_inference(model.tokenizer, num_codebooks=10)
        if encoded.shape[1] + budget > MAX_CONTEXT_TOKENS:
            raise ValueError("reference_context_limit_exceeded")
    codes = model._generate_codes_for_batch(
        conversation=conversation,
        batch_text=text,
        max_new_tokens=MAX_TOKENS,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    mx.eval(codes)
    validate_codes(codes, budget)
    return codes


def validate_codes(codes: mx.array, budget: int) -> None:
    if codes.ndim != 2 or codes.shape[0] != 10 or codes.shape[1] == 0:
        raise RuntimeError("invalid_acoustic_codes")
    if codes.shape[1] >= budget:
        raise RuntimeError("generation_token_limit_reached")
    if codes.shape[1] * FRAME_LENGTH > MAX_SECONDS * SAMPLE_RATE:
        raise RuntimeError("output_duration_limit")


def validate_additional_reference(path: Path) -> None:
    validate_reference(path)
    if sf.info(path).duration < 5:
        raise ValueError("additional_reference_requires_5_to_20_seconds")


def generate(request: Request) -> Summary:
    started = time.perf_counter()
    if request.mode != "tts":
        raise ValueError("fish_requires_tts_mode")
    if request.reference_audio is None or request.reference_text is None:
        raise ValueError("reference_and_transcript_required")
    if not mx.metal.is_available():
        raise RuntimeError("mlx_metal_unavailable")
    validate_reference(request.reference_audio)
    for extra in request.extra_references:
        validate_additional_reference(extra.reference_audio)
    verification = Verification.model_validate_json(
        (request.model_dir / "verified.json").read_text()
    )
    context = request.whisper_context if request.uses_whisper else None
    fingerprint: str | None = None
    if request.uses_whisper:
        fingerprint = whisper_fingerprint(
            reference_sha256=hashlib.sha256(
                request.reference_audio.read_bytes()
            ).hexdigest(),
            reference_text=request.reference_text,
            model_revision=verification.revision,
            extra_references=[
                (
                    hashlib.sha256(extra.reference_audio.read_bytes()).hexdigest(),
                    extra.reference_text,
                )
                for extra in request.extra_references
            ],
        )
        if context is not None and context.fingerprint != fingerprint:
            raise ValueError("whisper_context_fingerprint_mismatch")
    mx.set_default_device(mx.gpu)
    config: object = json.loads((request.model_dir / "config.json").read_text())
    quantized = isinstance(config, dict) and bool(config.get("quantization"))
    mx.set_memory_limit((14 if quantized else 17) * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    mx.reset_peak_memory()
    phase_started = time.perf_counter()
    model = cast(FishModel, load_model(request.model_dir, strict=True))
    model_load_seconds = time.perf_counter() - phase_started
    if model.codec is None or model.tokenizer is None:
        raise RuntimeError("fish_codec_or_tokenizer_missing")
    phase_started = time.perf_counter()
    reference, reference_rate = sf.read(request.reference_audio, dtype="float32")
    reference_audio = mx.array(
        resample(torch.from_numpy(reference), reference_rate, SAMPLE_RATE).numpy()
    )
    mx.random.seed(request.seed)
    generated_at = time.perf_counter()
    prompt_texts, prompt_tokens = model._prepare_reference_prompt(
        reference_audio, request.reference_text
    )
    if request.extra_references:
        # Separate utterances of the SAME speaker: stock implicit indexing would
        # assign every clip a different speaker. Keep stored ASR text untouched.
        mx.eval(*prompt_tokens)
        del reference_audio, reference
        mx.clear_cache()
        for extra in request.extra_references:
            samples, sample_rate = sf.read(extra.reference_audio, dtype="float32")
            extra_audio = mx.array(
                resample(torch.from_numpy(samples), sample_rate, SAMPLE_RATE).numpy()
            )
            texts, tokens = model._prepare_reference_prompt(
                extra_audio, extra.reference_text
            )
            mx.eval(*tokens)
            prompt_texts.extend(texts)
            prompt_tokens.extend(tokens)
            del samples, extra_audio
            mx.clear_cache()
        prompt_texts = ["<|speaker:0|>" + text for text in prompt_texts]
    # Only decoding follows in this one-request process. Materialize reference
    # codes first, then release the encoder and training-only analysis blocks.
    mx.eval(*prompt_tokens)
    delattr(model.codec, "encoder")
    delattr(model.codec.quantizer, "pre_module")
    delattr(model.codec.quantizer, "downsample")
    mx.clear_cache()
    reference_encode_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    if request.uses_whisper:
        assert fingerprint is not None
        if context is None:
            context = prepare_whisper_context(
                model,
                prompt_texts,
                prompt_tokens,
                fingerprint,
                check_context=bool(request.extra_references),
            )
        write_whisper_context(request.output.parent / "whisper-context.json", context)
        # Guide sampling must not advance the requested target seed or KV state.
        mx.random.seed(request.seed)
    conversation = model._build_conversation(
        prompt_texts, prompt_tokens, instruct=request.style_instruction
    )
    if context is not None:
        append_whisper_history(conversation, context)
    text = (
        whisper_target_text(request)
        if request.uses_whisper
        else delivery_text(request.text, request.delivery, request.native_tags)
    )
    codes = generate_codes(
        model,
        conversation,
        text,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        check_context=bool(request.extra_references),
    )
    generate_codes_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    audio, blocks = decode_bounded(model.codec, codes)
    decode_seconds = time.perf_counter() - phase_started
    expected_samples = codes.shape[1] * FRAME_LENGTH
    if len(audio) != expected_samples or not np.isfinite(audio).all():
        raise RuntimeError("invalid_decoded_audio")
    if len(audio) < SAMPLE_RATE // 10 or float(np.max(np.abs(audio))) < 0.01:
        raise RuntimeError("empty_or_silent_output")
    if request.uses_whisper:
        require_whisper(audio, SAMPLE_RATE)
    phase_started = time.perf_counter()
    write_audio(request.output, audio)
    write_seconds = time.perf_counter() - phase_started
    return Summary(
        mode="tts",
        duration_seconds=len(audio) / SAMPLE_RATE,
        elapsed_seconds=time.perf_counter() - started,
        generation_seconds=time.perf_counter() - generated_at,
        sample_rate=SAMPLE_RATE,
        peak_mlx_bytes=mx.get_peak_memory(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_revision=verification.revision,
        decoder_mode="bounded_causal",
        decoder_blocks=blocks,
        acoustic_tokens=int(codes.shape[1]),
        model_load_seconds=model_load_seconds,
        reference_encode_seconds=reference_encode_seconds,
        generate_codes_seconds=generate_codes_seconds,
        decode_seconds=decode_seconds,
        write_seconds=write_seconds,
    )


def write_audio(path: Path, audio: npt.NDArray[np.float32]) -> None:
    temporary = path.with_name(path.name + ".partial")
    try:
        sf.write(temporary, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        temporary.chmod(0o600)
        if sf.info(temporary).frames != len(audio):
            raise RuntimeError("output_frame_mismatch")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
