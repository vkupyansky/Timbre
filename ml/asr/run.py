"""Bounded offline Russian Whisper transcription, including audio beyond 30 s.

Only the private output JSON contains words. Exit codes: 2 invalid request/audio,
3 no confident speech, 4 text too long for TTS, 75 memory exhaustion, 1 failure.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import soundfile as sf
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

MAX_TEXT_CHARACTERS = 2000
MIN_SEGMENT_LOG_PROBABILITY = -1.2
ROOT = Path(__file__).resolve().parents[2]


class ASRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_audio: Path
    output: Path
    model_dir: Path

    @model_validator(mode="after")
    def local_paths(self) -> Self:
        if any(
            not path.is_absolute()
            for path in (self.source_audio, self.output, self.model_dir)
        ):
            raise ValueError("absolute_local_paths_required")
        if not self.source_audio.is_file() or not self.model_dir.is_dir():
            raise ValueError("source_or_model_missing")
        if self.output.suffix != ".json" or not self.output.parent.is_dir():
            raise ValueError("existing_output_parent_and_json_required")
        if self.output.exists() or self.output.is_symlink():
            raise ValueError("output_already_exists")
        if self.output.resolve() == self.source_audio.resolve():
            raise ValueError("output_cannot_replace_source")
        return self


class SpeechUnclear(ValueError):
    """Safe enum only: no speech, or insufficient recognition confidence."""


class TranscriptTooLong(ValueError):
    """The complete recognized text exceeds the supported TTS input size."""


def read_audio(path: Path) -> npt.NDArray[np.float32]:
    if path.stat().st_size > 6 * 1024 * 1024:
        raise ValueError("audio_too_large")
    info = sf.info(path)
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.channels != 1
        or info.samplerate != 24000
        or not 0.1 <= info.duration <= 120
    ):
        raise ValueError("mono_24khz_pcm16_wav_0_1_to_120_seconds_required")
    audio, _ = sf.read(path, dtype="float32")
    if not np.isfinite(audio).all():
        raise ValueError("invalid_audio_samples")
    # Unlike reference enrollment, a one-word message and long pauses are valid.
    # Reject near silence before Whisper can hallucinate a plausible sentence.
    frames = audio[: len(audio) // 480 * 480].reshape(-1, 480)
    active_frames = np.sqrt(np.mean(frames**2, axis=1)) >= 0.003
    if np.count_nonzero(active_frames) < 3:
        raise SpeechUnclear("no_audible_speech")
    return audio


def verify_marker(model_dir: Path) -> str:
    manifest_bytes = (ROOT / "models/asr.lock.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    marker = json.loads((model_dir / "verified.json").read_text())
    if marker != {
        "revision": manifest["revision"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }:
        raise RuntimeError("asr_verification_marker_mismatch")
    return str(manifest["revision"])


def validate_text(text: str) -> str:
    # Preserve punctuation and wording; only collapse whitespace from windows.
    text = " ".join(text.split())
    if not text or not any(character.isalnum() for character in text):
        raise SpeechUnclear("no_recognized_speech")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise TranscriptTooLong("recognized_text_too_long")
    return text


def validate_segments(result: dict[str, Any], eos_token_id: int) -> tuple[int, float]:
    import torch

    batches = result.get("segments")
    if not isinstance(batches, list) or len(batches) != 1:
        raise RuntimeError("invalid_asr_segments")
    if not batches[0]:
        raise SpeechUnclear("no_recognized_speech")
    probabilities: list[float] = []
    windows: set[int] = set()
    for segment in batches[0]:
        generated = segment["result"]
        windows.add(id(generated))
        sequence = generated["sequences"]
        scores = generated["scores"]
        # HF idxs include the decoder prompt; scores cover generated tokens only.
        prompt_length = len(sequence) - len(scores)
        start, end = segment["idxs"]
        if not 0 <= prompt_length <= start < end <= len(sequence):
            raise RuntimeError("invalid_asr_score_alignment")
        segment_probabilities: list[float] = []
        for position in range(start, end):
            token = int(sequence[position])
            if token >= eos_token_id:
                continue  # language, task, timestamp and EOS are not spoken words
            logits = scores[position - prompt_length].float().cpu()
            probability = float(torch.log_softmax(logits, dim=-1)[token])
            segment_probabilities.append(probability)
        if not segment_probabilities:
            continue
        confidence = sum(segment_probabilities) / len(segment_probabilities)
        if not math.isfinite(confidence) or confidence < MIN_SEGMENT_LOG_PROBABILITY:
            raise SpeechUnclear("recognition_uncertain")
        probabilities.extend(segment_probabilities)
    if not probabilities:
        raise SpeechUnclear("no_recognized_speech")
    return len(windows), sum(probabilities) / len(probabilities)


def configure_longform_compatibility(model: Any) -> None:
    """Keep HF 5.16's internal suppression arguments in GenerationConfig.

    Whisper's long-form loop clears these fields after building its processors,
    then passes them twice to GenerationMixin. Empty lists preserve that cleared
    state through defaults-only config merging. No warnings are filtered.
    """
    original_prepare = model._prepare_generation_config

    def prepare(configuration: Any, **kwargs: Any) -> Any:
        if configuration is not None and any(
            key in kwargs for key in ("suppress_tokens", "begin_suppress_tokens")
        ):
            configuration = deepcopy(configuration)
            for key in ("suppress_tokens", "begin_suppress_tokens"):
                if key in kwargs:
                    value = kwargs.pop(key)
                    setattr(configuration, key, [] if value is None else value)
        return original_prepare(configuration, **kwargs)

    model._prepare_generation_config = prepare


def decode_audio(
    model: Any,
    processor: Any,
    waveform_16k: npt.NDArray[np.float32],
    device: str,
) -> tuple[str, int, float]:
    """Use the installed HF Whisper sequential timestamp-based long-form path."""
    import torch

    inputs = processor(
        waveform_16k,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=False,
        padding="longest",
    )
    # Catch an incompatible processor before silently accepting only 30 seconds.
    expected_frames = len(waveform_16k) // 160
    if inputs.input_features.shape[-1] < expected_frames:
        raise RuntimeError("asr_features_truncated")
    generation = deepcopy(model.generation_config)
    generation.language = "russian"
    generation.task = "transcribe"
    generation.return_dict_in_generate = True
    generation.output_scores = True
    generation.condition_on_prev_tokens = False
    generation.no_speech_threshold = 0.6
    generation.logprob_threshold = -1.0
    generation.compression_ratio_threshold = 2.4
    with torch.inference_mode():
        result = model.generate(
            inputs.input_features.to(device),
            attention_mask=inputs.attention_mask.to(device),
            generation_config=generation,
            return_timestamps=True,
            return_segments=True,
            # Request segment results explicitly. The log-probability check
            # enables score dictionaries for individual decoder windows.
            return_dict_in_generate=False,
            logprob_threshold=-1.0,
            temperature=(0.0,),
        )
    windows, confidence = validate_segments(result, generation.eos_token_id)
    text = processor.batch_decode(
        result["sequences"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return validate_text(text), windows, confidence


def write_transcript(output: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".transcript-", dir=output.parent)
    partial = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"text": text, "language": "ru"}, stream, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without replacing an existing file, even in a race.
        os.link(partial, output)
    finally:
        partial.unlink(missing_ok=True)


def transcribe(request: ASRRequest) -> dict[str, str | int | float]:
    started = time.perf_counter()
    audio = read_audio(request.source_audio)
    revision = verify_marker(request.model_dir)
    import torch
    from torchaudio.functional import resample
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not torch.backends.mps.is_available():
        raise RuntimeError("mps_unavailable")
    processor = WhisperProcessor.from_pretrained(
        request.model_dir, local_files_only=True, clean_up_tokenization_spaces=False
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        request.model_dir, local_files_only=True
    ).to("mps")
    model.eval()
    configure_longform_compatibility(model)
    torch.manual_seed(0)
    waveform = resample(torch.from_numpy(audio), 24000, 16000).numpy()
    text, windows, confidence = decode_audio(model, processor, waveform, "mps")
    write_transcript(request.output, text)
    return {
        "mode": "asr",
        "backend": "mps",
        "language": "ru",
        "model": "openai/whisper-small",
        "revision": revision,
        "duration_seconds": len(audio) / 24000,
        "decoding_windows": windows,
        "text_characters": len(text),
        "mean_token_log_probability": confidence,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    stage = "request"
    try:
        if args.request.stat().st_size > 16 * 1024:
            raise ValueError("request_too_large")
        request = ASRRequest.model_validate_json(args.request.read_text())
        stage = "transcription"
        with contextlib.redirect_stdout(sys.stderr):
            metrics = transcribe(request)
        print(json.dumps(metrics))
        return 0
    except SpeechUnclear as error:
        print(json.dumps({"error": "speech_unclear", "reason": str(error)}))
        return 3
    except TranscriptTooLong:
        print(json.dumps({"error": "transcript_too_long"}))
        return 4
    except (ValidationError, ValueError, OSError) as error:
        print(json.dumps({"error": "asr_invalid", "type": type(error).__name__}))
        return 2
    except Exception as error:
        print(
            json.dumps(
                {"error": "asr_failed", "stage": stage, "type": type(error).__name__}
            )
        )
        if isinstance(error, MemoryError) or "out of memory" in str(error).lower():
            return 75
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
