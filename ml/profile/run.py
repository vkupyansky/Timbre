"""Validate and transcribe a Russian reference for the shared Fish voice profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
import soundfile as sf
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["profile"]
    source_audio: Path
    output_dir: Path
    model_dir: Path

    @model_validator(mode="after")
    def local_paths(self) -> Self:
        for path in (self.source_audio, self.output_dir, self.model_dir):
            if not path.is_absolute():
                raise ValueError("absolute_paths_required")
        if not self.source_audio.is_file() or not self.model_dir.is_dir():
            raise ValueError("source_or_model_missing")
        if not self.output_dir.is_dir():
            raise ValueError("output_directory_missing")
        if any(
            (self.output_dir / name).exists()
            for name in ("reference.wav", "transcript.json")
        ):
            raise ValueError("profile_output_already_exists")
        return self


class ProfileInvalid(ValueError):
    """Safe, enumerated quality reason, never includes private transcript."""


class AudioQuality(BaseModel):
    duration_seconds: float
    sample_rate: int
    rms_dbfs: float
    clipped_fraction: float
    silence_fraction: float


class GenerationConfig(BaseModel):
    lang_to_id: dict[str, int]


class Marker(BaseModel):
    revision: str
    manifest_sha256: str


def measure_audio(path: Path) -> tuple[npt.NDArray[np.float32], AudioQuality]:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ProfileInvalid("reference_too_large")
    info = sf.info(path)
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.channels != 1
        or info.samplerate not in (24000, 44100)
    ):
        raise ProfileInvalid("normalized_mono_24_or_44khz_pcm_wav_required")
    if not 15 <= info.duration <= 20:
        raise ProfileInvalid("reference_requires_15_to_20_seconds")
    audio, _ = sf.read(path, dtype="float32")
    if not np.isfinite(audio).all():
        raise ProfileInvalid("invalid_audio_samples")
    if float(np.max(np.abs(audio))) > 2:
        raise ProfileInvalid("reference_clipped")
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    if clipped > 0.01:
        raise ProfileInvalid("reference_clipped")
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    if rms_dbfs < -42:
        raise ProfileInvalid("reference_too_quiet")
    frame_samples = info.samplerate // 20
    frames = audio[: (len(audio) // frame_samples) * frame_samples].reshape(
        -1, frame_samples
    )
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    silent = float(np.mean(frame_rms < max(0.008, rms * 0.2)))
    if silent > 0.75:
        raise ProfileInvalid("reference_insufficient_speech")
    return audio, AudioQuality(
        duration_seconds=info.duration,
        sample_rate=info.samplerate,
        rms_dbfs=rms_dbfs,
        clipped_fraction=clipped,
        silence_fraction=silent,
    )


def prepare_profile(request: ProfileRequest) -> dict[str, str | float | int]:
    started = time.perf_counter()
    audio, quality = measure_audio(request.source_audio)
    marker = Marker.model_validate_json(
        (request.model_dir / "verified.json").read_text()
    )
    config = GenerationConfig.model_validate_json(
        (request.model_dir / "generation_config.json").read_text()
    )
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
    waveform = resample(torch.from_numpy(audio), quality.sample_rate, 16000)
    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    features = inputs.input_features.to("mps")
    generation = deepcopy(model.generation_config)
    generation.language = "russian"
    generation.task = "transcribe"
    generation.return_dict_in_generate = True
    generation.output_scores = True
    with torch.inference_mode():
        language_id = int(model.detect_language(input_features=features)[0])
        if language_id != config.lang_to_id["<|ru|>"]:
            raise ProfileInvalid("russian_speech_required")
        result = model.generate(
            features,
            attention_mask=inputs.attention_mask.to("mps"),
            generation_config=generation,
            return_dict_in_generate=True,
        )
        scores = model.compute_transition_scores(
            result.sequences, result.scores, normalize_logits=True
        )
        confidence = float(scores.mean())
    transcript = processor.batch_decode(
        result.sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if len(transcript.split()) < 4 or len(transcript) > 2000:
        raise ProfileInvalid("reference_insufficient_recognized_speech")
    if not math.isfinite(confidence) or confidence < -1.2:
        raise ProfileInvalid("reference_transcript_uncertain")
    data = {
        "reference_text": transcript,
        "language": "ru",
        "source": "automatic_asr",
        "review_recommended": True,
        "asr_model": "openai/whisper-small",
        "asr_revision": marker.revision,
        "mean_token_log_probability": confidence,
        "quality": quality.model_dump(),
    }
    wav = request.output_dir / "reference.wav.partial"
    text = request.output_dir / "transcript.json.partial"
    try:
        sf.write(wav, audio, quality.sample_rate, subtype="PCM_16", format="WAV")
        text.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        wav.chmod(0o600)
        text.chmod(0o600)
        os.replace(wav, request.output_dir / "reference.wav")
        os.replace(text, request.output_dir / "transcript.json")
    finally:
        wav.unlink(missing_ok=True)
        text.unlink(missing_ok=True)
    return {
        "mode": "profile",
        "backend": "mps",
        "language": "ru",
        "duration_seconds": quality.duration_seconds,
        "sample_rate": quality.sample_rate,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.request.stat().st_size > 16 * 1024:
            raise ValueError("request_too_large")
        request = ProfileRequest.model_validate_json(args.request.read_text())
        print(json.dumps(prepare_profile(request)))
        return 0
    except ProfileInvalid as error:
        print(json.dumps({"error": "profile_invalid", "reason": str(error)}))
        return 2
    except (ValidationError, ValueError, OSError) as error:
        print(
            json.dumps(
                {"error": "profile_request_invalid", "type": type(error).__name__}
            )
        )
        return 2
    except Exception as error:
        print(json.dumps({"error": "profile_failed", "type": type(error).__name__}))
        if isinstance(error, MemoryError) or (
            isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        ):
            return 75
        return 1


if __name__ == "__main__":
    sys.exit(main())
