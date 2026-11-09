"""Fish subprocess entry and validation shared with its bounded decoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict

from ml.tts.request import Request, Summary


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    files: int


def validate_reference(path: Path) -> None:
    info = sf.info(path)
    if (
        info.channels != 1
        or info.samplerate not in (24000, 44100)
        or info.subtype != "PCM_16"
    ):
        raise ValueError("reference_requires_mono_24_or_44khz_pcm_wav")
    if info.format != "WAV" or not 3 <= info.duration <= 20:
        raise ValueError("reference_requires_3_to_20_seconds_wav")
    audio, _ = sf.read(path, dtype="float32")
    if not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 0.01:
        raise ValueError("reference_nonfinite_or_silent")
    if float(np.mean(np.abs(audio) >= 0.999)) > 0.01:
        raise ValueError("reference_excessively_clipped")


def generate(request: Request) -> Summary:
    config: object = json.loads((request.model_dir / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != "fish_qwen3_omni":
        raise ValueError("fish_model_required")
    from ml.tts.fish_runtime import generate as generate_fish

    return generate_fish(request)
