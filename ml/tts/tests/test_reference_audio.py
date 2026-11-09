"""Prepared references retain their native rate while legacy profiles stay valid."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from ml.tts.runtime import validate_reference


@pytest.mark.parametrize("sample_rate", [24000, 44100])
@pytest.mark.parametrize("duration", [3, 15, 20])
def test_prepared_reference_accepts_native_and_legacy_rates(
    tmp_path: Path, sample_rate: int, duration: int
) -> None:
    path = tmp_path / "reference.wav"
    audio = 0.2 * np.sin(np.arange(sample_rate * duration, dtype=np.float32) * 0.1)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    validate_reference(path)


@pytest.mark.parametrize(
    "sample_rate,subtype,channels",
    [
        (16000, "PCM_16", 1),
        (48000, "PCM_16", 1),
        (44100, "FLOAT", 1),
        (44100, "PCM_16", 2),
    ],
)
def test_unprepared_reference_requires_normalization(
    tmp_path: Path, sample_rate: int, subtype: str, channels: int
) -> None:
    path = tmp_path / "reference.wav"
    audio = np.ones((sample_rate * 3, channels), dtype=np.float32) * 0.2
    sf.write(path, audio, sample_rate, subtype=subtype)
    with pytest.raises(ValueError, match="mono_24_or_44khz_pcm_wav"):
        validate_reference(path)
