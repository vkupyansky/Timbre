"""Synthetic unit signals exercise rejection, without model or audio outputs."""

import numpy as np
import pytest
from ml.tts.style_context import WhisperUnavailable, require_whisper
from numpy.typing import NDArray


def noise(sample_rate: int) -> NDArray[np.float32]:
    samples = np.random.default_rng(42).uniform(-0.1, 0.1, sample_rate)
    time = np.arange(sample_rate) / sample_rate
    return (samples * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * time) ** 2)).astype(
        np.float32
    )


@pytest.mark.parametrize("sample_rate", [16000, 24000, 44100])
def test_harmonic_voice_is_rejected(sample_rate: int) -> None:
    time = np.arange(sample_rate) / sample_rate
    audio = (
        0.1 * np.sin(2 * np.pi * 220 * time) + 0.04 * np.sin(2 * np.pi * 440 * time)
    ).astype(np.float32)
    with pytest.raises(WhisperUnavailable, match="voiced"):
        require_whisper(audio, sample_rate)


@pytest.mark.parametrize("sample_rate", [16000, 24000, 44100])
def test_bounded_unvoiced_noise_passes_rejection_only(sample_rate: int) -> None:
    require_whisper(noise(sample_rate), sample_rate)


def test_silence_cannot_dilute_voiced_fraction() -> None:
    time = np.arange(3200) / 16000
    voiced = (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    audio = np.concatenate((voiced, np.zeros(12800, dtype=np.float32)))
    with pytest.raises(WhisperUnavailable, match="voiced"):
        require_whisper(audio, 16000)


@pytest.mark.parametrize("value", [0.0, 0.1, 0.0001])
def test_silence_and_dc_are_rejected(value: float) -> None:
    with pytest.raises(WhisperUnavailable, match="insufficient_audio"):
        require_whisper(np.full(16000, value, dtype=np.float32), 16000)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_audio_is_rejected(value: float) -> None:
    audio = noise(16000)
    audio[0] = value
    with pytest.raises(WhisperUnavailable, match="invalid_audio"):
        require_whisper(audio, 16000)


def test_quiet_noise_and_single_click_are_insufficient() -> None:
    for audio in (noise(16000) * 0.001, np.zeros(16000, dtype=np.float32)):
        audio[8000] = 0.05
        with pytest.raises(WhisperUnavailable, match="insufficient_audio"):
            require_whisper(audio, 16000)


@pytest.mark.parametrize("samples", [0, 1279])
def test_short_audio_is_rejected(samples: int) -> None:
    with pytest.raises(WhisperUnavailable, match="too_short"):
        require_whisper(noise(16000)[:samples], 16000)


@pytest.mark.parametrize("value", [1.0, 1.1, -1.1])
def test_obvious_clipping_is_rejected(value: float) -> None:
    audio = noise(16000)
    audio[:200] = value
    with pytest.raises(WhisperUnavailable, match="clipped"):
        require_whisper(audio, 16000)


def test_stereo_is_invalid() -> None:
    with pytest.raises(WhisperUnavailable, match="invalid_audio"):
        require_whisper(np.zeros((16000, 2), dtype=np.float32), 16000)


@pytest.mark.parametrize("sample_rate", [0, -16000, 441, 192001])
def test_invalid_sample_rate_is_rejected(sample_rate: int) -> None:
    with pytest.raises(WhisperUnavailable, match="invalid_audio"):
        require_whisper(noise(16000), sample_rate)
