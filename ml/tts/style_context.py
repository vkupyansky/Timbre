"""CPU rejection heuristic for an attempted whisper context."""

import numpy as np
from numpy.typing import NDArray


class WhisperUnavailable(RuntimeError):
    """The attempted context did not pass the whisper rejection heuristic."""


def require_whisper(audio: NDArray[np.float32], sample_rate: int) -> None:
    """Reject voiced, silent or invalid context; do not certify whisper quality.

    This periodicity heuristic cannot guarantee speech, speaker identity or
    naturalness. It uses 40 ms frames, 10 ms hops and energy-normalized ACF over
    60–444 Hz pitch lags at the original sample rate, without loading ML or
    resampling dependencies. Broadband noise can pass; this is a rejection gate.
    """
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or not 8000 <= sample_rate <= 192000
        or audio.ndim != 1
        or not np.isfinite(audio).all()
    ):
        raise WhisperUnavailable("whisper_context_invalid_audio")
    frame_size = round(sample_rate * 0.04)
    hop = round(sample_rate * 0.01)
    if audio.size < frame_size + 4 * hop:
        raise WhisperUnavailable("whisper_context_too_short")
    magnitude = np.abs(audio)
    if np.any(magnitude > 1.0) or np.mean(magnitude >= 0.999) >= 0.01:
        raise WhisperUnavailable("whisper_context_clipped")

    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[::hop]
    centered = frames.astype(np.float64) - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    threshold = max(10 ** (-45 / 20), float(rms.max()) * 10 ** (-30 / 20))
    active = centered[rms >= threshold]
    if len(active) < 5:
        raise WhisperUnavailable("whisper_context_insufficient_audio")

    lags = np.arange(int(sample_rate / 444), int(sample_rate / 60) + 1)
    voiced = 0
    for frame in active:
        energy = np.concatenate((np.zeros(1), np.cumsum(frame * frame)))
        denominator = np.sqrt(
            energy[frame_size - lags] * np.maximum(energy[-1] - energy[lags], 0)
        )
        correlation = np.correlate(frame, frame, mode="full")[frame_size - 1 + lags]
        coefficients = np.divide(
            correlation,
            denominator,
            out=np.zeros_like(correlation),
            where=denominator > 1e-14,
        )
        voiced += bool(np.any(coefficients >= 0.65))
    if voiced / len(active) >= 0.2:
        raise WhisperUnavailable("whisper_context_voiced")
