"""Bound the waveform decoder of the pinned mlx-audio 0.5.3 Fish S1 DAC.

The post-quantization transformer processes all codes once, retaining its exact
causal context and positions. The following decoder contains only finite causal
convolutions: overlap is discarded, never crossfaded or synthesized separately.
"""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx
import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 44100
CODE_HOP = 2048
DECODER_HOP = 512
MAX_CODE_FRAMES = 180 * SAMPLE_RATE // CODE_HOP
CHUNK_LATENT_FRAMES = 256
LEFT_CONTEXT_FRAMES = 32


class Quantizer(Protocol):
    def decode(self, indices: mx.array) -> mx.array: ...


class Decoder(Protocol):
    def __call__(self, latent: mx.array) -> mx.array: ...


class FishCodec(Protocol):
    quantizer: Quantizer
    decoder: Decoder
    sample_rate: int
    frame_length: int
    decoder_rates: list[int]
    latent_dim: int


def decoder_left_context() -> int:
    """Exact input history at a 512-sample boundary for this causal decoder.

    Final conv7 consumes six earlier samples. In reverse order, each block has
    three conv7 residual units (dilations 1,3,9), followed by the inverse index
    mapping of transpose conv k=2*stride: floor(sample/stride)-1. The initial
    conv7 contributes another six latent frames. This gives 19 frames; 32 are
    retained so implementation details have a conservative margin.
    """
    earliest = -6
    for stride in (2, 4, 8, 8):
        earliest = (earliest - 6 * (1 + 3 + 9)) // stride - 1
    return -(earliest - 6)


def decode_bounded(
    codec: FishCodec, codes: mx.array
) -> tuple[npt.NDArray[np.float32], int]:
    """Decode one complete utterance, returning mono float32 and chunk count.

    The supported codec has ten codebooks and a 2048-sample code hop. Its
    transformer position table has 4096 entries; the 180-second job limit is
    stricter (3875 codes). Callers must preserve the same sequential GPU lock
    and memory limit used during language-model generation.
    """
    if (
        codec.sample_rate != SAMPLE_RATE
        or codec.frame_length != CODE_HOP
        or tuple(codec.decoder_rates) != (8, 8, 4, 2)
        or codec.latent_dim != 1024
        or LEFT_CONTEXT_FRAMES < decoder_left_context()
    ):
        raise ValueError("unsupported_fish_codec_architecture")
    if codes.ndim == 2:
        codes = codes[None]
    if (
        codes.ndim != 3
        or codes.shape[0] != 1
        or codes.shape[1] != 10
        or not 1 <= codes.shape[2] <= MAX_CODE_FRAMES
        or not mx.issubdtype(codes.dtype, mx.integer)
    ):
        raise ValueError("invalid_fish_codes")

    latent = codec.quantizer.decode(codes)
    expected_latent_frames = codes.shape[-1] * (CODE_HOP // DECODER_HOP)
    if latent.shape != (1, 1024, expected_latent_frames):
        raise RuntimeError("unexpected_fish_latent_shape")
    mx.eval(latent)
    mx.clear_cache()
    output = np.empty(codes.shape[-1] * CODE_HOP, dtype=np.float32)
    chunks = 0
    for start in range(0, expected_latent_frames, CHUNK_LATENT_FRAMES):
        stop = min(start + CHUNK_LATENT_FRAMES, expected_latent_frames)
        context_start = max(0, start - LEFT_CONTEXT_FRAMES)
        decoded = codec.decoder(latent[..., context_start:stop])
        expected_samples = (stop - context_start) * DECODER_HOP
        if decoded.shape != (1, 1, expected_samples):
            raise RuntimeError("unexpected_fish_decoder_shape")
        mx.eval(decoded)
        # Assignment owns the CPU copy; retaining an MLX-backed NumPy view would
        # keep decoder buffers alive across chunks on unified memory.
        skip = (start - context_start) * DECODER_HOP
        output[start * DECODER_HOP : stop * DECODER_HOP] = np.asarray(
            decoded[0, 0, skip:], dtype=np.float32
        )
        del decoded
        mx.clear_cache()
        chunks += 1
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite_fish_decoder_output")
    return output, chunks
