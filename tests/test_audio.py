import asyncio
import math
import struct
import wave
from pathlib import Path

import pytest

from shift.audio import (
    AudioError,
    to_voice,
    to_wav,
    validate_enrollment_reference,
    validate_reference,
    validate_result,
    validate_voice,
)


def make_pcm(
    path: Path,
    seconds: float,
    amplitude: int = 9000,
    rate: int = 24000,
    frequency: int = 220,
) -> None:
    # Anonymous tone is solely a codec/bounds fixture, never a voice profile for users.
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(
            b"".join(
                struct.pack("<h", int(amplitude * math.sin(i * 2 * math.pi * frequency / rate)))
                for i in range(int(seconds * rate))
            )
        )


async def test_real_ogg_opus_roundtrip_and_actual_duration(tmp_path: Path) -> None:
    wav, result = tmp_path / "source.wav", tmp_path / "result.ogg"
    make_pcm(wav, 1)
    info = await to_voice(wav, result)
    assert info.codec == "opus"
    assert 0.99 < info.duration < 1.02
    assert result.read_bytes()[:4] == b"OggS"
    converted = tmp_path / "converted.wav"
    await to_wav(result, converted)
    assert (await validate_voice(converted)).sample_rate == 24000


async def test_fish_reference_retains_audio_above_12khz(tmp_path: Path) -> None:
    source, reference, speech = (tmp_path / name for name in ("source.wav", "ref.wav", "asr.wav"))
    make_pcm(source, 1, rate=48000, frequency=14000)
    await to_wav(source, reference, sample_rate=44100)
    await to_wav(source, speech)

    def rms(path: Path, expected_rate: int) -> float:
        with wave.open(str(path), "rb") as audio:
            assert audio.getframerate() == expected_rate
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getnframes() == expected_rate
            pcm = audio.readframes(audio.getnframes())
        samples = struct.unpack(f"{len(pcm) // 2}h", pcm)[100:-100]
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    source_rms = rms(source, 48000)
    assert rms(reference, 44100) > source_rms * 0.95
    assert rms(speech, 24000) < source_rms * 0.01


async def test_duration_is_decoded_even_when_extension_lies(tmp_path: Path) -> None:
    misleading = tmp_path / "pretend.ogg"
    make_pcm(misleading, 2)
    assert (await validate_voice(misleading)).codec == "pcm_s16le"
    with pytest.raises(AudioError):
        await validate_voice(misleading, max_duration=1)
    with pytest.raises(AudioError):
        await validate_result(misleading)


async def test_corrupt_and_external_playlist_rejected(tmp_path: Path) -> None:
    for contents in (b"not audio", b"#EXTM3U\nhttps://127.0.0.1/secret\n"):
        file = tmp_path / "voice.ogg"
        file.write_bytes(contents)
        with pytest.raises(AudioError):
            await validate_voice(file)


async def test_reference_rejects_short_silent_and_clipping(tmp_path: Path) -> None:
    wav = tmp_path / "reference.wav"
    for seconds, amplitude in ((1, 9000), (6, 0), (6, 32767)):
        make_pcm(wav, seconds, amplitude)
        with pytest.raises(AudioError):
            await validate_reference(wav)
    make_pcm(wav, 6)
    info = await validate_reference(wav)
    assert info.duration == 6


@pytest.mark.parametrize("seconds", [14.99, 15.0, 20.0, 20.01])
async def test_new_voice_duration_uses_actual_decoded_boundaries(
    tmp_path: Path, seconds: float
) -> None:
    path = tmp_path / "new-reference.wav"
    make_pcm(path, seconds)
    if 15 <= seconds <= 20:
        assert (await validate_enrollment_reference(path)).duration == seconds
    else:
        with pytest.raises(AudioError):
            await validate_enrollment_reference(path)


async def test_existing_short_profile_remains_usable_but_cannot_be_new_upload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing-reference.wav"
    make_pcm(path, 6)
    assert (await validate_reference(path)).duration == 6
    with pytest.raises(AudioError, match="от 15 до 20"):
        await validate_enrollment_reference(path)


async def test_oversized_sparse_input_rejected_before_decoder(tmp_path: Path) -> None:
    wav = tmp_path / "large.wav"
    with wav.open("wb") as file:
        file.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(AudioError):
        await validate_voice(wav)


async def test_audio_check_does_not_block_event_loop(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    make_pcm(wav, 10)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.01)
            ticks += 1

    await asyncio.gather(validate_voice(wav), ticker())
    assert ticks == 10


@pytest.mark.parametrize("speed", [0.9, 1.1])
async def test_tempo_changes_duration_without_shifting_pitch(tmp_path: Path, speed: float) -> None:
    source, result, decoded = (tmp_path / name for name in ("tone.wav", "voice.ogg", "decoded.wav"))
    make_pcm(source, 2)
    info = await to_voice(source, result, speed=speed)
    assert abs(info.duration - 2 / speed) < 0.05
    await to_wav(result, decoded)
    with wave.open(str(decoded), "rb") as audio:
        rate = audio.getframerate()
        data = audio.readframes(audio.getnframes())
    samples = struct.unpack(f"{len(data) // 2}h", data)
    middle = samples[rate // 5 : -rate // 5]
    crossings = sum(left < 0 <= right for left, right in zip(middle, middle[1:], strict=False))
    frequency = crossings / (len(middle) / rate)
    assert abs(frequency - 220) < 3


@pytest.mark.parametrize("speed", [0.0, 2.0, float("nan"), True])
async def test_invalid_tempo_rejected_before_decoding(tmp_path: Path, speed: float) -> None:
    with pytest.raises(AudioError, match="скорость"):
        await to_voice(tmp_path / "missing.wav", tmp_path / "output.ogg", speed=speed)
