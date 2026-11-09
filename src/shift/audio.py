"""Bounded local-only decoding; metadata is never the duration authority."""

import array
import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_SECONDS = 180.0
SAMPLE_RATE = 24000
FORMATS = "wav,ogg,mp3,flac,aac,mov,matroska,webm"
CODECS = "pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_f64le,opus,vorbis,mp3,mp3float,flac,aac"


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str
    rms_db: float
    clipping_fraction: float


async def _command(*args: str) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 35)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        raise AudioError("Не удалось декодировать аудио.")
    return stdout


def _inspect(path: Path, max_bytes: int) -> str:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= max_bytes:
        raise AudioError("Недопустимый файл или превышен размер аудио.")
    return str(path.resolve())


def _levels(pcm: bytes) -> tuple[float, float]:
    samples = array.array("h", pcm)
    rms = math.sqrt(sum(float(x) ** 2 for x in samples) / len(samples)) / 32768
    clipping = sum(abs(x) >= 32760 for x in samples) / len(samples)
    return 20 * math.log10(max(rms, 1e-12)), clipping


def _ogg_header(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(4) == b"OggS"


async def _validate(path: Path, max_duration: float, max_bytes: int) -> AudioInfo:
    absolute = await asyncio.to_thread(_inspect, path, max_bytes)
    try:
        raw = await _command(
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-format_whitelist",
            FORMATS,
            "-codec_whitelist",
            CODECS,
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            absolute,
        )
        metadata = json.loads(raw)
        streams = metadata["streams"]
        if len(streams) != 1 or streams[0]["codec_type"] != "audio":
            raise AudioError("Нужна запись с одной звуковой дорожкой без видео.")
        stream = streams[0]
        rate, channels = int(stream["sample_rate"]), int(stream["channels"])
        if not 8000 <= rate <= 96000 or channels not in (1, 2):
            raise AudioError("Неподдерживаемые параметры звуковой дорожки.")
        pcm = await _command(
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            FORMATS,
            "-codec_whitelist",
            CODECS,
            "-i",
            absolute,
            "-t",
            str(max_duration + 0.1),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        )
        duration = len(pcm) / (SAMPLE_RATE * 2)
        if not 0.1 <= duration <= max_duration:
            raise AudioError("Превышена длительность или запись слишком короткая.")
        rms_db, clipping = await asyncio.to_thread(_levels, pcm)
        return AudioInfo(
            duration,
            rate,
            channels,
            str(stream["codec_name"]),
            rms_db,
            clipping,
        )
    except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as exc:
        raise AudioError("Некорректная звуковая дорожка.") from exc
    except TimeoutError as exc:
        raise AudioError("Проверка аудио заняла слишком много времени.") from exc


async def validate_voice(path: Path, max_duration: float = 120) -> AudioInfo:
    return await _validate(path, max_duration, MAX_INPUT_BYTES)


async def validate_result(path: Path) -> AudioInfo:
    info = await _validate(path, MAX_OUTPUT_SECONDS, MAX_RESULT_BYTES)
    if info.codec != "opus" or not await asyncio.to_thread(_ogg_header, path):
        raise AudioError("Результат должен быть голосовым OGG/Opus.")
    return info


async def validate_reference(path: Path) -> AudioInfo:
    """Accept prepared references, including profiles created before the new limit."""
    return await _validate_reference(path, min_duration=5)


async def validate_enrollment_reference(path: Path) -> AudioInfo:
    """Require the selected 15–20 second range when adding a new voice."""
    return await _validate_reference(path, min_duration=15)


async def _validate_reference(path: Path, min_duration: int) -> AudioInfo:
    info = await _validate(path, 20, MAX_INPUT_BYTES)
    if info.duration < min_duration:
        raise AudioError(
            f"Для профиля нужна чистая речь длительностью от {min_duration} до 20 секунд."
        )
    if info.rms_db < -40 or info.clipping_fraction > 0.01:
        raise AudioError("Запись слишком тихая или содержит сильные перегрузки.")
    return info


async def to_wav(
    source: Path,
    output: Path,
    max_duration: float = 120,
    *,
    sample_rate: Literal[24000, 44100] = 24000,
) -> AudioInfo:
    if sample_rate not in (24000, 44100):
        raise AudioError("Неподдерживаемая частота аудио.")
    info = await validate_voice(source, max_duration)
    await _command(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-y",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        FORMATS,
        "-codec_whitelist",
        CODECS,
        "-i",
        str(await asyncio.to_thread(source.resolve)),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(await asyncio.to_thread(output.resolve)),
    )
    return info


async def to_voice(source: Path, output: Path, *, speed: float = 1.0) -> AudioInfo:
    if isinstance(speed, bool) or speed not in (0.9, 1.0, 1.1):
        raise AudioError("Недопустимая скорость озвучки.")
    info = await _validate(source, MAX_OUTPUT_SECONDS, 72 * 1024 * 1024)
    if info.duration / speed > MAX_OUTPUT_SECONDS:
        raise AudioError("При этой скорости голосовое будет длиннее трёх минут.")
    # WSOLA changes tempo while retaining pitch. The MLX library's speed
    # interpolation changes pitch too and is unsuitable for a chosen voice.
    tempo = ("-af", f"atempo={speed}") if speed != 1.0 else ()
    await _command(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-y",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        FORMATS,
        "-codec_whitelist",
        CODECS,
        "-i",
        str(await asyncio.to_thread(source.resolve)),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        *tempo,
        "-c:a",
        "libopus",
        "-b:a",
        "48k",
        "-vbr",
        "on",
        "-application",
        "voip",
        "-f",
        "ogg",
        str(await asyncio.to_thread(output.resolve)),
    )
    return await validate_result(output)
