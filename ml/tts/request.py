"""Validated local subprocess contract. Request contents are never logged."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self
from unicodedata import category

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Delivery = Literal["neutral", "angry", "happy", "sad", "whisper"]
LEGACY_DELIVERY_TAGS: dict[Delivery, tuple[str, ...]] = {
    "neutral": (),
    "angry": ("angry",),
    "happy": ("delight",),
    "sad": ("sad",),
    "whisper": ("whisper",),
}
WHISPER_TAGS = frozenset({"whisper", "whispering", "whisper in small voice"})
WHISPER_RECIPE = "fish-whisper-context-v1"


def whisper_fingerprint(
    *,
    reference_sha256: str,
    reference_text: str,
    model_revision: str,
    extra_references: Sequence[tuple[str, str]] = (),
) -> str:
    payload: dict[str, object] = {
        "recipe": WHISPER_RECIPE,
        "reference_sha256": reference_sha256,
        "reference_text": reference_text,
        "model_revision": model_revision,
    }
    if extra_references:
        payload = {
            "recipe": "fish-whisper-context-v2",
            "model_revision": model_revision,
            "references": [
                {"reference_sha256": digest, "reference_text": text}
                for digest, text in (
                    (reference_sha256, reference_text),
                    *extra_references,
                )
            ],
        }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_instruction(value: str, *, tag: bool = False) -> str:
    if (
        any(category(char).startswith("C") for char in value)
        or "<|" in value
        or "|>" in value
    ):
        raise ValueError("control_or_reserved_instruction_character")
    value = value.strip()
    depth = 0
    for char in value:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if depth < 0:
            raise ValueError("unbalanced_instruction_brackets")
    if depth or (
        tag and (not value or "[" in value or "]" in value or len(value) > 100)
    ):
        raise ValueError("invalid_native_voice_cue")
    return value


WhisperCode = Annotated[int, Field(strict=True, ge=0, le=4095)]
WhisperCodeRow = Annotated[list[WhisperCode], Field(min_length=1, max_length=256)]


class WhisperContext(BaseModel):
    """Bounded private native history, tied to the exact voice and model."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1, max_length=300)
    codes: list[WhisperCodeRow] = Field(min_length=10, max_length=10)

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("whisper_context_version_must_be_integer")
        return value

    @field_validator("text")
    @classmethod
    def context_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty_whisper_context_text")
        validate_instruction(value)
        return value

    @model_validator(mode="after")
    def rectangular_codes(self) -> Self:
        if len({len(row) for row in self.codes}) != 1:
            raise ValueError("whisper_context_code_rows_must_match")
        return self


class ReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reference_audio: Path
    reference_text: str = Field(min_length=1, max_length=2000)

    @field_validator("reference_audio")
    @classmethod
    def existing_absolute_audio(cls, value: Path) -> Path:
        if not value.is_absolute() or not value.is_file():
            raise ValueError("absolute_reference_audio_required")
        return value

    @field_validator("reference_text")
    @classmethod
    def exact_transcript(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("exact_reference_transcript_required")
        return value


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["tts"]
    text: str = Field(min_length=1, max_length=2000)
    reference_audio: Path
    reference_text: str = Field(min_length=1, max_length=2000)
    extra_references: list[ReferenceRequest] = Field(default_factory=list, max_length=2)
    output: Path
    model_dir: Path
    delivery: Delivery = "neutral"
    native_tags: list[str] = Field(default_factory=list, max_length=3)
    style_instruction: str | None = Field(default=None, max_length=500)
    temperature: float = Field(default=0.7, ge=0, le=1)
    top_p: float = Field(default=0.7, ge=0, le=1)
    top_k: int = Field(default=30, ge=0, le=4096)
    seed: int = Field(default=42, ge=0, le=4294967295)
    whisper_context: WhisperContext | None = None

    @field_validator("native_tags")
    @classmethod
    def valid_tags(cls, value: list[str]) -> list[str]:
        tags = [validate_instruction(tag, tag=True) for tag in value]
        if len(set(tags)) != len(tags):
            raise ValueError("duplicate_native_voice_cues")
        return tags

    @field_validator("style_instruction")
    @classmethod
    def valid_instruction(cls, value: str | None) -> str | None:
        return (validate_instruction(value) or None) if value is not None else None

    @property
    def effective_native_tags(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*LEGACY_DELIVERY_TAGS[self.delivery], *self.native_tags))
        )

    @property
    def uses_whisper(self) -> bool:
        return any(
            tag.strip().casefold() in WHISPER_TAGS for tag in self.effective_native_tags
        )

    @field_validator("text")
    @classmethod
    def spoken_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("empty_or_invalid_text")
        return value

    @field_validator("output", "model_dir", "reference_audio")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("absolute_path_required")
        return value

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if len(self.effective_native_tags) > 3:
            raise ValueError("at_most_three_effective_native_cues")
        if not self.model_dir.is_dir():
            raise ValueError("model_directory_missing")
        if self.output.suffix.lower() != ".wav" or not self.output.parent.is_dir():
            raise ValueError("wav_output_directory_required")
        if self.output.exists():
            raise ValueError("output_already_exists")
        if not self.reference_audio.is_file():
            raise ValueError("reference_audio_required")
        if not self.reference_text.strip():
            raise ValueError("exact_reference_transcript_required")
        return self


class Summary(BaseModel):
    backend: Literal["mlx-metal"] = "mlx-metal"
    mode: Literal["tts"]
    duration_seconds: float
    elapsed_seconds: float
    generation_seconds: float
    sample_rate: int
    peak_mlx_bytes: int
    peak_rss_bytes: int
    model_revision: str
    decoder_mode: Literal["bounded_causal"]
    decoder_blocks: int | None = None
    acoustic_tokens: int | None = None
    model_load_seconds: float | None = Field(default=None, ge=0)
    reference_encode_seconds: float | None = Field(default=None, ge=0)
    generate_codes_seconds: float | None = Field(default=None, ge=0)
    decode_seconds: float | None = Field(default=None, ge=0)
    write_seconds: float | None = Field(default=None, ge=0)
