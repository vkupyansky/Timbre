"""Validated payloads shared by the local worker and Telegram application."""

from typing import Literal, Self
from unicodedata import category

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_TEXT = 2000
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 120
MAX_WAITING = 3
LEASE_SECONDS = 90
JOB_TIMEOUT_SECONDS = 15 * 60
RETENTION_SECONDS = 24 * 60 * 60
MAX_PROFILE_NAME = 80
MAX_PROFILE_REFERENCE_BYTES = 4 * 1024 * 1024
MAX_PROFILE_EXAMPLE_BYTES = 1024 * 1024
PREVIEW_TEXT = (
    "Привет! Это пример моего голоса. Теперь можно отправить текст или голосовое сообщение."
)

JobKind = Literal["tts", "vc", "profile", "reference"]
VoiceDelivery = Literal["neutral", "happy", "angry", "sad", "whisper"]
VoiceSettingKey = Literal[
    "native_tag", "style_instruction", "temperature", "top_p", "top_k", "seed"
]
LEGACY_DELIVERY_TAGS: dict[VoiceDelivery, tuple[str, ...]] = {
    "neutral": (),
    "happy": ("delight",),
    "angry": ("angry",),
    "sad": ("sad",),
    "whisper": ("whisper",),
}


def validate_voice_instruction(value: str, *, tag: bool = False) -> str:
    if any(category(char).startswith("C") for char in value) or "<|" in value or "|>" in value:
        raise ValueError("Control characters and reserved tokens are not allowed")
    value = value.strip()
    depth = 0
    for char in value:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if depth < 0:
            raise ValueError("Unbalanced instruction brackets")
    if depth or (tag and (not value or "[" in value or "]" in value or len(value) > 100)):
        raise ValueError("Invalid native voice cue")
    return value


FailureReason = Literal[
    "model_error",
    "out_of_memory",
    "invalid_audio",
    "timeout",
    "worker_lost",
    "restart",
    "storage_error",
    "delivery_error",
    "not_ready",
    "profile_invalid",
    "reference_transcription_failed",
    "speech_unrecognized",
    "speech_too_long",
    "whisper_preparation_failed",
    "insufficient_balance",
]
JobStatus = Literal[
    "queued",
    "running",
    "ready",
    "sending",
    "sent",
    "uncertain",
    "failed",
    "cancelled",
]


class ReferenceSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    reference_audio: str = Field(min_length=1, max_length=250)
    reference_text: str = Field(min_length=1, max_length=2000)

    @field_validator("reference_text")
    @classmethod
    def transcript_required(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Reference transcript is required")
        return value


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    name: str = Field(min_length=1, max_length=80)
    reference_audio: str = Field(min_length=1, max_length=250)
    reference_text: str = Field(min_length=1, max_length=2000)
    extra_references: tuple[ReferenceSample, ...] = Field(default=(), max_length=2)
    delivery: VoiceDelivery = "neutral"
    speed: float = 1.0
    native_tags: tuple[str, ...] = Field(default=(), max_length=3)
    style_instruction: str | None = Field(default=None, max_length=500)
    temperature: float = Field(default=0.7, ge=0, le=1)
    top_p: float = Field(default=0.7, ge=0, le=1)
    top_k: int = Field(default=30, ge=0, le=4096)
    seed: int = Field(default=42, ge=0, le=4294967295)

    @field_validator("extra_references", mode="before")
    @classmethod
    def references_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @property
    def reference_samples(self) -> tuple[ReferenceSample, ...]:
        return (
            ReferenceSample(
                reference_audio=self.reference_audio, reference_text=self.reference_text
            ),
            *self.extra_references,
        )

    @field_validator("native_tags", mode="before")
    @classmethod
    def tags_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("native_tags")
    @classmethod
    def valid_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        tags = tuple(validate_voice_instruction(tag, tag=True) for tag in value)
        if len(set(tags)) != len(tags):
            raise ValueError("Duplicate native voice cues")
        return tags

    @field_validator("style_instruction")
    @classmethod
    def valid_instruction(cls, value: str | None) -> str | None:
        return (validate_voice_instruction(value) or None) if value is not None else None

    @property
    def effective_native_tags(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*LEGACY_DELIVERY_TAGS[self.delivery], *self.native_tags)))

    @model_validator(mode="after")
    def effective_tag_limit(self) -> Self:
        if len(self.effective_native_tags) > 3:
            raise ValueError("At most three effective native cues are allowed")
        return self

    @field_validator("speed")
    @classmethod
    def supported_speed(cls, value: float) -> float:
        if value not in (0.9, 1.0, 1.1):
            raise ValueError("Unsupported voice speed")
        return value


class WorkerJob(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    kind: JobKind
    text: str | None
    profile: Profile | None
    lease: str


class ClaimResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    job: WorkerJob | None


class FailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: FailureReason


class TranscriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=MAX_TEXT)

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Transcript must contain speech")
        return value


class HeartbeatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cancel: bool
