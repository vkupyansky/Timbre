"""Private per-voice Fish continuation cache, bound to the exact reference and model."""

import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from shift.protocol import MAX_PROFILE_REFERENCE_BYTES, Profile, validate_voice_instruction

CACHE_NAME = "whisper-context.json"
MAX_CONTEXT_BYTES = 32 * 1024
MAX_REQUEST_BYTES = 64 * 1024
WHISPER_TAGS = frozenset({"whisper", "whispering", "whisper in small voice"})
Code = Annotated[int, Field(strict=True, ge=0, le=4095)]
CodeRow = Annotated[list[Code], Field(min_length=1, max_length=256)]


class WhisperContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1, max_length=300)
    codes: list[CodeRow] = Field(min_length=10, max_length=10)

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        if not validate_voice_instruction(value):
            raise ValueError("Whisper context text must not be blank")
        return value

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported whisper context version")
        return value

    @model_validator(mode="after")
    def rectangular_codes(self) -> Self:
        if len({len(row) for row in self.codes}) != 1:
            raise ValueError("Whisper context codebooks must have equal lengths")
        return self


def whisper_fingerprint(
    *,
    reference_sha256: str,
    reference_text: str,
    model_revision: str,
    extra_references: Sequence[tuple[str, str]] = (),
) -> str:
    if extra_references:
        payload: dict[str, object] = {
            "recipe": "fish-whisper-context-v2",
            "model_revision": model_revision,
            "references": [
                {"reference_sha256": digest, "reference_text": text}
                for digest, text in ((reference_sha256, reference_text), *extra_references)
            ],
        }
    else:
        payload = {
            "recipe": "fish-whisper-context-v1",
            "reference_sha256": reference_sha256,
            "reference_text": reference_text,
            "model_revision": model_revision,
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def wants_whisper(profile: Profile) -> bool:
    return any(tag.strip().casefold() in WHISPER_TAGS for tag in profile.effective_native_tags)


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Open each directory without following links, including the state/profile ancestors."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in absolute.parts[1:]:
            if part in (".", ".."):
                raise ValueError("Invalid cache directory")
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _read_file(directory: int, name: str, limit: int, *, private: bool = False) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as source:
        details = os.fstat(source.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size > limit:
            raise ValueError("Invalid whisper cache file")
        if private and stat.S_IMODE(details.st_mode) != 0o600:
            raise ValueError("Whisper cache requires private permissions")
        content = source.read(limit + 1)
        if len(content) > limit:
            raise ValueError("Whisper cache exceeds its size limit")
        return content


def _read_path(path: Path, limit: int, *, private: bool = False) -> bytes:
    with _directory(path.parent) as directory:
        return _read_file(directory, path.name, limit, private=private)


def _model_revision(state: Path) -> str:
    # The active model directory may intentionally be a link to a pinned local model.
    marker = (state / "models/tts/base").resolve(strict=True) / "verified.json"
    values = json.loads(_read_path(marker, MAX_CONTEXT_BYTES))
    revision = values.get("revision") if isinstance(values, dict) else None
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        raise ValueError("Invalid model revision")
    return revision


def _expected_fingerprint(
    state: Path, profile: Profile, reference: Path, *, extra_references: Sequence[Path] = ()
) -> str:
    if len(extra_references) != len(profile.extra_references):
        raise ValueError("Whisper cache reference count mismatch")
    digest = hashlib.sha256(_read_path(reference, MAX_PROFILE_REFERENCE_BYTES)).hexdigest()
    return whisper_fingerprint(
        reference_sha256=digest,
        reference_text=profile.reference_text,
        model_revision=_model_revision(state),
        extra_references=tuple(
            (
                hashlib.sha256(_read_path(path, MAX_PROFILE_REFERENCE_BYTES)).hexdigest(),
                sample.reference_text,
            )
            for path, sample in zip(extra_references, profile.extra_references, strict=True)
        ),
    )


def load_whisper_context(
    state: Path, profile: Profile, reference: Path, *, extra_references: Sequence[Path] = ()
) -> dict[str, object] | None:
    if not wants_whisper(profile):
        return None
    try:
        expected = _expected_fingerprint(
            state, profile, reference, extra_references=extra_references
        )
        with _directory(state / "data/profiles" / profile.id) as directory:
            context = WhisperContext.model_validate_json(
                _read_file(directory, CACHE_NAME, MAX_CONTEXT_BYTES, private=True)
            )
        return context.model_dump(mode="json") if context.fingerprint == expected else None
    except (OSError, ValueError, ValidationError):
        return None


def _relative_reference_digest(directory: int, name: str) -> str:
    relative = Path(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in (".", "..") for part in relative.parts)
    ):
        raise ValueError("Invalid live reference path")
    parent = os.dup(directory)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        return hashlib.sha256(
            _read_file(parent, relative.name, MAX_PROFILE_REFERENCE_BYTES)
        ).hexdigest()
    finally:
        os.close(parent)


def _live_reference_matches(directory: int, profile: Profile, expected: str, revision: str) -> bool:
    current = Profile.model_validate_json(_read_file(directory, "profile.json", MAX_CONTEXT_BYTES))
    if current.id != profile.id or tuple(
        sample.reference_text for sample in current.reference_samples
    ) != tuple(sample.reference_text for sample in profile.reference_samples):
        return False
    references = tuple(
        (_relative_reference_digest(directory, sample.reference_audio), sample.reference_text)
        for sample in current.reference_samples
    )
    return (
        whisper_fingerprint(
            reference_sha256=references[0][0],
            reference_text=references[0][1],
            model_revision=revision,
            extra_references=references[1:],
        )
        == expected
    )


def _still_present(path: Path, descriptor: int) -> bool:
    with _directory(path) as current:
        first, second = os.fstat(descriptor), os.fstat(current)
        return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def save_whisper_context(
    state: Path,
    profile: Profile,
    reference: Path,
    exported: Path,
    *,
    extra_references: Sequence[Path] = (),
) -> bool:
    """Save a successful generation's cache without recreating or following a removed profile."""
    if not wants_whisper(profile):
        return False
    try:
        expected = _expected_fingerprint(
            state, profile, reference, extra_references=extra_references
        )
        context = WhisperContext.model_validate_json(
            _read_path(exported, MAX_CONTEXT_BYTES, private=True)
        )
        if context.fingerprint != expected:
            return False
        payload = context.model_dump_json().encode()
        folder = state / "data/profiles" / profile.id
        with _directory(folder) as directory:
            if not _live_reference_matches(directory, profile, expected, _model_revision(state)):
                return False
            if not _still_present(folder, directory):
                return False
            try:
                existing = os.stat(CACHE_NAME, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(existing.st_mode):
                    return False
            temporary = f".whisper-context-{secrets.token_hex(12)}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            try:
                with os.fdopen(descriptor, "wb") as target:
                    os.fchmod(target.fileno(), 0o600)
                    target.write(payload)
                    target.flush()
                    os.fsync(target.fileno())
                if not _still_present(folder, directory):
                    return False
                if not _live_reference_matches(
                    directory, profile, expected, _model_revision(state)
                ):
                    return False
                os.replace(temporary, CACHE_NAME, src_dir_fd=directory, dst_dir_fd=directory)
                return True
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
    except (OSError, ValueError, ValidationError):
        return False
