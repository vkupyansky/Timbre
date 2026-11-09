"""Download only the pinned public HF files; verify and reuse existing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]


class ModelFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repo_id: str | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_url: str
    original: str
    license: str
    quantization: str
    local_dir: str
    files: list[ModelFile]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(alias="schema")
    models: list[ModelEntry]


def valid_file(path: Path, entry: ModelFile) -> bool:
    if not path.is_file() or path.stat().st_size != entry.bytes:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == entry.sha256


def fetch(models: list[ModelEntry], model_dir: Path) -> None:
    seen: dict[str, Path] = {}
    for model in models:
        destination = (model_dir / model.key).resolve()
        if not destination.is_relative_to(model_dir):
            raise ValueError("model_destination_outside_project")
        destination.mkdir(parents=True, exist_ok=True)
        for entry in model.files:
            path = (destination / entry.path).resolve()
            if not path.is_relative_to(destination):
                raise ValueError("invalid_model_file_path")
            path.parent.mkdir(parents=True, exist_ok=True)
            if valid_file(path, entry):
                seen[entry.sha256] = path
                continue
            if entry.sha256 in seen:
                path.unlink(missing_ok=True)
                os.link(seen[entry.sha256], path)
                continue
            if shutil.disk_usage(destination).free < entry.bytes + 2_000_000_000:
                raise OSError("insufficient_disk_for_model")
            if (entry.repo_id is None) != (entry.revision is None):
                raise ValueError("incomplete_file_provenance")
            url = (
                f"https://huggingface.co/{entry.repo_id or model.repo_id}/resolve/"
                f"{entry.revision or model.revision}/{entry.path}?download=true"
            )
            part = path.with_name(path.name + ".partial")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        with part.open("wb") as stream:
                            received = 0
                            while chunk := response.read(1024 * 1024):
                                received += len(chunk)
                                if received > entry.bytes:
                                    raise ValueError(
                                        "model_download_exceeds_pinned_size"
                                    )
                                stream.write(chunk)
                    if not valid_file(part, entry):
                        raise ValueError("model_checksum_mismatch")
                    os.replace(part, path)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
                finally:
                    part.unlink(missing_ok=True)
            seen[entry.sha256] = path
            print(json.dumps({"model": model.key, "verified_file": entry.path}))
        (destination / "verified.json").write_text(
            json.dumps({"revision": model.revision, "files": len(model.files)}) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".shift/models/tts")
    args = parser.parse_args()
    manifest = Manifest.model_validate_json((ROOT / "models/tts.lock.json").read_text())
    fetch(
        manifest.models,
        args.model_dir.resolve(),
    )


if __name__ == "__main__":
    main()
