"""Prepare pinned Whisper ASR locally; preserve already verified model files."""

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
LOCK = ROOT / "models/asr.lock.json"


class ASRFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reused_from: str | None = None  # Historical provenance in the unchanged manifest.


class ASRManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: str
    repo_id: str
    repo_type: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    local_dir: str
    files: list[ASRFile]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(path: Path, entry: ASRFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry.bytes
        and digest(path) == entry.sha256
    )


def prepare(model_dir: Path) -> None:
    manifest = ASRManifest.model_validate_json(LOCK.read_text())
    model_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest.files:
        path = (model_dir / entry.path).resolve()
        if not path.is_relative_to(model_dir):
            raise ValueError("invalid_asr_model_path")
        if matches(path, entry):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(model_dir).free < entry.bytes + 1024**3:
            raise OSError("insufficient_disk_for_profile_asr")
        partial = path.with_name(path.name + ".partial")
        url = (
            f"https://huggingface.co/{manifest.repo_id}/resolve/"
            f"{manifest.revision}/{entry.path}"
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    with partial.open("wb") as stream:
                        received = 0
                        while chunk := response.read(1024 * 1024):
                            received += len(chunk)
                            if received > entry.bytes:
                                raise ValueError("asr_model_exceeds_pinned_size")
                            stream.write(chunk)
                if not matches(partial, entry):
                    raise ValueError("asr_model_checksum_mismatch")
                os.replace(partial, path)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
            finally:
                partial.unlink(missing_ok=True)
    (model_dir / "verified.json").write_text(
        json.dumps({"manifest_sha256": digest(LOCK), "revision": manifest.revision})
        + "\n"
    )
    print(json.dumps({"ready": True, "mode": "profile", "files": len(manifest.files)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.model_dir.resolve())


if __name__ == "__main__":
    main()
