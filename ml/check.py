"""Offline readiness: pinned file integrity and real Metal/MPS tensor work."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


class TTSFile(BaseModel):
    path: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TTSModel(BaseModel):
    key: str
    revision: str
    files: list[TTSFile]


class TTSManifest(BaseModel):
    models: list[TTSModel]


class ASRManifest(BaseModel):
    files: list[TTSFile]


class TTSVerification(BaseModel):
    revision: str
    files: int


class ASRVerification(BaseModel):
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def verify_file(root: Path, path: Path, size: int, checksum: str) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("model_file_outside_model_directory")
    if not resolved.is_file() or resolved.stat().st_size != size:
        raise ValueError("model_file_missing_or_wrong_size")
    with resolved.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
            raise ValueError("model_file_checksum_mismatch")


def verify_models(model_dir: Path) -> int:
    files = 0
    tts = TTSManifest.model_validate_json((ROOT / "models/tts.lock.json").read_text())
    for model in tts.models:
        marker = TTSVerification.model_validate_json(
            (model_dir / "tts" / model.key / "verified.json").read_text()
        )
        if marker.revision != model.revision or marker.files != len(model.files):
            raise ValueError("tts_verification_marker_mismatch")
        for entry in model.files:
            verify_file(
                model_dir,
                model_dir / "tts" / model.key / entry.path,
                entry.bytes,
                entry.sha256,
            )
            files += 1
    asr_lock = ROOT / "models/asr.lock.json"
    asr = ASRManifest.model_validate_json(asr_lock.read_text())
    asr_marker = ASRVerification.model_validate_json(
        (model_dir / "asr/verified.json").read_text()
    )
    with asr_lock.open("rb") as stream:
        if (
            hashlib.file_digest(stream, "sha256").hexdigest()
            != asr_marker.manifest_sha256
        ):
            raise ValueError("asr_verification_marker_mismatch")
    for asr_file in asr.files:
        verify_file(
            model_dir,
            model_dir / "asr" / asr_file.path,
            asr_file.bytes,
            asr_file.sha256,
        )
        files += 1
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    stage = "platform"
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("native_apple_silicon_required")
        stage = "model_files"
        if not args.model_dir.is_absolute():
            raise ValueError("absolute_model_directory_required")
        verified = verify_models(args.model_dir.resolve())
        stage = "accelerators"
        import mlx.core as mx
        import torch

        if not mx.metal.is_available() or not torch.backends.mps.is_available():
            raise RuntimeError("metal_or_mps_unavailable")
        mx.set_default_device(mx.gpu)
        mx.eval(mx.ones((2, 2)) @ mx.ones((2, 2)))
        tensor = torch.ones((2, 2), device="mps")
        if float((tensor @ tensor).cpu()[0, 0]) != 2.0:
            raise RuntimeError("mps_operation_failed")
        print(
            json.dumps(
                {"ready": True, "files_verified": verified, "mlx": True, "mps": True}
            )
        )
        return 0
    except Exception as error:
        print(
            json.dumps({"ready": False, "stage": stage, "error": type(error).__name__})
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
