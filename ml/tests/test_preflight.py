import hashlib
import json
from pathlib import Path

import pytest
from ml import check
from ml.check import verify_file


def test_integrity_requires_content_not_only_size(tmp_path: Path) -> None:
    path = tmp_path / "model"
    path.write_bytes(b"safe")
    checksum = hashlib.sha256(b"safe").hexdigest()
    verify_file(tmp_path, path, 4, checksum)
    path.write_bytes(b"evil")
    with pytest.raises(ValueError, match="checksum"):
        verify_file(tmp_path, path, 4, checksum)


def test_symlink_cannot_escape_models(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"safe")
    link = models / "model"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        verify_file(models, link, 4, hashlib.sha256(b"safe").hexdigest())


def test_readiness_needs_current_tts_and_asr_without_historical_vc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    locks = project / "models"
    locks.mkdir(parents=True)
    models = tmp_path / "installed"
    tts = models / "tts/base"
    asr = models / "asr"
    tts.mkdir(parents=True)
    asr.mkdir()
    payload = b"pinned model"
    entry = {
        "path": "model.bin",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    for directory in (tts, asr):
        (directory / "model.bin").write_bytes(payload)
    (locks / "tts.lock.json").write_text(
        json.dumps({"models": [{"key": "base", "revision": "r1", "files": [entry]}]})
    )
    (tts / "verified.json").write_text(json.dumps({"revision": "r1", "files": 1}))
    asr_lock = json.dumps({"files": [entry]}).encode()
    (locks / "asr.lock.json").write_bytes(asr_lock)
    (asr / "verified.json").write_text(
        json.dumps({"manifest_sha256": hashlib.sha256(asr_lock).hexdigest()})
    )
    monkeypatch.setattr(check, "ROOT", project)
    assert check.verify_models(models) == 2
    assert not (models / "vc").exists()
