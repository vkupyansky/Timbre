import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import accept_local
from tests.test_audio import make_pcm


def forbid_probe(*args: object, **kwargs: object) -> None:
    raise AssertionError("Неверный ввод не должен проверять порт или запускать сервисы")


@pytest.mark.parametrize("arguments", [[], ["--reference", "reference.wav"]])
def test_acceptance_requires_explicit_audio_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(accept_local, "PROJECT", tmp_path)
    monkeypatch.setattr(accept_local, "socket", SimpleNamespace(socket=forbid_probe))
    monkeypatch.setattr(accept_local, "initialize", forbid_probe)
    with pytest.raises(SystemExit) as caught:
        accept_local.main(arguments)
    assert caught.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "invalid", ["missing_reference", "short_reference", "missing_source", "source"]
)
def test_acceptance_rejects_audio_without_state_output_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    reference, source = tmp_path / "reference.wav", tmp_path / "source.wav"
    if invalid != "missing_reference":
        make_pcm(reference, 1 if invalid == "short_reference" else 15)
    if invalid == "source":
        source.write_bytes(b"not audio")
    elif invalid != "missing_source":
        make_pcm(source, 1)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(accept_local, "PROJECT", tmp_path)
    monkeypatch.setattr(accept_local, "socket", SimpleNamespace(socket=forbid_probe))
    monkeypatch.setattr(accept_local, "initialize", forbid_probe)

    with pytest.raises(SystemExit) as caught:
        accept_local.main(["--reference", str(reference), "--source", str(source)])

    assert caught.value.code == 2
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_acceptance_validates_supplied_audio_without_modifying_inputs(tmp_path: Path) -> None:
    reference, source = tmp_path / "reference.wav", tmp_path / "source.wav"
    make_pcm(reference, 15)
    make_pcm(source, 1)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    asyncio.run(accept_local.validate_inputs(reference, source))

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original
