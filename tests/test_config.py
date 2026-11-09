import secrets
from pathlib import Path

import pytest

from shift.config import ConfigError, atomic_private, initialize, read_env, read_worker_token


def env(path: Path, owner: str = "456789") -> str:
    token = "123456:" + secrets.token_urlsafe(32)
    atomic_private(
        path,
        (f"TELEGRAM_BOT_TOKEN={token}\nTELEGRAM_ADMIN_USER_IDS={owner}\n").encode(),
    )
    return token


def test_internal_secret_is_automatic_and_idempotent(tmp_path: Path) -> None:
    state = tmp_path / "state"
    initialize(state)
    original = read_worker_token(state)
    initialize(state)
    assert read_worker_token(state) == original
    assert not (state / "secrets/telegram.token").exists()
    assert not (state / "config.json").exists()
    assert state.stat().st_mode & 0o777 == 0o700
    assert (state / "secrets/worker.token").stat().st_mode & 0o777 == 0o600


def test_unsafe_secret_permissions_and_symlink_rejected(tmp_path: Path) -> None:
    initialize(tmp_path)
    target = tmp_path / "secrets/worker.token"
    target.chmod(0o644)
    with pytest.raises(ConfigError):
        read_worker_token(tmp_path)
    target.unlink()
    outside = tmp_path / "outside"
    outside.write_text(secrets.token_urlsafe(48))
    target.symlink_to(outside)
    with pytest.raises(ConfigError):
        initialize(tmp_path)
    with pytest.raises(ConfigError):
        atomic_private(target, b"should not be written")


def test_only_env_is_needed_and_secrets_are_redacted(tmp_path: Path) -> None:
    file = tmp_path / ".env"
    token = env(file)
    settings = read_env(file)
    assert settings.admin_user_ids == (456789,)
    assert settings.telegram_token.get_secret_value() == token
    assert token not in repr(settings)
    assert file.stat().st_mode & 0o777 == 0o600
    for owner in ("0", "-1", "null", "", "@username"):
        env(file, owner)
        with pytest.raises(ConfigError):
            read_env(file)


def test_missing_or_unknown_settings_fail_without_environment_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / ".env"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "not a source")
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "456789")
    for data in (b"", b"TELEGRAM_BOT_TOKEN=\nTELEGRAM_OWNER_ID=\n", b"OWNER_ID=456789\n"):
        atomic_private(file, data)
        with pytest.raises(ConfigError):
            read_env(file)


def test_dotenv_interpolation_cannot_pull_secrets_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / ".env"
    monkeypatch.setenv("INJECTED", "456789")
    env(file, "${INJECTED}")
    with pytest.raises(ConfigError):
        read_env(file)


def test_administrators_are_explicit_required_and_never_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / ".env"
    env(file)
    original = file.read_bytes().split(b"TELEGRAM_ADMIN_USER_IDS=")[0]
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "456789,999999")
    assert read_env(file).admin_user_ids == (456789,)
    atomic_private(file, original)
    with pytest.raises(ConfigError):
        read_env(file)
    for value, expected in (
        ("456789", (456789,)),
        ("123, 456789, 456,123", (123, 456789, 456)),
    ):
        atomic_private(file, original + f"TELEGRAM_ADMIN_USER_IDS={value}\n".encode())
        assert read_env(file).admin_user_ids == expected
    for value in (
        "",
        "0",
        "-7",
        "@name",
        "123,",
        "1,,2",
        "${TELEGRAM_OWNER_ID}",
        f"456789,{2**63}",
        "456789,0",
    ):
        atomic_private(file, original + f"TELEGRAM_ADMIN_USER_IDS={value}\n".encode())
        with pytest.raises(ConfigError):
            read_env(file)


def test_obsolete_fields_are_rejected_and_ids_have_sqlite_bounds(tmp_path: Path) -> None:
    file = tmp_path / ".env"
    env(file)
    atomic_private(file, file.read_bytes() + b"TELEGRAM_ALLOWED_USER_IDS=123\n")
    with pytest.raises(ConfigError):
        read_env(file)
    env(file, str(2**63))
    with pytest.raises(ConfigError):
        read_env(file)
    env(file, str(2**63 - 1))
    assert read_env(file).admin_user_ids == (2**63 - 1,)
    atomic_private(file, file.read_bytes() + b"TELEGRAM_OWNER_ID=456789\n")
    with pytest.raises(ConfigError):
        read_env(file)
