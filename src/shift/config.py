"""One canonical source per setting; worker never reads the Telegram credential."""

import fcntl
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator


class ConfigError(ValueError):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    telegram_token: SecretStr
    admin_user_ids: tuple[int, ...] = Field(min_length=1, repr=False)

    @model_validator(mode="after")
    def validate_administrators(self) -> "Settings":
        if any(
            type(user_id) is not int or not 0 < user_id < 2**63 for user_id in self.admin_user_ids
        ):
            raise ValueError("Administrators must use positive Telegram IDs")
        return self


def private_dir(path: Path) -> None:
    if path.is_symlink():
        raise ConfigError("Каталог настроек не должен быть символической ссылкой.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid():
        raise ConfigError("Каталог настроек принадлежит другому пользователю.")
    path.chmod(0o700)


def atomic_private(path: Path, data: bytes) -> None:
    private_dir(path.parent)
    if path.is_symlink():
        raise ConfigError("Файл настроек не должен быть символической ссылкой.")
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as file:
            os.fchmod(file.fileno(), 0o600)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_private(path: Path, *, check_owner: bool = True) -> str:
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_mode & 0o077:
            raise ConfigError("Настройки требуют обычного файла с правами 0600.")
        if check_owner and details.st_uid != os.getuid():
            raise ConfigError("Настройки принадлежат другому пользователю.")
        return path.read_text().strip()
    except OSError as exc:
        raise ConfigError("Не найден защищённый файл конфигурации или служебный ключ.") from exc


def read_env(path: Path) -> Settings:
    read_private(path)
    values = dotenv_values(path, interpolate=False)
    required = {"TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_USER_IDS"}
    if not required.issubset(values) or any(not values[key] for key in required):
        raise ConfigError("Заполните TELEGRAM_BOT_TOKEN и TELEGRAM_ADMIN_USER_IDS в .env.")
    if set(values) - required:
        raise ConfigError("В .env есть неизвестные параметры. Проверьте названия полей.")
    token = values["TELEGRAM_BOT_TOKEN"]
    if token is None:
        raise ConfigError("Заполните TELEGRAM_BOT_TOKEN в .env.")
    validate_bot_token(token)
    extra_ids = values["TELEGRAM_ADMIN_USER_IDS"]
    if extra_ids is None:
        raise ConfigError("Укажите ID администраторов через запятую.")
    parts = [part.strip() for part in extra_ids.split(",")] if extra_ids.strip() else []
    if any(not re.fullmatch(r"[0-9]+", part) or not 0 < int(part) < 2**63 for part in parts):
        raise ConfigError("TELEGRAM_ADMIN_USER_IDS: нужны положительные числовые ID через запятую.")
    try:
        return Settings(
            telegram_token=SecretStr(token),
            admin_user_ids=tuple(dict.fromkeys(int(part) for part in parts)),
        )
    except ValidationError as error:
        raise ConfigError("Проверьте список администраторов в .env.") from error


def initialize(state: Path) -> None:
    for path in (
        state,
        state / "secrets",
        state / "data",
        state / "data/profiles",
        state / "data/jobs",
        state / "work",
        state / "logs",
    ):
        private_dir(path)
    ensure_worker_token(state)


def ensure_worker_token(state: Path) -> None:
    private_dir(state / "secrets")
    target = state / "secrets/worker.token"
    with (state / "secrets/initialize.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            read_worker_token(state)
        else:
            atomic_private(target, secrets.token_urlsafe(48).encode())


def read_worker_token(state: Path, *, check_owner: bool = True) -> str:
    token = read_private(state / "secrets/worker.token", check_owner=check_owner)
    if not re.fullmatch(r"[A-Za-z0-9_-]{64}", token):
        raise ConfigError("Внутренний ключ повреждён; автоматическая замена запрещена.")
    return token


def validate_bot_token(token: str) -> str:
    if not re.fullmatch(r"[0-9]{5,20}:[A-Za-z0-9_-]{30,100}", token):
        raise ConfigError("Неверный формат токена BotFather.")
    return token
