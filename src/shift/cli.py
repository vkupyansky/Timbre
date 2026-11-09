"""Internal process entry points. Everyday operation uses Docker Compose and bot buttons."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from aiogram.exceptions import TelegramAPIError
from pydantic import ValidationError

from shift.config import ConfigError, ensure_worker_token, read_env, read_worker_token

PROJECT = Path(__file__).resolve().parents[2]


async def worker(state: Path) -> None:
    from shift.worker import run_worker

    task = asyncio.create_task(run_worker(PROJECT, state))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        pass


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description="Internal VoxShift service process")
    parser.add_argument("--state", type=Path, default=PROJECT / ".shift")
    commands = parser.add_subparsers(dest="command", required=True)
    application = commands.add_parser("app")
    application.add_argument("--container", action="store_true")
    commands.add_parser("worker")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        state = args.state.resolve()
        if args.command == "worker":
            asyncio.run(worker(state))
        else:
            from shift.app import AppConfig, run_app

            if args.container and not Path("/.dockerenv").is_file():
                raise ConfigError("Контейнерный режим доступен только внутри Docker.")
            ensure_worker_token(state)
            config = read_env(Path("/config/.env") if args.container else PROJECT / ".env")
            asyncio.run(
                run_app(
                    AppConfig(
                        owner_id=config.admin_user_ids[0],
                        admin_user_ids=frozenset(config.admin_user_ids),
                        data_dir=state / "data",
                        bind_host="0.0.0.0" if args.container else "127.0.0.1",
                    ),
                    config.telegram_token.get_secret_value(),
                    read_worker_token(state),
                )
            )
    except ConfigError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except TelegramAPIError:
        print("Telegram недоступен. Проверьте токен в .env и подключение к сети.", file=sys.stderr)
        sys.exit(1)
    except (ValidationError, OSError, RuntimeError):
        print("Сервис не готов: ошибка конфигурации, файлов или моделей.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
