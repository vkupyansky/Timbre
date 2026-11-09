"""Disposable container acceptance harness; never uses a real Telegram account."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncGenerator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import GetFile, SendMessage, SendVoice, TelegramMethod
from aiogram.methods.base import TelegramType
from aiogram.types import CallbackQuery, Chat, File, FSInputFile, Message, Update, User, Voice
from aiohttp import web
from pydantic import TypeAdapter

from shift.app import CONTROLLER, RUNTIME, AppConfig, create_app
from shift.audio import validate_enrollment_reference, validate_result, validate_voice
from shift.config import read_env, read_worker_token

ROOT = Path("/acceptance")
TEXT = "Анна, сегодня в 12:30 я был у реки. Всё правильно?"


class LocalTelegramSession(BaseSession):
    def __init__(self, owner_id: int) -> None:
        super().__init__()
        self.deliveries: list[dict[str, object]] = []
        self.message_number = 0
        self.owner_id = owner_id

    async def close(self) -> None:
        return None

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: int | None = None,  # noqa: ASYNC109 -- required BaseSession signature.
    ) -> TelegramType:
        self.message_number += 1
        result: object
        if isinstance(method, GetFile):
            filename = {"local-source": "source.ogg", "local-reference": "reference.ogg"}[
                method.file_id
            ]
            result = File(
                file_id=method.file_id,
                file_unique_id=method.file_id,
                file_path=filename,
                file_size=(ROOT / filename).stat().st_size,
            ).model_dump(mode="json")
        elif isinstance(method, (SendVoice, SendMessage)):
            result = Message(
                message_id=self.message_number,
                date=datetime.now(UTC),
                chat=Chat(id=self.owner_id, type="private"),
            ).model_dump(mode="json")
            if isinstance(method, SendVoice):
                if not isinstance(method.voice, FSInputFile):
                    raise TypeError("Expected real saved output audio")
                source = Path(method.voice.path)
                info = await validate_result(source)
                destination = ROOT / "results" / f"{source.parent.name}.ogg"
                await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
                content = await asyncio.to_thread(source.read_bytes)
                await asyncio.to_thread(destination.write_bytes, content)
                self.deliveries.append(
                    {
                        "file": str(destination),
                        "job_id": source.parent.name,
                        "duration": info.duration,
                        "codec": info.codec,
                        "caption": method.caption,
                        "protect_content": bool(method.protect_content)
                        if isinstance(method.protect_content, bool)
                        else False,
                    }
                )
        else:
            result = True
        adapter: TypeAdapter[TelegramType] = TypeAdapter(method.__returning__)
        return adapter.validate_python(result)

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,  # noqa: ASYNC109 -- required BaseSession signature.
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        filename = url.rsplit("/", 1)[-1]
        if filename not in {"source.ogg", "reference.ogg"}:
            raise ValueError("Unknown local acceptance source")
        source = await asyncio.to_thread((ROOT / filename).read_bytes)
        for index in range(0, len(source), chunk_size):
            yield source[index : index + chunk_size]


async def application() -> web.Application:
    settings = read_env(ROOT / ".env")
    owner_id = settings.admin_user_ids[0]
    session = LocalTelegramSession(owner_id)
    bot = Bot(settings.telegram_token.get_secret_value(), session=session)
    worker_token = read_worker_token(ROOT)
    app = await create_app(
        AppConfig(owner_id, ROOT / "data", admin_user_ids=frozenset(settings.admin_user_ids)),
        "",
        worker_token,
        bot=bot,
    )
    dispatcher = Dispatcher()
    dispatcher.message.register(app[CONTROLLER].on_message)
    dispatcher.callback_query.register(app[CONTROLLER].on_callback)
    next_update = 1000

    async def input_message(request: web.Request) -> web.Response:
        nonlocal next_update
        kind = request.match_info["kind"]
        if kind not in {"tts", "vc"}:
            raise web.HTTPBadRequest()
        source = ROOT / "source.ogg"
        source_duration = math.ceil((await validate_voice(source)).duration) if kind == "vc" else 0
        next_update += 1
        message = Message(
            message_id=next_update,
            date=datetime.now(UTC),
            chat=Chat(id=owner_id, type="private"),
            from_user=User(id=owner_id, is_bot=False, first_name="Acceptance"),
            text=TEXT if kind == "tts" else None,
            voice=Voice(
                file_id="local-source",
                file_unique_id="local-source",
                duration=source_duration,
                file_size=source.stat().st_size,
            )
            if kind == "vc"
            else None,
        )
        await dispatcher.feed_update(bot, Update(update_id=next_update, message=message))
        return web.json_response({"update_id": next_update, "kind": kind})

    async def add_voice(request: web.Request) -> web.Response:
        nonlocal next_update
        reference = ROOT / "reference.ogg"
        reference_info = await validate_enrollment_reference(reference)
        menu = Message(
            message_id=next_update + 1,
            date=datetime.now(UTC),
            chat=Chat(id=owner_id, type="private"),
            from_user=User(id=owner_id, is_bot=False, first_name="Acceptance"),
            text="/start",
        )
        next_update += 1
        await dispatcher.feed_update(bot, Update(update_id=next_update, message=menu))
        panel_id = await asyncio.to_thread(request.app[RUNTIME].store.panel_id, owner_id)
        if panel_id is None:
            raise web.HTTPUnprocessableEntity(text="panel was not created")
        panel = menu.model_copy(update={"message_id": panel_id, "text": None})
        for action in ("voices", "add"):
            next_update += 1
            await dispatcher.feed_update(
                bot,
                Update(
                    update_id=next_update,
                    callback_query=CallbackQuery(
                        id=f"acceptance-{action}-{next_update}",
                        from_user=User(id=owner_id, is_bot=False, first_name="Acceptance"),
                        chat_instance="acceptance",
                        message=panel,
                        data=action,
                    ),
                ),
            )
        next_update += 1
        upload = menu.model_copy(
            update={
                "message_id": next_update,
                "text": None,
                "voice": Voice(
                    file_id="local-reference",
                    file_unique_id="local-reference",
                    duration=math.ceil(reference_info.duration),
                    file_size=reference.stat().st_size,
                ),
            }
        )
        await dispatcher.feed_update(bot, Update(update_id=next_update, message=upload))
        next_update += 1
        name = menu.model_copy(update={"message_id": next_update, "text": "Вика (приёмка)"})
        await dispatcher.feed_update(bot, Update(update_id=next_update, message=name))
        jobs = await asyncio.to_thread(request.app[RUNTIME].store.list_jobs, owner_id)
        if not any(job.kind == "profile" for job in jobs):
            raise web.HTTPUnprocessableEntity(text="profile was not queued")
        return web.json_response({"kind": "profile", "queued": True})

    async def status(request: web.Request) -> web.Response:
        jobs = await asyncio.to_thread(request.app[RUNTIME].store.list_jobs, owner_id)
        profiles = await asyncio.to_thread(request.app[RUNTIME].store.profiles, owner_id)
        selected = await asyncio.to_thread(request.app[RUNTIME].store.selected_profile, owner_id)
        return web.json_response(
            {
                "jobs": [asdict(job) for job in jobs],
                "deliveries": session.deliveries,
                "text": TEXT,
                "profiles": [profile.model_dump() for profile in profiles],
                "selected_profile": selected.id if selected is not None else None,
            }
        )

    app.router.add_post("/test/input/{kind}", input_message)
    app.router.add_post("/test/add_voice", add_voice)
    app.router.add_get("/test/status", status)
    return app


if __name__ == "__main__":
    web.run_app(application(), host="0.0.0.0", port=8765, access_log=None)
