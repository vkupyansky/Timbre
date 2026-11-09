import asyncio
import secrets
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramRetryAfter
from aiogram.methods import SendVoice
from aiogram.types import (
    Audio,
    CallbackQuery,
    Chat,
    Document,
    Message,
    ReplyKeyboardRemove,
    Update,
    User,
    Voice,
)

from shift.bot import BotController, LimitedBuffer
from shift.protocol import MAX_AUDIO_BYTES, PREVIEW_TEXT
from shift.store import NoProfile, Store
from shift.voice_controls import CUES
from tests.test_store import add_profile


@pytest.fixture
async def bot(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Bot]:
    # Generated syntax-only credential, never a real external token.
    instance = Bot(token="123:" + secrets.token_urlsafe(30))
    monkeypatch.setattr(instance, "edit_message_text", AsyncMock(return_value=True))
    monkeypatch.setattr(instance, "send_message", AsyncMock(return_value=telegram_message()))
    monkeypatch.setattr(instance, "delete_message", AsyncMock(return_value=True))
    monkeypatch.setattr(instance.session, "make_request", AsyncMock(return_value=True))
    yield instance
    await instance.session.close()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = Store(
        tmp_path / "db.sqlite3", tmp_path, owner_id=7, admin_user_ids=frozenset({7, 8, 9})
    )
    result.initialize()
    add_profile(tmp_path, "anna", db=result.db)
    return result


def telegram_message(
    user_id: int = 7,
    chat_id: int = 7,
    *,
    text: str | None = None,
    voice: Voice | None = None,
) -> Message:
    return Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="Owner"),
        text=text,
        voice=voice,
    )


async def make_ogg(path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=0.2",
        "-c:a",
        "libopus",
        str(path),
    )
    assert await process.wait() == 0


async def make_reference(path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=16",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(path),
    )
    assert await process.wait() == 0


async def ready_result(store: Store) -> str:
    store.enqueue(1, 7, "tts", text="Один и тот же текст")
    job = store.claim()
    assert job is not None
    result = store.data_dir / "generated.ogg"
    await make_ogg(result)
    store.finish(job.id, job.lease, result, duration_ms=200)
    return job.id


async def test_mismatched_private_chats_refused_before_download(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr(bot, "download", download)
    monkeypatch.setattr(bot, "send_message", send)
    controller = BotController(bot, store)
    voice = Voice(file_id="opaque", file_unique_id="opaque", duration=1)
    for user, chat in [(8, 7), (7, 9)]:
        message = telegram_message(user, chat, voice=voice)
        await controller.on_message(message, Update(update_id=1, message=message))
    download.assert_not_called()
    send.assert_not_called()
    assert store.list_jobs(7) == []


async def test_real_dispatcher_preserves_text_deduplicates_and_rejects_extra_commands(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bot, "send_message", AsyncMock(return_value=telegram_message()))
    controller = BotController(bot, store)
    dispatcher = Dispatcher()
    dispatcher.message.register(controller.on_message)
    await dispatcher.feed_update(bot, Update(update_id=0, message=telegram_message(text="/start")))
    text = "Я был у Анны в 12:30. Хорошо?"
    update = Update(update_id=1, message=telegram_message(text=text))
    await dispatcher.feed_update(bot, update)
    await dispatcher.feed_update(bot, update)
    for number, label in enumerate(["/start", "/help", "/state", "/anything", "  "], 2):
        await dispatcher.feed_update(
            bot, Update(update_id=number, message=telegram_message(text=label))
        )
    assert len(store.list_jobs(7)) == 1
    job = store.claim()
    assert job is not None and job.text == text


async def test_voice_actual_content_validated_not_telegram_metadata(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def download(_: object, destination: BinaryIO, **options: object) -> None:
        destination.write(b"not audio despite Telegram metadata")

    send = AsyncMock()
    edit = AsyncMock()
    store.set_panel(7, 1)
    monkeypatch.setattr(bot, "download", AsyncMock(side_effect=download))
    monkeypatch.setattr(bot, "send_message", send)
    monkeypatch.setattr(bot, "edit_message_text", edit)
    message = telegram_message(voice=Voice(file_id="id", file_unique_id="id", duration=1))
    await BotController(bot, store).on_message(message, Update(update_id=1, message=message))
    assert store.list_jobs(7) == []
    assert not list((store.data_dir / "incoming").iterdir())
    assert "Не удалось прочитать" in send.call_args.args[1]
    assert send.call_args.kwargs["reply_parameters"].message_id == message.message_id
    edit.assert_not_awaited()


async def test_timeout_is_uncertain_and_retry_reuses_stored_result(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = await ready_result(store)
    send_voice = AsyncMock(side_effect=TimeoutError)
    monkeypatch.setattr(bot, "send_voice", send_voice)
    controller = BotController(bot, store)
    assert await controller.deliver_once()
    assert store.list_jobs(7)[0].status == "uncertain"
    assert not await controller.deliver_once()
    send_voice.assert_awaited_once()
    assert store.retry_delivery(job_id, 7)
    send_voice.side_effect = None
    send_voice.return_value = telegram_message()
    assert await controller.deliver_once()
    assert store.list_jobs(7)[0].status == "sent"
    assert store.claim() is None
    assert send_voice.call_count == 2
    kwargs = send_voice.call_args.kwargs
    assert kwargs.get("caption") is None and "protect_content" not in kwargs
    assert isinstance(kwargs["reply_markup"], ReplyKeyboardRemove)


async def test_retry_after_is_bounded(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await ready_result(store)
    send_voice = AsyncMock(
        side_effect=TelegramRetryAfter(
            method=SendVoice(chat_id=7, voice="opaque"),
            message="rate limited",
            retry_after=61,
        )
    )
    monkeypatch.setattr(bot, "send_voice", send_voice)
    assert await BotController(bot, store).deliver_once()
    send_voice.assert_awaited_once()
    assert store.list_jobs(7)[0].failure == "delivery_error"


async def test_callbacks_authorized_and_cancellation_not_premature(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store.enqueue(1, 7, "tts", text="Тест")
    job = store.claim()
    assert job is not None
    answer = AsyncMock(return_value=True)
    store.set_panel(7, 1)
    assert store.claim_status(job.id, 7)
    assert store.attach_status(job.id, 7, 2, "running:0")
    monkeypatch.setattr(bot.session, "make_request", answer)
    controller = BotController(bot, store)
    for user_id in (8, 7):
        callback = CallbackQuery(
            id="callback",
            from_user=User(id=user_id, is_bot=False, first_name="Test"),
            chat_instance="instance",
            message=telegram_message().model_copy(update={"message_id": 2}),
            data=f"cancel:{job.id}",
        ).as_(bot)
        await controller.on_callback(callback)
        assert store.list_jobs(7)[0].status == "running"
        assert store.list_jobs(7)[0].cancel_requested is (user_id == 7)
    answer.assert_awaited_once()
    store.confirm_cancel(job.id, job.lease)
    assert store.list_jobs(7)[0].status == "cancelled"


async def test_status_message_is_updated_without_chat_spam(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send_message = AsyncMock(
        side_effect=[telegram_message(), telegram_message().model_copy(update={"message_id": 2})]
    )
    edit_message = AsyncMock()
    monkeypatch.setattr(bot, "send_message", send_message)
    monkeypatch.setattr(bot, "edit_message_text", edit_message)
    controller = BotController(bot, store)
    start = telegram_message(text="/start")
    await controller.on_message(start, Update(update_id=0, message=start))
    message = telegram_message(text="Тест состояния")
    await controller.on_message(message, Update(update_id=1, message=message))
    job = store.claim()
    assert job is not None
    await controller.refresh_statuses()
    assert "Готовлю голосовое…" in edit_message.call_args.kwargs["text"]
    store.request_cancel(job.id, 7)
    await controller.refresh_statuses()
    assert "Останавливаю" in edit_message.call_args.kwargs["text"]
    store.confirm_cancel(job.id, job.lease)
    await controller.refresh_statuses()
    saved = store.panel_content(7)
    assert saved is not None
    assert edit_message.call_args.kwargs["text"] == "Отменено."
    assert edit_message.call_args.kwargs["message_id"] == 2
    assert "Озвуч" not in saved[0]
    final_text = edit_message.call_args.kwargs["text"]
    await controller.refresh_statuses()
    assert edit_message.call_args.kwargs["text"] == final_text
    assert send_message.await_count == 2


def test_download_buffer_enforces_actual_size() -> None:
    with LimitedBuffer() as destination:
        destination.write(b"x" * MAX_AUDIO_BYTES)
        with pytest.raises(ValueError):
            destination.write(b"x")


@pytest.mark.parametrize("user_id", [7, 8])
@pytest.mark.parametrize("media_kind", ["voice", "audio", "document"])
async def test_profile_add_is_authorized_persistent_and_not_tts(
    bot: Bot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    media_kind: str,
    user_id: int,
) -> None:
    store = Store(
        tmp_path / "queue.sqlite3",
        tmp_path,
        owner_id=7,
        admin_user_ids=frozenset({7, 8, 9}),
    )
    store.initialize()
    reference = tmp_path / "reference.wav"
    await make_reference(reference)
    content = reference.read_bytes()

    async def download(_: object, destination: BinaryIO, **options: object) -> None:
        destination.write(content)

    send_message = AsyncMock(
        side_effect=[
            telegram_message(user_id, user_id),
            telegram_message(user_id, user_id).model_copy(update={"message_id": 2}),
        ]
    )
    edit_message = AsyncMock()
    download_mock = AsyncMock(side_effect=download)
    monkeypatch.setattr(bot, "download", download_mock)
    monkeypatch.setattr(bot, "send_message", send_message)
    monkeypatch.setattr(bot, "edit_message_text", edit_message)
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    controller = BotController(bot, store)
    dispatcher = Dispatcher()
    dispatcher.message.register(controller.on_message)
    dispatcher.callback_query.register(controller.on_callback)
    await dispatcher.feed_update(
        bot, Update(update_id=1, message=telegram_message(user_id, user_id, text="/start"))
    )
    await controller.show_profiles(user_id)
    keyboard = edit_message.call_args.kwargs["reply_markup"]
    assert any(
        button.text == "+ Добавить голос" for row in keyboard.inline_keyboard for button in row
    )
    for sender_id, update_id in ((9, 2), (user_id, 3)):
        callback = CallbackQuery(
            id=f"add-{sender_id}",
            from_user=User(id=sender_id, is_bot=False, first_name="Test"),
            chat_instance="test",
            message=telegram_message(user_id, user_id),
            data="add",
        )
        await dispatcher.feed_update(bot, Update(update_id=update_id, callback_query=callback))
        assert (store.profile_draft(user_id) is not None) is (sender_id == user_id)
    media = {
        "document": Document(file_id="file", file_unique_id="file", file_name="../../outside.wav"),
        "audio": Audio(file_id="file", file_unique_id="file", duration=16),
        "voice": Voice(file_id="file", file_unique_id="file", duration=16),
    }[media_kind]
    upload = telegram_message(user_id, user_id).model_copy(update={media_kind: media})
    await dispatcher.feed_update(bot, Update(update_id=4, message=upload))
    await dispatcher.feed_update(bot, Update(update_id=4, message=upload))
    download_mock.assert_awaited_once()
    assert store.list_jobs(user_id) == []
    reopened = Store(store.db, tmp_path, owner_id=7, admin_user_ids=store.admin_user_ids)
    reopened.initialize()
    reopened.recover()
    draft = reopened.profile_draft(user_id)
    assert draft is not None and draft.stage == "name"
    controller = BotController(bot, reopened)
    name = telegram_message(user_id, user_id, text="Анна после перезапуска")
    await controller.on_message(name, Update(update_id=5, message=name))
    await controller.on_message(name, Update(update_id=5, message=name))
    jobs = reopened.list_jobs(user_id)
    assert len(jobs) == 1 and jobs[0].kind == "profile"
    assert reopened.profiles(user_id) == []
    saved_panel = reopened.panel_content(user_id)
    assert saved_panel is not None and saved_panel[0].startswith("<b>Timbre</b>")
    assert "name" not in saved_panel[1] and "draftcancel" not in saved_panel[1]
    job = reopened.claim()
    assert job is not None and job.profile is None and job.text == name.text
    assert (reopened.job_dir(job.id) / "input.audio").read_bytes() == content
    assert not (tmp_path.parent / "outside.wav").exists()
    assert send_message.await_count == 2
    assert {call.kwargs["message_id"] for call in edit_message.call_args_list} == {1}


async def test_bot_rename_and_delete_need_active_owner_confirmation(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bot, "send_message", AsyncMock(return_value=telegram_message()))
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    store.set_panel(7, 1)
    controller = BotController(bot, store)

    async def click(data: str, identifier: str, owner: int = 7) -> None:
        callback = CallbackQuery(
            id=identifier,
            from_user=User(id=owner, is_bot=False, first_name="Test"),
            chat_instance="test",
            message=telegram_message(),
            data=data,
        ).as_(bot)
        await controller.on_callback(callback)

    await click("rename:anna", "rename")
    name = telegram_message(text="Анна новая")
    await controller.on_message(name, Update(update_id=10, message=name))
    selected = store.selected_profile(7)
    assert selected is not None and selected.name == "Анна новая"
    await click("delete:anna", "prepare-delete")
    draft = store.profile_draft(7)
    assert draft is not None and draft.stage == "delete"
    assert store.profiles(7)
    await click(f"deleteyes:{draft.id}", "foreign-delete", owner=8)
    assert store.profiles(7)
    await click(f"deleteyes:{draft.id}", "own-delete")
    assert store.profiles(7) == [] and store.selected_profile(7) is None
    await click(f"deleteyes:{draft.id}", "own-delete")
    assert store.profiles(7) == []


@pytest.mark.parametrize(
    ("signal", "reason"),
    [
        ("sine=frequency=440:duration=2", "слишком короткий"),
        ("aevalsrc=0:d=16", "слишком тихая"),
        ("aevalsrc=0.9999:d=16", "перегружен"),
    ],
)
async def test_profile_rejection_explains_actual_audio_problem(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    signal: str,
    reason: str,
) -> None:
    source = store.data_dir / "rejected.wav"
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        signal,
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(source),
    )
    assert await process.wait() == 0
    content = source.read_bytes()

    async def download(_: object, destination: BinaryIO, **options: object) -> None:
        destination.write(content)

    send = AsyncMock(return_value=telegram_message())
    edit = AsyncMock()
    store.set_panel(7, 1)
    monkeypatch.setattr(bot, "send_message", send)
    monkeypatch.setattr(bot, "edit_message_text", edit)
    monkeypatch.setattr(bot, "download", AsyncMock(side_effect=download))
    store.begin_profile_draft(7)
    message = telegram_message(voice=Voice(file_id="file", file_unique_id="file", duration=16))
    await BotController(bot, store).on_message(message, Update(update_id=900, message=message))
    assert reason in send.call_args.args[1]
    assert send.call_args.kwargs["reply_parameters"].message_id == message.message_id
    edit.assert_not_awaited()
    draft = store.profile_draft(7)
    assert draft is not None and draft.stage == "upload"
    assert store.list_jobs(7) == []
    assert not list((store.data_dir / "incoming").iterdir())


async def test_start_names_current_voice(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = AsyncMock(return_value=telegram_message())
    edit = AsyncMock()
    monkeypatch.setattr(bot, "send_message", send)
    monkeypatch.setattr(bot, "edit_message_text", edit)
    message = telegram_message(text="/start")
    await BotController(bot, store).on_message(message, Update(update_id=50, message=message))
    assert "anna" in send.call_args.args[1]
    edit.assert_not_awaited()


@pytest.mark.parametrize("text", ["Помощь", "Настройки", "Состояние"])
async def test_legacy_keyboard_words_are_ordinary_speech(
    bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    add_profile(store.data_dir, "guest", user_id=8, db=store.db)
    send = AsyncMock(return_value=telegram_message(8, 8))
    monkeypatch.setattr(bot, "send_message", send)
    controller = BotController(bot, store)
    start = telegram_message(8, 8, text="/start")
    await controller.on_message(start, Update(update_id=51, message=start))
    message = telegram_message(8, 8, text=text)
    await controller.on_message(message, Update(update_id=52, message=message))
    job = store.claim()
    assert job is not None and job.text == text
    assert store.list_jobs(8)[0].id == job.id and store.list_jobs(7) == []


async def test_user_generates_with_own_selected_voice_and_full_controls(
    bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    send = AsyncMock(return_value=telegram_message(8, 8))
    edit = AsyncMock()
    store.set_panel(8, 1)
    monkeypatch.setattr(bot, "send_message", send)
    monkeypatch.setattr(bot, "edit_message_text", edit)
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    controller = BotController(bot, store)
    add_profile(store.data_dir, "maria", user_id=8, db=store.db)
    callback = CallbackQuery(
        id="guest-select",
        from_user=User(id=8, is_bot=False, first_name="Guest"),
        chat_instance="test",
        message=telegram_message(8, 8),
        data="card:maria",
    ).as_(bot)
    await controller.on_callback(callback)
    keyboard = edit.call_args.kwargs["reply_markup"]
    apply = next(
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if "Применить" in button.text
    )
    await controller.on_callback(
        callback.model_copy(update={"id": "guest-apply", "data": apply}).as_(bot)
    )
    selected = store.selected_profile(8)
    assert selected is not None and selected.id == "maria"
    owner_selected = store.selected_profile(7)
    assert owner_selected is not None and owner_selected.id == "anna"
    keyboard = edit.call_args.kwargs["reply_markup"]
    actions = [button.callback_data for row in keyboard.inline_keyboard for button in row]
    assert any(action in {"preview:maria", "listen:maria"} for action in actions)
    assert {"rename:maria", "delete:maria", "samples:maria"} <= set(actions)
    message = telegram_message(8, 8, text="Привет от второго пользователя")
    await controller.on_message(message, Update(update_id=501, message=message))
    assert store.list_jobs(7) == []
    assert len(store.list_jobs(8)) == 1
    job = store.claim()
    assert job is not None and job.profile is not None and job.profile.id == "maria"
    assert job.text == message.text


async def test_user_cannot_edit_foreign_catalog_or_touch_other_jobs(
    bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bot, "send_message", AsyncMock(return_value=telegram_message(8, 8)))
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    store.set_panel(8, 1)
    controller = BotController(bot, store)
    add_profile(store.data_dir, "guest", user_id=8, db=store.db)
    owner_job = store.enqueue(601, 7, "tts", text="Текст владельца")
    guest_job = store.enqueue(602, 8, "tts", text="Текст гостя")
    assert owner_job is not None and guest_job is not None
    owner_draft = store.begin_profile_draft(7, "delete", "anna")
    for index, action in enumerate(
        [
            "rename:anna",
            "delete:anna",
            f"deleteyes:{owner_draft.id}",
            f"draftcancel:{owner_draft.id}",
            f"cancel:{owner_job}",
            f"retry:{owner_job}",
        ]
    ):
        callback = CallbackQuery(
            id=f"guest-forged-{index}",
            from_user=User(id=8, is_bot=False, first_name="Guest"),
            chat_instance="test",
            message=telegram_message(8, 8),
            data=action,
        ).as_(bot)
        await controller.on_callback(callback)
    assert store.profile_draft(8) is None
    assert store.profile_draft(7) == owner_draft
    assert store.list_jobs(7)[0].status == "queued"
    assert store.profiles(7)[0].id == "anna"
    assert store.claim_status(guest_job, 8)
    assert store.attach_status(guest_job, 8, 2, "queued:0")
    callback = CallbackQuery(
        id="guest-own-cancel",
        from_user=User(id=8, is_bot=False, first_name="Guest"),
        chat_instance="test",
        message=telegram_message(8, 8).model_copy(update={"message_id": 2}),
        data=f"cancel:{guest_job}",
    ).as_(bot)
    await controller.on_callback(callback)
    assert store.list_jobs(8)[0].status == "cancelled"
    assert store.begin_profile_draft(8).stage == "upload"


async def test_invited_user_still_requires_matching_private_chat_before_download(
    bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr(bot, "download", download)
    monkeypatch.setattr(bot, "send_message", send)
    controller = BotController(bot, store)
    voice = Voice(file_id="opaque", file_unique_id="opaque", duration=1)
    wrong_chat = telegram_message(8, 7, voice=voice)
    group = telegram_message(8, 8, voice=voice).model_copy(
        update={"chat": Chat(id=8, type="group")}
    )
    bot_message = telegram_message(9, 9, voice=voice).model_copy(
        update={"from_user": User(id=9, is_bot=True, first_name="Bot")}
    )
    for update_id, message in enumerate([wrong_chat, group, bot_message], 700):
        await controller.on_message(message, Update(update_id=update_id, message=message))
    download.assert_not_called()
    send.assert_not_called()
    assert store.list_jobs(8) == []


async def test_public_user_receives_only_their_result(
    bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_profile(store.data_dir, "guest", user_id=8, db=store.db)
    job_id = store.enqueue(801, 8, "tts", text="Новый пользователь")
    job = store.claim()
    assert job_id is not None and job is not None
    result = store.data_dir / "saved.ogg"
    await make_ogg(result)
    store.finish(job.id, job.lease, result, duration_ms=200)
    send = AsyncMock(return_value=telegram_message(8, 8))
    monkeypatch.setattr(bot, "send_voice", send)
    assert await BotController(bot, store).deliver_once()
    assert store.list_jobs(8)[0].status == "sent"
    assert send.call_args.args[0] == 8
    assert store.list_jobs(7) == []


async def test_settings_and_previews_are_private_to_each_profile_owner(
    bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    send = AsyncMock(return_value=telegram_message())
    edit = AsyncMock()
    monkeypatch.setattr(bot, "send_message", send)
    monkeypatch.setattr(bot, "edit_message_text", edit)
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    controller = BotController(bot, store)
    store.set_panel(7, 1)
    store.set_panel(8, 1)
    await controller.show_voice_settings(7, "anna")
    assert "Нейтральн" in edit.call_args.kwargs["text"]
    actions = [
        button.callback_data
        for row in edit.call_args.kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    assert "cc:anna:e:0" in actions and "cs:anna" in actions
    assert "listen:anna" in actions and "reset:anna" in actions

    async def click(data: str, identifier: str, user: int = 7) -> None:
        callback = CallbackQuery(
            id=identifier,
            from_user=User(id=user, is_bot=False, first_name="Test"),
            chat_instance="test",
            message=telegram_message(user, user),
            data=data,
        ).as_(bot)
        await controller.on_callback(callback)

    happy = next(cue.id for cue in CUES if cue.tag == "delight")
    sad = next(cue.id for cue in CUES if cue.tag == "sad")
    await click(f"cg:anna:{happy}", "choose-happy")
    await click("speed:anna:0.9", "choose-slower")
    profile = store.selected_profile(7)
    assert profile is not None and profile.native_tags == ("delight",) and profile.speed == 0.9
    assert profile.native_tags == ("delight",)
    for number, action in enumerate((f"cg:anna:{sad}", "speed:anna:1.1", "reset:anna")):
        await click(action, f"guest-forged-setting-{number}", user=8)
    assert store.selected_profile(7) == profile
    with pytest.raises(NoProfile):
        await controller.show_voice_settings(8, "anna")
    await click("listen:anna", "guest-foreign-listen", user=8)
    assert store.list_jobs(8) == []
    add_profile(store.data_dir, "guest", user_id=8, db=store.db)
    await click(f"cg:guest:{sad}", "guest-own-setting", user=8)
    await controller.show_voice_settings(8, "guest")
    guest_actions = [
        button.callback_data
        for row in edit.call_args.kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    assert "listen:guest" in guest_actions and "reset:guest" in guest_actions
    await click("listen:guest", "guest-listen", user=8)
    await click("listen:guest", "guest-listen", user=8)
    assert len(store.list_jobs(8)) == 1
    job = store.claim()
    assert job is not None and job.kind == "tts" and job.text == PREVIEW_TEXT
    assert job.profile is not None and job.profile.native_tags == ("sad",)
    assert job.profile.speed == 1.0 and store.selected_profile(7) == profile
    await click("reset:anna", "owner-reset")
    reset = store.selected_profile(7)
    assert reset is not None and reset.delivery == "neutral" and reset.speed == 1.0
    assert job.profile.native_tags == ("sad",)


async def test_changed_settings_preview_generates_instead_of_sending_stale_example(
    bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    send_voice = AsyncMock(return_value=telegram_message())
    monkeypatch.setattr(bot, "send_voice", send_voice)
    monkeypatch.setattr(bot, "send_message", AsyncMock(return_value=telegram_message()))
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    store.set_panel(7, 1)
    example = store.data_dir / "profiles" / "anna" / "example.ogg"
    await make_ogg(example)
    controller = BotController(bot, store)

    async def preview(identifier: str) -> None:
        callback = CallbackQuery(
            id=identifier,
            from_user=User(id=7, is_bot=False, first_name="Test"),
            chat_instance="test",
            message=telegram_message(),
            data="listen:anna",
        ).as_(bot)
        await controller.on_callback(callback)

    await preview("neutral-example")
    send_voice.assert_not_awaited()
    neutral = store.claim()
    assert neutral is not None and neutral.profile is not None
    assert neutral.profile.effective_native_tags == ()
    store.fail(neutral.id, neutral.lease, "model_error")
    store.update_voice_settings(7, "anna", delivery="whisper")
    await preview("whisper-example")
    send_voice.assert_not_awaited()
    job = store.claim()
    assert job is not None and job.profile is not None and job.profile.native_tags == ("whisper",)


def test_speech_failures_describe_the_failed_stage(store: Store) -> None:
    for update_id, reason, expected in (
        (1001, "speech_unrecognized", "не удалось разобрать речь"),
        (1002, "speech_too_long", "больше 2000 символов"),
        (1003, "whisper_preparation_failed", "не получилось подготовить шёпот для этого голоса"),
    ):
        store.enqueue(update_id, 7, "tts", text="Проверка")
        job = store.claim()
        assert job is not None
        from pydantic import TypeAdapter

        from shift.protocol import FailureReason

        store.fail(job.id, job.lease, TypeAdapter(FailureReason).validate_python(reason))
        assert expected in BotController.failure_text(store.list_jobs(7)[0])
