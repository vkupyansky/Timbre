"""Exercise persistent panel navigation through the real Telegram dispatcher."""

import sqlite3
from itertools import count
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter
from aiogram.methods import DeleteMessage, EditMessageText, SendMessage, SendVoice
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
    User,
    Voice,
)

from shift.bot import BotController
from shift.store import Store
from tests.test_bot import bot as bot
from tests.test_bot import make_ogg, telegram_message
from tests.test_store import add_profile


class PanelChat:
    def __init__(self, bot: Bot, store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bot = bot
        self.store = store
        self.ids = count(100)
        self.updates = count(1)
        self.messages: dict[tuple[int, int], tuple[str, object]] = {}
        self.current: tuple[int, int] | None = None
        self.send = AsyncMock(side_effect=self._send)
        self.edit = AsyncMock(side_effect=self._edit)
        self.delete = AsyncMock(side_effect=self._delete)
        self.answer = AsyncMock(return_value=True)
        monkeypatch.setattr(bot, "send_message", self.send)
        monkeypatch.setattr(bot, "edit_message_text", self.edit)
        monkeypatch.setattr(bot, "delete_message", self.delete)
        monkeypatch.setattr(bot, "answer_callback_query", self.answer)
        monkeypatch.setattr(bot.session, "make_request", self.answer)
        self.controller = BotController(bot, store)
        self.dispatcher = Dispatcher()
        self.dispatcher.message.register(self.controller.on_message)
        self.dispatcher.callback_query.register(self.controller.on_callback)

    async def _send(self, chat_id: int, text: str, **kwargs: object) -> object:
        message_id = next(self.ids)
        self.current = (chat_id, message_id)
        self.messages[self.current] = (text, kwargs.get("reply_markup"))
        return telegram_message(chat_id, chat_id, text=text).model_copy(
            update={"message_id": message_id}
        )

    async def _edit(
        self,
        *,
        text: str,
        chat_id: int,
        message_id: int,
        reply_markup: InlineKeyboardMarkup,
        parse_mode: str | None = None,
    ) -> bool:
        key = (chat_id, message_id)
        method = EditMessageText(
            text=text, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup
        )
        if key not in self.messages:
            raise TelegramBadRequest(method=method, message="message to edit not found")
        # Telegram refuses to edit a message sent with ReplyKeyboardRemove.
        # An unconditional successful AsyncMock hid this live API restriction.
        if isinstance(self.messages[key][1], ReplyKeyboardRemove):
            raise TelegramBadRequest(method=method, message="message can't be edited")
        self.messages[key] = (text, reply_markup)
        self.current = key
        return True

    async def _delete(self, chat_id: int, message_id: int) -> bool:
        # Deleting an old panel is safe only after the new anchor is durable.
        assert self.store.panel_id(chat_id) != message_id
        key = (chat_id, message_id)
        if key not in self.messages:
            raise TelegramBadRequest(
                method=DeleteMessage(chat_id=chat_id, message_id=message_id),
                message="message to delete not found",
            )
        del self.messages[key]
        return True

    @property
    def text(self) -> str:
        assert self.current is not None
        return self.messages[self.current][0]

    @property
    def keyboard(self) -> InlineKeyboardMarkup:
        assert self.current is not None
        keyboard = self.messages[self.current][1]
        assert isinstance(keyboard, InlineKeyboardMarkup)
        return keyboard

    async def message(self, text: str, user: int = 7) -> None:
        await self.dispatcher.feed_update(
            self.bot,
            Update(
                update_id=next(self.updates),
                message=telegram_message(user, user, text=text),
            ),
        )

    async def callback(self, data: str, user: int = 7, message_id: int | None = None) -> None:
        identifier = next(self.updates)
        callback = CallbackQuery(
            id=f"panel-{identifier}",
            from_user=User(id=user, is_bot=False, first_name="Test"),
            chat_instance="private-test",
            message=telegram_message(user, user).model_copy(
                update={"message_id": message_id or self.store.panel_id(user) or 1}
            ),
            data=data,
        )
        await self.dispatcher.feed_update(
            self.bot, Update(update_id=identifier, callback_query=callback)
        )

    def main_text(self, user: int = 7) -> str:
        anchor = self.store.panel_id(user)
        assert anchor is not None
        return self.messages[(user, anchor)][0]

    def job_message(self, job_id: str, user: int = 7) -> tuple[str, InlineKeyboardMarkup]:
        entry = self.store.status_entry(job_id, user)
        assert entry is not None and entry[1] is not None
        text, keyboard = self.messages[(user, entry[1])]
        assert isinstance(keyboard, InlineKeyboardMarkup)
        return text, keyboard

    async def button(self, label: str, user: int = 7) -> None:
        anchor = self.store.panel_id(user)
        assert anchor is not None
        self.current = (user, anchor)
        choices = [
            button
            for row in self.keyboard.inline_keyboard
            for button in row
            if button.text.removeprefix("✓ ") == label
        ]
        assert len(choices) == 1, (label, self.keyboard)
        assert choices[0].callback_data is not None
        await self.callback(choices[0].callback_data, user=user)


@pytest.fixture
def panel_store(tmp_path: Path) -> Store:
    store = Store(
        tmp_path / "panel.sqlite3", tmp_path, owner_id=7, admin_user_ids=frozenset({7, 8, 9})
    )
    store.initialize()
    for profile in ("anna", "bella", "maria", "zoya"):
        add_profile(tmp_path, profile, db=store.db)
    store.select_profile(7, "anna")
    return store


async def test_first_start_delivers_working_inline_navigation_without_editing_the_message(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    chat.send.assert_awaited_once()
    chat.edit.assert_not_awaited()
    assert any(
        button.text == "Мои голоса" and button.callback_data == "voices"
        for row in chat.keyboard.inline_keyboard
        for button in row
    )
    await chat.button("Мои голоса")
    chat.edit.assert_awaited_once()
    assert panel_store.panel_id(7) == 100
    assert any(
        button.callback_data == "card:anna"
        for row in chat.keyboard.inline_keyboard
        for button in row
    )
    chat.send.assert_awaited_once()


async def test_start_moves_home_down_then_only_edits_the_new_anchor(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    selected = panel_store.selected_profile(7)
    await chat.message("/start")
    assert panel_store.panel_id(7) == 101
    assert (7, 100) not in chat.messages and (7, 101) in chat.messages
    chat.delete.assert_awaited_once_with(7, 100)
    saved = panel_store.panel_content(7)
    assert saved is not None and saved[0] == chat.text
    assert isinstance(chat.send.call_args.kwargs["reply_markup"], InlineKeyboardMarkup)
    await chat.callback("profile:maria", message_id=100)
    assert panel_store.selected_profile(7) == selected
    await chat.button("Мои голоса")
    assert chat.edit.call_args.kwargs["message_id"] == 101
    assert chat.send.await_count == 2


@pytest.mark.parametrize("failure", ["network", "timeout", "rate_limited"])
async def test_failed_start_send_keeps_the_working_old_anchor_and_base(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    saved = panel_store.panel_content(7)
    method = SendMessage(chat_id=7, text="Главная")
    errors = {
        "network": TelegramNetworkError(method=method, message="connection unavailable"),
        "timeout": TimeoutError(),
        "rate_limited": TelegramRetryAfter(method=method, message="retry later", retry_after=61),
    }
    chat.send.side_effect = errors[failure]
    await chat.message("/start")
    assert panel_store.panel_id(7) == 100
    assert panel_store.panel_content(7) == saved
    assert (7, 100) in chat.messages and len(chat.messages) == 1
    assert chat.send.await_count == 2
    chat.delete.assert_not_awaited()


@pytest.mark.parametrize("failure", ["denied", "timeout"])
async def test_failed_old_panel_deletion_keeps_new_anchor_and_rejects_old_buttons(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    selected = panel_store.selected_profile(7)
    chat.delete.side_effect = (
        TelegramBadRequest(
            method=DeleteMessage(chat_id=7, message_id=100), message="message can't be deleted"
        )
        if failure == "denied"
        else TimeoutError()
    )
    await chat.message("/start")
    assert panel_store.panel_id(7) == 101
    assert (7, 100) in chat.messages and (7, 101) in chat.messages
    await chat.callback("profile:maria", message_id=100)
    assert panel_store.selected_profile(7) == selected
    await chat.button("Мои голоса")
    assert chat.edit.call_args.kwargs["message_id"] == 101
    assert chat.send.await_count == 2
    chat.delete.assert_awaited_once_with(7, 100)


async def test_failed_new_anchor_persistence_does_not_delete_the_old_panel(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    saved = panel_store.panel_content(7)

    def fail_replace(*args: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(panel_store, "replace_panel", fail_replace)
    await chat.message("/start")
    assert panel_store.panel_id(7) == 100
    assert panel_store.panel_content(7) == saved
    assert (7, 100) in chat.messages
    assert chat.send.await_count == 3
    chat.delete.assert_not_awaited()


async def test_entire_voice_management_flow_edits_one_panel(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    anchor = panel_store.panel_id(7)
    assert anchor == 100
    assert isinstance(chat.send.call_args.kwargs["reply_markup"], InlineKeyboardMarkup)
    await chat.button("Мои голоса")
    voice_rows = [
        row
        for row in chat.keyboard.inline_keyboard
        if any((button.callback_data or "").startswith("card:") for button in row)
    ]
    assert len(voice_rows) == 2 and all(len(row) == 2 for row in voice_rows)
    await chat.button("maria")
    selected = panel_store.selected_profile(7)
    assert selected is not None and selected.id == "anna"
    await chat.button("Применить")
    selected = panel_store.selected_profile(7)
    assert selected is not None and selected.id == "maria"
    await chat.button("Темп")
    assert "темп" in chat.text.lower() or "скорост" in chat.text.lower()
    await chat.callback("card:maria")
    await chat.button("Переименовать")
    draft = panel_store.profile_draft(7)
    assert draft is not None and draft.stage == "rename"
    await chat.message("Мария новая")
    selected = panel_store.selected_profile(7)
    assert selected is not None and selected.name == "Мария новая"
    await chat.callback("card:maria")
    await chat.button("Удалить")
    draft = panel_store.profile_draft(7)
    assert draft is not None and draft.stage == "delete"
    assert any(profile.id == "maria" for profile in panel_store.profiles(7))
    await chat.button("Удалить")
    assert all(profile.id != "maria" for profile in panel_store.profiles(7))
    await chat.button("+ Добавить голос")
    draft = panel_store.profile_draft(7)
    assert draft is not None and draft.stage == "upload"
    await chat.button("Отмена")
    assert panel_store.profile_draft(7) is None
    chat.send.assert_awaited_once()
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {anchor}
    assert all(
        button.text != "Состояние"
        for call in chat.edit.call_args_list
        for row in call.kwargs["reply_markup"].inline_keyboard
        for button in row
    )


async def test_message_uses_its_own_status_while_navigation_keeps_the_open_card(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.message("Короткое сообщение для озвучки")
    job = panel_store.list_jobs(7)[0]
    assert chat.job_message(job.id)[0] == "В очереди."
    assert chat.send.call_args.kwargs["reply_parameters"].message_id == 1
    assert "В очереди" not in chat.main_text()
    await chat.button("Мои голоса")
    await chat.button("anna")
    saved = panel_store.panel_content(7)
    claimed = panel_store.claim()
    assert claimed is not None and claimed.id == job.id
    await chat.controller.refresh_statuses()
    assert chat.job_message(job.id)[0] == "Готовлю голосовое…"
    assert saved is not None and chat.main_text() == saved[0]
    assert panel_store.panel_content(7) == saved
    assert chat.send.await_count == 2
    assert panel_store.panel_id(7) == 100
    entry = panel_store.status_entry(job.id, 7)
    assert entry is not None and entry[1] == 101


async def test_request_errors_stay_near_request_and_job_cancel_leaves_settings_open(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    saved = panel_store.panel_content(7)
    await chat.message("Сохрани открытую карточку")
    job = panel_store.claim()
    assert job is not None
    await chat.controller.refresh_statuses()
    await chat.message("x" * 2001)
    assert "2000" in chat.text and "anna" not in chat.text
    assert chat.send.call_args.kwargs["reply_parameters"].message_id == 1
    assert panel_store.panel_content(7) == saved
    entry = panel_store.status_entry(job.id, 7)
    assert entry is not None and entry[1] is not None
    await chat.callback(f"cancel:{job.id}", message_id=entry[1])
    assert panel_store.list_jobs(7)[0].cancel_requested
    assert chat.job_message(job.id)[0] == "Останавливаю…"
    panel_store.confirm_cancel(job.id, job.lease)
    await chat.controller.refresh_statuses()
    assert chat.job_message(job.id)[0] == "Отменено."
    assert saved is not None and chat.main_text() == saved[0]
    await chat.message("/help")
    assert "/start" in chat.text and "anna" not in chat.text
    assert panel_store.panel_content(7) == saved
    assert chat.send.await_count == 4


async def test_missing_or_old_panel_callbacks_never_change_data(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.callback("profile:maria", message_id=1)
    selected = panel_store.selected_profile(7)
    assert selected is not None and selected.id == "anna"
    assert panel_store.panel_id(7) is None
    assert "/start" in str(chat.answer.call_args)
    chat.send.assert_not_awaited()
    await chat.message("/start")
    await chat.callback("profile:maria", message_id=99)
    await chat.callback("add", message_id=99)
    assert panel_store.selected_profile(7) == selected
    assert panel_store.profile_draft(7) is None
    assert panel_store.panel_id(7) == 100
    chat.send.assert_awaited_once()


@pytest.mark.parametrize("failure", ["not_modified", "network", "timeout", "rate_limited"])
async def test_temporary_edit_failures_never_create_another_panel(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    method = EditMessageText(text="Тест", chat_id=7, message_id=100)
    errors = {
        "not_modified": TelegramBadRequest(method=method, message="message is not modified"),
        "network": TelegramNetworkError(method=method, message="connection unavailable"),
        "timeout": TimeoutError(),
        "rate_limited": TelegramRetryAfter(method=method, message="retry later", retry_after=61),
    }
    chat.edit.side_effect = errors[failure]
    await chat.controller.show_profiles(7)
    assert panel_store.panel_id(7) == 100
    chat.send.assert_awaited_once()


async def test_deleted_panel_recovers_once_on_start(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    chat.edit.side_effect = TelegramBadRequest(
        method=EditMessageText(text="Тест", chat_id=7, message_id=100),
        message="message to edit not found",
    )
    await chat.callback("voices")
    assert panel_store.panel_id(7) is None
    chat.send.assert_awaited_once()
    chat.edit.side_effect = chat._edit
    await chat.message("/start")
    assert panel_store.panel_id(7) == 101 and chat.send.await_count == 2
    await chat.message("/start")
    assert chat.send.await_count == 3 and panel_store.panel_id(7) == 102


async def test_each_user_keeps_their_anchor_after_restart(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=7)
    await chat.message("/start", user=8)
    assert panel_store.panel_id(7) == 100 and panel_store.panel_id(8) == 101
    reopened = Store(
        panel_store.db,
        panel_store.data_dir,
        owner_id=7,
        admin_user_ids=panel_store.admin_user_ids,
    )
    reopened.initialize()
    controller = BotController(bot, reopened)
    await controller.show_profiles(7)
    await controller.show_profiles(8)
    assert chat.send.await_count == 2
    assert (
        chat.edit.call_args_list[-2].kwargs["chat_id"],
        chat.edit.call_args_list[-2].kwargs["message_id"],
    ) == (7, 100)
    assert (
        chat.edit.call_args_list[-1].kwargs["chat_id"],
        chat.edit.call_args_list[-1].kwargs["message_id"],
    ) == (8, 101)


async def test_start_keeps_uploaded_reference_and_resumes_naming_in_same_panel(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("add")
    draft = panel_store.profile_draft(7)
    assert draft is not None
    source = panel_store.data_dir / "pending.audio"
    source.write_bytes(b"reference preserved across navigation")
    assert panel_store.save_profile_upload(1000, 7, draft.id, source)
    await chat.message("/start")
    current = panel_store.profile_draft(7)
    assert current is not None and current.id == draft.id and current.stage == "name"
    assert (panel_store.draft_dir(draft.id) / "input.audio").read_bytes() == source.read_bytes()
    await chat.button("Мои голоса")
    resume = [
        button.callback_data
        for row in chat.keyboard.inline_keyboard
        for button in row
        if (button.callback_data or "").startswith("resume")
    ]
    assert len(resume) == 1 and resume[0] is not None
    await chat.callback(resume[0])
    assert "назван" in chat.text.lower()
    assert chat.send.await_count == 2 and panel_store.panel_id(7) == 101
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100, 101}


async def test_successful_delivery_finishes_separate_status_and_keeps_card_editable(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    card_text = chat.text
    job_id = panel_store.enqueue(1000, 7, "tts", text="Проверка готового голосового")
    assert job_id is not None
    await chat.controller.announce_job(7, job_id)
    job = panel_store.claim()
    assert job is not None
    result = panel_store.data_dir / "sent.ogg"
    await make_ogg(result)
    panel_store.finish(job.id, job.lease, result, duration_ms=200)
    send_voice = AsyncMock(return_value=telegram_message())
    monkeypatch.setattr(bot, "send_voice", send_voice)
    assert await chat.controller.deliver_once()
    await chat.controller.refresh_statuses()
    assert panel_store.list_jobs(7)[0].status == "sent"
    assert chat.job_message(job_id)[0] == "Готово."
    assert chat.main_text() == card_text
    assert isinstance(send_voice.call_args.kwargs["reply_markup"], ReplyKeyboardRemove)
    await chat.callback("voices")
    assert panel_store.panel_id(7) == 100
    assert chat.send.await_count == 2


@pytest.mark.parametrize("action", ["rename:anna", "delete:anna", "ce:anna:r"])
async def test_start_exits_temporary_input_without_treating_next_text_as_a_setting(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback(action)
    assert panel_store.profile_draft(7) is not None
    await chat.message("/start")
    assert panel_store.profile_draft(7) is None
    await chat.message("Обычный текст для озвучки")
    job = panel_store.claim()
    assert job is not None and job.kind == "tts" and job.text == "Обычный текст для озвучки"
    assert job.profile is not None and job.profile.name == "anna" and job.profile.seed == 42
    assert chat.send.await_count == 3 and panel_store.panel_id(7) == 101


async def test_start_replaces_a_confirmed_missing_anchor_on_the_first_request(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    del chat.messages[(7, 100)]
    await chat.message("/start")
    assert panel_store.panel_id(7) == 101 and chat.send.await_count == 2
    chat.edit.assert_not_awaited()
    chat.delete.assert_awaited_once_with(7, 100)


@pytest.mark.parametrize("uncertain", [False, True])
async def test_delivery_errors_offer_retry_on_job_status_without_resending_audio(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch, uncertain: bool
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    job_id = panel_store.enqueue(1000, 7, "tts", text="Проверка отправки")
    assert job_id is not None
    await chat.controller.announce_job(7, job_id)
    job = panel_store.claim()
    assert job is not None
    result = panel_store.data_dir / "delivery.ogg"
    await make_ogg(result)
    panel_store.finish(job.id, job.lease, result, duration_ms=200)
    send_voice = AsyncMock(
        side_effect=TimeoutError()
        if uncertain
        else TelegramBadRequest(
            method=SendVoice(chat_id=7, voice="test-audio"), message="upload rejected"
        )
    )
    monkeypatch.setattr(bot, "send_voice", send_voice)
    assert await chat.controller.deliver_once()
    await chat.controller.refresh_statuses()
    assert "anna" in chat.main_text()
    assert (
        "Telegram не подтвердил" if uncertain else "Не получилось отправить"
    ) in chat.job_message(job_id)[0]
    assert panel_store.list_jobs(7)[0].status == ("uncertain" if uncertain else "failed")
    assert f"retry:{job_id}" in {
        button.callback_data for row in chat.keyboard.inline_keyboard for button in row
    }
    assert not await chat.controller.deliver_once()
    send_voice.assert_awaited_once()
    assert chat.send.await_count == 2
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100, 101}


@pytest.mark.parametrize("user_id", [7, 8, 9, 10])
async def test_only_admins_enter_and_their_personal_catalogs_remain_separate(
    bot: Bot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user_id: int
) -> None:
    store = Store(
        tmp_path / "accounts.sqlite3", tmp_path, owner_id=7, admin_user_ids=frozenset({7, 8, 9})
    )
    store.initialize()
    add_profile(tmp_path, "owner-voice", db=store.db)
    chat = PanelChat(bot, store, monkeypatch)
    await chat.message("/start", user=user_id)
    if user_id == 10:
        chat.send.assert_not_awaited()
        with sqlite3.connect(store.db) as connection:
            assert connection.execute("SELECT 1 FROM accounts WHERE user_id=10").fetchone() is None
        return
    account = store.account(user_id)
    assert account.is_admin
    assert account.balance_ms == 0
    assert "Безлимит" in chat.text
    await chat.button("Профиль", user=user_id)
    assert ("Безлимит" in chat.text) is account.is_admin
    await chat.button("← На главную", user=user_id)
    await chat.button("Мои голоса", user=user_id)
    profile_actions = {
        button.callback_data for row in chat.keyboard.inline_keyboard for button in row
    }
    assert ("card:owner-voice" in profile_actions) is (user_id == 7)
    if user_id != 7:
        await chat.callback("card:owner-voice", user=user_id)
        assert "owner-voice" not in chat.text
        await chat.callback("listen:owner-voice", user=user_id)
        assert store.list_jobs(user_id) == []
    await chat.callback("add", user=user_id)
    draft = store.profile_draft(user_id)
    assert draft is not None and draft.stage == "upload"
    assert "15–20 секунд" in chat.text
    chat.send.assert_awaited_once()
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100}


async def test_admin_result_keeps_unlimited_access_and_the_same_panel(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_profile(panel_store.data_dir, "personal", user_id=8, db=panel_store.db)
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=8)
    await chat.message("Моё первое сообщение", user=8)
    await chat.callback("account", user=8)
    assert "Безлимит" in chat.text
    job = panel_store.claim()
    assert job is not None and panel_store.list_jobs(8)[0].id == job.id
    result = panel_store.data_dir / "account-result.ogg"
    await make_ogg(result)
    panel_store.finish(job.id, job.lease, result, duration_ms=200)
    await chat.controller.refresh_statuses()
    assert "Безлимит" in chat.main_text(8)
    assert "отложена" not in chat.main_text(8)
    assert panel_store.account(8).balance_ms == 0
    assert chat.send.await_count == 2
    assert panel_store.panel_id(8) == 100


async def test_existing_exhausted_trial_is_preserved_without_public_access(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_profile(panel_store.data_dir, "personal", user_id=10, db=panel_store.db)
    with sqlite3.connect(panel_store.db) as connection:
        connection.execute("UPDATE accounts SET balance_ms=0 WHERE user_id=10")
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=10)
    await chat.message("Баланс уже закончился", user=10)
    assert chat.current is None
    chat.send.assert_not_awaited()
    assert panel_store.list_jobs(10) == []
    download = AsyncMock()
    monkeypatch.setattr(bot, "download", download)
    message = telegram_message(
        10, 10, voice=Voice(file_id="file", file_unique_id="file", duration=1)
    )
    await chat.controller.on_message(message, Update(update_id=99, message=message))
    download.assert_not_awaited()
    assert panel_store.list_jobs(10) == []
    await chat.message("/start", user=10)
    assert panel_store.account(10).balance_ms == 0


async def test_default_voice_is_ready_on_signup_and_its_settings_are_personal(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_accounts_store import create_default_template

    create_default_template(panel_store, "anna")
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=8)
    assert "Вика" in chat.text and "Начните с голоса" not in chat.text
    first = panel_store.selected_profile(8)
    assert first is not None and panel_store.is_default_voice(8, first.id)
    await chat.message("/start", user=9)
    second = panel_store.selected_profile(9)
    assert second is not None and first.id != second.id
    await chat.callback(f"speed:{first.id}:0.9", user=8)
    await chat.callback(f"card:{first.id}", user=8)
    actions = {button.callback_data for row in chat.keyboard.inline_keyboard for button in row}
    assert f"samples:{first.id}" not in actions
    assert {f"rename:{first.id}", f"delete:{first.id}", f"listen:{first.id}"} <= actions
    assert panel_store.get_profile(8, first.id).speed == 0.9
    assert panel_store.get_profile(9, second.id).speed == 1.0
    assert panel_store.get_profile(7, "anna").speed == 1.0
    await chat.callback(f"listen:{first.id}", user=8)
    assert len(panel_store.list_jobs(8)) == 1
    assert panel_store.account(8).balance_ms == 0
    await chat.callback("voices", user=8)
    await chat.button("+ Добавить голос", user=8)
    draft = panel_store.profile_draft(8)
    assert draft is not None and draft.stage == "upload"
    assert chat.send.await_count == 3
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100}


async def test_default_voice_reference_callbacks_never_expose_original_recordings(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_accounts_store import create_default_template

    create_default_template(panel_store, "anna")
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=8)
    profile = panel_store.selected_profile(8)
    assert profile is not None
    send = AsyncMock()
    monkeypatch.setattr(chat.controller, "send_reference_audio", send)
    for action in (
        f"samples:{profile.id}",
        f"sample:{profile.id}:0",
        f"sampleadd:{profile.id}",
        f"samplelisten:{profile.id}:0",
        f"sampleremove:{profile.id}:0",
        f"samplepreview:{'a' * 32}",
        f"sampleapply:{'a' * 32}",
        f"samplediscard:{'a' * 32}",
    ):
        await chat.callback(action, user=8)
        assert "недоступн" in chat.text
    send.assert_not_awaited()
    assert panel_store.profile_draft(8) is None
    assert panel_store.reference_change(8, profile.id) is None
    assert panel_store.list_jobs(8) == []
    await chat.message("/start", user=7)
    await chat.callback("card:anna", user=7)
    assert any(
        button.callback_data == "samples:anna"
        for row in chat.keyboard.inline_keyboard
        for button in row
    )


async def test_profile_shows_only_current_telegram_identity_and_updates_changes(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    user = User(id=8, is_bot=False, first_name="<Анна>", last_name="Иванова &", username="voice_42")
    message = telegram_message(8, 8, text="/start").model_copy(update={"from_user": user})
    await chat.controller.on_message(message, Update(update_id=7000, message=message))
    callback = CallbackQuery(
        id="identity-first",
        from_user=user,
        chat_instance="personal",
        message=message.model_copy(update={"message_id": panel_store.panel_id(8)}),
        data="account",
    ).as_(bot)
    await chat.controller.on_callback(callback)
    assert "&lt;Анна&gt; Иванова &amp;" in chat.text
    assert "@voice_42" in chat.text and "<code>8</code>" in chat.text
    assert "Безлимит" in chat.text and "Администратор" in chat.text
    changed = callback.model_copy(
        update={
            "id": "identity-changed",
            "from_user": User(id=8, is_bot=False, first_name="Новое имя"),
        }
    ).as_(bot)
    await chat.controller.on_callback(changed)
    assert "Новое имя" in chat.text and "Имя пользователя: Не указан" in chat.text
    assert "voice_42" not in chat.text
    account = panel_store.account(8)
    assert account.username is None and account.display_name == "Новое имя"
    assert panel_store.account(7).display_name is None
    chat.send.assert_awaited_once()
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100}
