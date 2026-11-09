"""Job notifications remain independent of navigation and other users' requests."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.methods import EditMessageText

from shift.bot import BotController
from shift.store import Store
from tests.test_bot import bot as bot
from tests.test_bot import make_ogg, telegram_message
from tests.test_bot_panel import PanelChat
from tests.test_bot_panel import panel_store as panel_store
from tests.test_store import add_profile


def status_anchor(store: Store, job_id: str, user: int = 7) -> int:
    entry = store.status_entry(job_id, user)
    assert entry is not None and entry[1] is not None
    return entry[1]


async def test_parallel_jobs_keep_distinct_statuses_and_cancel_only_their_own_request(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_profile(panel_store.data_dir, "guest", user_id=8, db=panel_store.db)
    panel_store.select_profile(8, "guest")
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    owner_card = chat.main_text()
    await chat.message("Первый текст")
    first = panel_store.list_jobs(7)[0]
    await chat.message("Второй текст")
    second = next(job for job in panel_store.list_jobs(7) if job.id != first.id)
    await chat.message("/start", user=8)
    await chat.message("Текст другого человека", user=8)
    guest = panel_store.list_jobs(8)[0]
    first_anchor = status_anchor(panel_store, first.id)
    second_anchor = status_anchor(panel_store, second.id)
    guest_anchor = status_anchor(panel_store, guest.id, 8)
    assert len({first_anchor, second_anchor, guest_anchor}) == 3
    assert chat.main_text() == owner_card
    assert panel_store.claim() is not None
    await chat.controller.refresh_statuses()
    assert chat.job_message(first.id)[0] == "Готовлю голосовое…"
    assert chat.job_message(second.id)[0] == "В очереди."
    assert chat.job_message(guest.id, 8)[0] == "В очереди."
    await chat.callback(f"cancel:{first.id}", message_id=second_anchor)
    await chat.callback(f"cancel:{first.id}", user=8, message_id=guest_anchor)
    assert not panel_store.list_jobs(7)[0].cancel_requested
    await chat.callback(f"cancel:{second.id}", message_id=second_anchor)
    assert chat.job_message(second.id)[0] == "Отменено."
    assert chat.job_message(first.id)[0] == "Готовлю голосовое…"
    assert chat.main_text() == owner_card
    assert chat.send.await_count == 5


async def test_start_and_restart_preserve_job_controls_without_cancelling_current_input(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.message("Текст")
    job = panel_store.list_jobs(7)[0]
    anchor = status_anchor(panel_store, job.id)
    await chat.message("/start")
    assert (7, anchor) in chat.messages
    assert panel_store.panel_id(7) != anchor
    await chat.callback("rename:anna")
    draft = panel_store.profile_draft(7)
    assert draft is not None
    reopened = Store(
        panel_store.db,
        panel_store.data_dir,
        owner_id=7,
        admin_user_ids=panel_store.admin_user_ids,
    )
    reopened.initialize()
    chat.controller.store = reopened
    restarted = BotController(bot, reopened)
    await restarted.refresh_statuses()
    assert status_anchor(reopened, job.id) == anchor
    await chat.callback(f"cancel:{job.id}", message_id=anchor)
    assert reopened.profile_draft(7) == draft
    assert chat.job_message(job.id)[0] == "Отменено."
    assert "назван" in chat.main_text().lower()
    assert chat.send.await_count == 3


async def test_status_reloads_state_after_another_callback_completed_cancellation(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.message("Текст")
    stale_job = panel_store.list_jobs(7)[0]
    anchor = status_anchor(panel_store, stale_job.id)
    await chat.callback(f"cancel:{stale_job.id}", message_id=anchor)
    assert panel_store.status_updates() == []
    # A polling snapshot taken before the callback must never restore "В очереди".
    assert await chat.controller.refresh_job_status(stale_job, anchor)
    assert chat.job_message(stale_job.id)[0] == "Отменено."
    assert panel_store.status_updates() == []


@pytest.mark.parametrize("failure", ["network", "rate_limit", "deleted"])
async def test_status_edit_failures_do_not_create_messages_or_touch_navigation(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    card = chat.main_text()
    await chat.message("Текст")
    job = panel_store.claim()
    assert job is not None
    anchor = status_anchor(panel_store, job.id)
    method = EditMessageText(text="Тест", chat_id=7, message_id=anchor)
    if failure == "deleted":
        del chat.messages[(7, anchor)]
    elif failure == "network":
        chat.edit.side_effect = TelegramNetworkError(method=method, message="unavailable")
    else:
        chat.edit.side_effect = TelegramRetryAfter(
            method=method, message="retry later", retry_after=123
        )
    now = 1000.0
    monkeypatch.setattr("shift.bot.time.monotonic", lambda: now)
    await chat.controller.refresh_statuses()
    previous_edits = chat.edit.await_count
    chat.edit.side_effect = chat._edit
    await chat.controller.refresh_statuses()
    assert chat.edit.await_count == previous_edits
    if failure == "rate_limit":
        now += 122
        await chat.controller.refresh_statuses()
        assert chat.edit.await_count == previous_edits
    now += 200
    await chat.controller.refresh_statuses()
    if failure == "deleted":
        entry = panel_store.status_entry(job.id, 7)
        assert entry is not None and entry[1] is None
        assert panel_store.status_updates() == []
    else:
        assert chat.job_message(job.id)[0] == "Готовлю голосовое…"
        assert status_anchor(panel_store, job.id) == anchor
    assert chat.main_text() == card
    assert chat.send.await_count == 2


async def test_concurrent_announcements_create_one_reply_for_the_job(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    job_id = panel_store.enqueue(1000, 7, "tts", text="Проверка")
    assert job_id is not None
    await asyncio.gather(
        chat.controller.announce_job(7, job_id, reply_to_message_id=41),
        chat.controller.announce_job(7, job_id, reply_to_message_id=41),
        chat.controller.refresh_statuses(),
    )
    chat.send.assert_awaited_once()
    assert chat.send.call_args.kwargs["reply_parameters"].message_id == 41
    assert status_anchor(panel_store, job_id) == 100
    assert panel_store.panel_id(7) is None


async def test_uncertain_status_send_is_not_repeated_after_restart_and_delivery_updates_balance(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_profile(panel_store.data_dir, "guest", user_id=8, db=panel_store.db)
    panel_store.select_profile(8, "guest")
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=8)
    initial_home = chat.main_text(8)
    chat.send.side_effect = TimeoutError()
    await chat.message("Проверка", user=8)
    job = panel_store.claim()
    assert job is not None
    reopened = Store(
        panel_store.db,
        panel_store.data_dir,
        owner_id=7,
        admin_user_ids=panel_store.admin_user_ids,
    )
    reopened.initialize()
    restarted = BotController(bot, reopened)
    await restarted.announce_job(8, job.id)
    await restarted.refresh_statuses()
    assert chat.send.await_count == 2
    result = panel_store.data_dir / "uncertain-status.ogg"
    await make_ogg(result)
    reopened.finish(job.id, job.lease, result, duration_ms=200)
    monkeypatch.setattr(bot, "send_voice", AsyncMock(return_value=telegram_message(8, 8)))
    assert await restarted.deliver_once()
    assert reopened.list_jobs(8)[0].status == "sent"
    assert chat.main_text(8) == initial_home
    assert "Безлимит" in chat.main_text(8)
    assert chat.send.await_count == 2


async def test_terminal_first_announcement_also_refreshes_the_home_balance(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_profile(panel_store.data_dir, "guest", user_id=8, db=panel_store.db)
    panel_store.select_profile(8, "guest")
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=8)
    job_id = panel_store.enqueue(1000, 8, "tts", text="Проверка")
    assert job_id is not None
    job = panel_store.claim()
    assert job is not None
    result = panel_store.data_dir / "ready-before-status.ogg"
    await make_ogg(result)
    panel_store.finish(job.id, job.lease, result, duration_ms=200)
    await chat.controller.refresh_statuses()
    assert chat.job_message(job.id, 8)[0] == "Отправляю голосовое…"
    assert "Безлимит" in chat.main_text(8)
    assert chat.send.await_count == 2


async def test_status_storage_error_during_delivery_does_not_prevent_audio_sending(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.message("Проверка")
    job = panel_store.claim()
    assert job is not None
    result = panel_store.data_dir / "status-error.ogg"
    await make_ogg(result)
    panel_store.finish(job.id, job.lease, result, duration_ms=200)

    def inaccessible_status(job_id: str, chat_id: int) -> None:
        raise sqlite3.OperationalError("busy")

    monkeypatch.setattr(panel_store, "status_entry", inaccessible_status)
    send_voice = AsyncMock(return_value=telegram_message())
    monkeypatch.setattr(bot, "send_voice", send_voice)
    assert await chat.controller.deliver_once()
    send_voice.assert_awaited_once()
    assert panel_store.list_jobs(7)[0].status == "sent"


async def test_delivery_retry_uses_original_status_and_charges_only_once(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_profile(panel_store.data_dir, "guest", user_id=8, db=panel_store.db)
    panel_store.select_profile(8, "guest")
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start", user=8)
    await chat.message("Проверка", user=8)
    job = panel_store.claim()
    assert job is not None
    anchor = status_anchor(panel_store, job.id, 8)
    result = panel_store.data_dir / "retry-status.ogg"
    await make_ogg(result)
    panel_store.finish(job.id, job.lease, result, duration_ms=200)
    send_voice = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(bot, "send_voice", send_voice)
    assert await chat.controller.deliver_once()
    await chat.controller.refresh_statuses()
    await chat.message("/start", user=8)
    await chat.callback(f"retry:{job.id}", user=8, message_id=anchor)
    assert panel_store.list_jobs(8)[0].status == "ready"
    send_voice.side_effect = None
    send_voice.return_value = telegram_message(8, 8)
    assert await chat.controller.deliver_once()
    await chat.controller.refresh_statuses()
    assert chat.job_message(job.id, 8)[0] == "Готово."
    assert status_anchor(panel_store, job.id, 8) == anchor
    assert panel_store.account(8).balance_ms == 0
    assert send_voice.await_count == 2
    assert chat.send.await_count == 3


async def test_profile_preparation_leaves_the_name_prompt_and_updates_home_when_ready(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("add")
    draft = panel_store.profile_draft(7)
    assert draft is not None
    source = panel_store.data_dir / "enrollment.audio"
    source.write_bytes(b"validated source recording")
    assert panel_store.save_profile_upload(1000, 7, draft.id, source)
    current = panel_store.profile_draft(7)
    assert current is not None
    await chat.controller.show_draft(7, current)
    assert "назван" in chat.main_text().lower()
    await chat.message("Новый голос")
    job = panel_store.claim()
    assert job is not None and job.kind == "profile"
    assert chat.main_text().startswith("<b>Timbre</b>")
    assert "anna" in chat.main_text() and "Новый голос" not in chat.main_text()
    assert chat.job_message(job.id)[0] == "Образец в очереди."
    prepared = panel_store.job_dir(job.id) / "prepared"
    prepared.mkdir()
    (prepared / "reference.wav").write_bytes(b"validated reference fixture")
    (prepared / "example.ogg").write_bytes(b"validated preview fixture")
    profile = panel_store.finish_profile(job.id, job.lease, prepared, "Текст образца.")
    await chat.controller.refresh_statuses()
    assert panel_store.selected_profile(7) == profile
    assert "Новый голос" in chat.main_text()
    assert "назван" not in chat.main_text().lower()
    assert chat.job_message(job.id)[0] == "Отправляю голосовое…"
    assert chat.send.await_count == 2


async def test_invalid_job_callback_does_not_replace_main_navigation(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    card = chat.main_text()
    await chat.callback("cancel:../../invalid")
    await chat.callback("retry:../../invalid")
    assert chat.main_text() == card
    assert "больше не действует" in str(chat.answer.call_args)
    chat.send.assert_awaited_once()
