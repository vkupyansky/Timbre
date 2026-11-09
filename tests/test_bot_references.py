import asyncio
import shutil
from typing import BinaryIO
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, Update, Voice

from shift.protocol import PREVIEW_TEXT
from shift.store import ReferenceChange, Store
from tests.test_bot import bot as bot
from tests.test_bot import make_ogg, make_reference, telegram_message
from tests.test_bot_panel import PanelChat
from tests.test_bot_panel import panel_store as panel_store


@pytest.fixture
async def reference_chat(
    bot: Bot, panel_store: Store, monkeypatch: pytest.MonkeyPatch
) -> PanelChat:
    source = panel_store.data_dir / "sample-upload.wav"
    await make_reference(source)
    content = await asyncio.to_thread(source.read_bytes)
    await asyncio.to_thread(
        (panel_store.data_dir / "profiles/anna/reference.wav").write_bytes, content
    )

    async def download(_: object, destination: BinaryIO, **options: object) -> None:
        destination.write(content)

    monkeypatch.setattr(bot, "download", AsyncMock(side_effect=download))
    monkeypatch.setattr(bot, "send_voice", AsyncMock(return_value=telegram_message()))
    chat = PanelChat(bot, panel_store, monkeypatch)
    await chat.message("/start")
    await chat.callback("card:anna")
    return chat


def callbacks(chat: PanelChat) -> set[str]:
    assert chat.current is not None
    user_id = chat.current[0]
    anchor = chat.store.panel_id(user_id)
    assert anchor is not None
    keyboard = chat.messages[(user_id, anchor)][1]
    assert isinstance(keyboard, InlineKeyboardMarkup)
    return {button.callback_data or "" for row in keyboard.inline_keyboard for button in row}


async def upload_sample(chat: PanelChat) -> None:
    message = telegram_message(
        voice=Voice(file_id="reference-upload", file_unique_id="reference-upload", duration=16)
    )
    await chat.dispatcher.feed_update(
        chat.bot, Update(update_id=next(chat.updates), message=message)
    )


async def finish_candidate(chat: PanelChat) -> ReferenceChange:
    job = chat.store.claim()
    assert job is not None and job.kind == "reference" and job.profile is not None
    prepared = chat.store.job_dir(job.id) / "prepared"
    await asyncio.to_thread(prepared.mkdir)
    texts: list[str] = []
    for index, sample in enumerate(job.profile.reference_samples):
        name = "reference.wav" if index == 0 else f"reference-{index + 1}.wav"
        await asyncio.to_thread(
            shutil.copyfile, chat.store.data_dir / sample.reference_audio, prepared / name
        )
        texts.append(sample.reference_text)
    if job.text == "add":
        await asyncio.to_thread(
            shutil.copyfile,
            chat.store.job_dir(job.id) / "input.audio",
            prepared / f"reference-{len(texts) + 1}.wav",
        )
        texts.append("Второй образец того же голоса.")
    await make_ogg(prepared / "example.ogg")
    await asyncio.to_thread(chat.store.finish_reference, job.id, job.lease, prepared, texts)
    await chat.controller.refresh_statuses()
    change = chat.store.reference_change(7, job.profile.id)
    assert change is not None and change.stage == "ready"
    return change


async def test_reference_upload_queues_candidate_without_changing_current_voice_or_blocking_speech(
    reference_chat: PanelChat,
) -> None:
    chat = reference_chat
    original = chat.store.selected_profile(7)
    await chat.button("Образцы · 1 из 3")
    await chat.button("+ Добавить образец")
    draft = chat.store.profile_draft(7)
    assert draft is not None and draft.stage == "reference"
    assert "15–20 секунд" in chat.text and "anna" in chat.text
    await upload_sample(chat)
    change = chat.store.reference_change(7, "anna")
    assert change is not None and change.stage == "processing" and change.candidate_count == 2
    assert chat.store.profile_draft(7) is None
    assert chat.store.selected_profile(7) == original
    assert "Образец в очереди." in chat.text
    assert not any(action.startswith("sampleapply:") for action in callbacks(chat))
    await chat.message("Обычное сообщение, пока пример готовится.")
    assert {job.kind for job in chat.store.list_jobs(7)} == {"reference", "tts"}
    await chat.callback("card:anna")
    await chat.button("Темп")
    settings = chat.store.panel_content(7)
    job = chat.store.claim()
    assert job is not None and job.kind == "reference"
    await chat.controller.refresh_statuses()
    assert settings is not None and chat.main_text() == settings[0]
    assert chat.job_message(job.id)[0] == "Готовлю пример…"
    await chat.message("/start")
    assert chat.store.reference_change(7, "anna") == change
    assert chat.store.selected_profile(7) == original
    assert chat.send.await_count == 4
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100, 101}


@pytest.mark.parametrize("exit_action", ["start", "cancel", "discard"])
async def test_leaving_reference_upload_discards_only_the_candidate(
    reference_chat: PanelChat, exit_action: str
) -> None:
    chat = reference_chat
    original = chat.store.selected_profile(7)
    await chat.callback("sampleadd:anna")
    change = chat.store.reference_change(7, "anna")
    assert change is not None and change.stage == "upload"
    if exit_action == "start":
        await chat.message("/start")
    elif exit_action == "cancel":
        await chat.button("Отмена")
    else:
        await chat.callback("samples:anna")
        await chat.button("Оставить прежний")
    assert chat.store.reference_change(7, "anna") is None
    assert chat.store.profile_draft(7) is None
    assert chat.store.selected_profile(7) == original
    assert not chat.store.list_jobs(7)
    assert (chat.store.data_dir / "profiles/anna/reference.wav").exists()


@pytest.mark.parametrize(
    "action",
    [
        "samples:anna",
        "sample:anna:0",
        "sampleadd:anna",
        "sampleremove:anna:0",
        "samplelisten:anna:0",
        f"samplepreview:{'a' * 32}",
        f"sampleapply:{'a' * 32}",
        f"samplediscard:{'a' * 32}",
    ],
)
async def test_guest_cannot_access_reference_sources_or_change_candidates(
    reference_chat: PanelChat, action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = reference_chat
    read_sample = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr(chat.controller, "send_reference_audio", read_sample)
    monkeypatch.setattr(chat.bot, "send_voice", send)
    await chat.message("/start", user=8)
    await chat.callback("card:anna", user=8)
    assert "samples:anna" not in callbacks(chat)
    await chat.callback(action, user=8)
    assert "недоступн" in chat.text
    assert chat.store.reference_change(7, "anna") is None
    assert not chat.store.list_jobs(8)
    read_sample.assert_not_awaited()
    send.assert_not_awaited()


async def test_existing_pending_candidate_is_reopened_instead_of_overwritten(
    reference_chat: PanelChat,
) -> None:
    chat = reference_chat
    await chat.callback("sampleadd:anna")
    first = chat.store.reference_change(7, "anna")
    assert first is not None
    await chat.callback("sampleadd:anna")
    assert chat.store.reference_change(7, "anna") == first
    assert "resume" in callbacks(chat)
    await chat.button("Продолжить добавление")
    await upload_sample(chat)
    queued = chat.store.reference_change(7, "anna")
    assert queued is not None and queued.stage == "processing"
    await chat.callback("sampleadd:anna")
    await chat.callback("sampleremove:anna:0")
    assert chat.store.reference_change(7, "anna") == queued
    assert len(chat.store.list_jobs(7)) == 1
    assert chat.send.await_count == 2


async def test_rejected_reference_keeps_upload_and_explains_retry_beside_the_recording(
    reference_chat: PanelChat, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = reference_chat
    await chat.callback("sampleadd:anna")
    change = chat.store.reference_change(7, "anna")

    async def invalid_audio(_: object, destination: BinaryIO, **options: object) -> None:
        destination.write(b"not an audio file")

    monkeypatch.setattr(chat.bot, "download", AsyncMock(side_effect=invalid_audio))
    await upload_sample(chat)
    assert chat.store.reference_change(7, "anna") == change
    assert change is not None and f"draftcancel:{change.id}" in callbacks(chat)
    assert "Не удалось прочитать аудиофайл" in chat.text
    assert not chat.store.list_jobs(7)
    assert not list((chat.store.data_dir / "incoming").iterdir())
    assert chat.send.await_count == 2


async def test_last_sample_cannot_be_removed_even_by_forged_callback(
    reference_chat: PanelChat,
) -> None:
    chat = reference_chat
    original = chat.store.selected_profile(7)
    await chat.callback("sample:anna:0")
    assert "sampleremove:anna:0" not in callbacks(chat)
    await chat.callback("sampleremove:anna:0")
    assert chat.store.selected_profile(7) == original
    assert chat.store.reference_change(7, "anna") is None
    assert not chat.store.list_jobs(7)


@pytest.mark.parametrize("uncertain", [False, True])
async def test_listening_to_reference_sends_only_audio_and_cleans_conversion(
    reference_chat: PanelChat, monkeypatch: pytest.MonkeyPatch, uncertain: bool
) -> None:
    chat = reference_chat
    source = chat.store.reference_sample_path(7, "anna", 0)
    original = source.read_bytes()
    send = AsyncMock(
        side_effect=TimeoutError() if uncertain else None, return_value=telegram_message()
    )
    monkeypatch.setattr(chat.bot, "send_voice", send)
    await chat.callback("sample:anna:0")
    await chat.button("▶ Прослушать")
    send.assert_awaited_once()
    assert send.call_args.kwargs["caption"] == "anna · Образец 1"
    assert source.read_bytes() == original
    assert not list((chat.store.data_dir / "incoming").iterdir())
    assert not chat.store.list_jobs(7)
    if uncertain:
        assert "Telegram не подтвердил" in chat.text
    chat.send.assert_awaited_once()
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100}


@pytest.mark.parametrize("decision", ["apply", "discard"])
async def test_ready_candidate_offers_same_text_comparison_and_requires_explicit_decision(
    reference_chat: PanelChat, decision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = reference_chat
    original = chat.store.selected_profile(7)
    assert original is not None
    await chat.callback("sampleadd:anna")
    await upload_sample(chat)
    change = await finish_candidate(chat)
    assert chat.store.selected_profile(7) == original
    assert f"sampleapply:{change.id}" in callbacks(chat)
    assert "Пример с 2 образцами готов" in chat.text
    assert await chat.controller.deliver_once()
    await chat.controller.refresh_statuses()
    assert f"sampleapply:{change.id}" in callbacks(chat)
    send = AsyncMock(return_value=telegram_message())
    monkeypatch.setattr(chat.bot, "send_voice", send)
    await chat.button("▶ Новый пример")
    send.assert_awaited_once()
    assert send.call_args.kwargs["caption"] == "anna · Новый пример"
    await chat.button("▶ Прежний")
    old_example = chat.store.claim()
    assert old_example is not None and old_example.kind == "tts"
    assert old_example.text == PREVIEW_TEXT
    assert old_example.profile is not None and len(old_example.profile.reference_samples) == 1
    assert chat.store.selected_profile(7) == original
    if decision == "apply":
        # Settings changed during the comparison must survive applying the references.
        chat.store.update_voice_settings(7, "anna", speed=0.9, changes={"seed": 123})
        await chat.button("Применить")
        changed = chat.store.selected_profile(7)
        assert changed is not None and changed.id == original.id and changed.name == original.name
        assert len(changed.reference_samples) == 2 and changed.speed == 0.9 and changed.seed == 123
        assert "Образцы обновлены" in chat.text
    else:
        await chat.button("Оставить прежний")
        assert chat.store.selected_profile(7) == original
        assert "Оставлен прежний голос" in chat.text
    assert chat.store.reference_change(7, "anna") is None
    assert not (chat.store.data_dir / "reference-changes" / change.id).exists()
    assert old_example.profile.reference_text == original.reference_text
    assert chat.send.await_count == 3
    assert {call.kwargs["message_id"] for call in chat.edit.call_args_list} == {100, 101}


async def test_reference_removal_is_previewed_and_applied_without_changing_own_selection(
    reference_chat: PanelChat,
) -> None:
    chat = reference_chat
    await chat.callback("sampleadd:anna")
    await upload_sample(chat)
    await finish_candidate(chat)
    await chat.button("Применить")
    profile = chat.store.selected_profile(7)
    assert profile is not None and len(profile.reference_samples) == 2
    chat.store.select_profile(7, "maria")
    await chat.button("Образец 1")
    await chat.button("Удалить образец")
    assert (
        len(next(item for item in chat.store.profiles(7) if item.id == "anna").reference_samples)
        == 2
    )
    assert not any(action.startswith("sampleapply:") for action in callbacks(chat))
    change = await finish_candidate(chat)
    assert change.candidate_count == 1 and "1 образцом" in chat.text
    await chat.button("Применить")
    current = next(item for item in chat.store.profiles(7) if item.id == "anna")
    assert len(current.reference_samples) == 1
    assert current.reference_text == profile.extra_references[0].reference_text
    selected = chat.store.selected_profile(7)
    assert selected is not None and selected.id == "maria"
    await chat.button("Образец 1")
    assert "sampleremove:anna:0" not in callbacks(chat)
    assert chat.send.await_count == 3


async def test_ready_candidate_survives_start_and_store_restart(reference_chat: PanelChat) -> None:
    chat = reference_chat
    await chat.callback("sampleadd:anna")
    await upload_sample(chat)
    change = await finish_candidate(chat)
    reopened = Store(chat.store.db, chat.store.data_dir, owner_id=7)
    reopened.initialize()
    chat.controller.store = reopened
    await chat.message("/start")
    assert reopened.reference_change(7, "anna") == change
    await chat.callback("card:anna")
    await chat.button("Образцы · 1 из 3")
    assert f"samplepreview:{change.id}" in callbacks(chat)
    assert f"sampleapply:{change.id}" in callbacks(chat)
    assert chat.send.await_count == 3
    assert chat.store.panel_id(7) == 102


async def test_stale_reference_candidate_explains_retry_without_overwriting_profile(
    reference_chat: PanelChat,
) -> None:
    chat = reference_chat
    await chat.callback("sampleadd:anna")
    await upload_sample(chat)
    change = await finish_candidate(chat)
    reference = chat.store.reference_sample_path(7, "anna", 0)
    reference.write_bytes(b"a newer reference replaced this sample")
    await chat.button("Применить")
    assert "Образцы уже изменились" in chat.text and "подготовьте пример заново" in chat.text
    assert reference.read_bytes() == b"a newer reference replaced this sample"
    assert chat.store.reference_change(7, "anna") == change
    await chat.button("Оставить прежний")
    assert chat.store.reference_change(7, "anna") is None
    assert chat.send.await_count == 2
