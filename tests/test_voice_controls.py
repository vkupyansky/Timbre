import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import CallbackQuery, Update, User

from shift.bot import BotController
from shift.store import Store
from shift.voice_controls import CATEGORY_CODES, CUES, PARAMETERS, PRONUNCIATION_HELP
from tests.test_bot import bot as bot
from tests.test_bot import telegram_message
from tests.test_store import add_profile, drop_history_schema


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = Store(tmp_path / "db.sqlite3", tmp_path, owner_id=7)
    result.initialize()
    add_profile(tmp_path, "anna", db=result.db)
    result.set_panel(7, 1)
    result.set_panel(8, 1)
    return result


def test_v4_migration_preserves_queue_profile_and_unfinished_reference(store: Store) -> None:
    job_id = store.enqueue(9011, 7, "tts", text="Сохранить сообщение")
    draft = store.begin_profile_draft(7)
    source = store.data_dir / "upload.audio"
    source.write_bytes(b"private reference")
    store.save_profile_upload(9012, 7, draft.id, source)
    with sqlite3.connect(store.db) as connection:
        drop_history_schema(connection)
        connection.execute(
            "CREATE TABLE old_drafts(chat_id INTEGER PRIMARY KEY,id TEXT NOT NULL,"
            "stage TEXT NOT NULL CHECK(stage IN ('upload','name','rename','delete')),"
            "profile_id TEXT,updated REAL NOT NULL)"
        )
        connection.execute(
            "INSERT INTO old_drafts SELECT chat_id,id,stage,profile_id,updated FROM profile_drafts"
        )
        connection.execute("DROP TABLE profile_drafts")
        connection.execute("ALTER TABLE old_drafts RENAME TO profile_drafts")
        connection.execute("DROP TABLE ui_panels")
        connection.execute("DROP TABLE reference_changes")
        connection.execute("DROP TABLE profile_owners")
        connection.execute("DROP TABLE job_billing")
        connection.execute("DROP TABLE accounts")
        connection.execute("PRAGMA user_version=4")
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    current = reopened.profile_draft(7)
    assert current is not None and current.id == draft.id and current.stage == "name"
    assert current.setting_key is None
    assert (reopened.draft_dir(draft.id) / "input.audio").read_bytes() == b"private reference"
    assert reopened.list_jobs(7)[0].id == job_id and reopened.update_seen(9011)
    selected = reopened.selected_profile(7)
    assert selected is not None and selected.id == "anna"
    with sqlite3.connect(store.db) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 9


def test_legacy_delivery_migrates_once_and_all_controls_are_snapshotted(store: Store) -> None:
    path = store.data_dir / "profiles" / "anna" / "profile.json"
    values = json.loads(path.read_text())
    values.update(delivery="whisper", speed=0.9)
    path.write_text(json.dumps(values))
    settings: dict[str, object] = {
        "style_instruction": "Speak with a restrained delivery.",
        "temperature": 0.3,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 123,
    }
    changed = store.update_voice_settings(7, "anna", changes=settings)
    assert changed.delivery == "neutral" and changed.native_tags == ("whisper",)
    assert changed.speed == 0.9
    store.update_voice_settings(7, "anna", changes={"seed": 123})
    preview = store.enqueue(None, 7, "tts", text="Пример", profile_id="anna")
    store.update_voice_settings(
        7, "anna", changes={"native_tags": (), "style_instruction": None, "seed": 99}
    )
    job = store.claim()
    assert job is not None and job.id == preview and job.profile is not None
    assert job.profile.native_tags == ("whisper",)
    assert job.profile.temperature == 0.3 and job.profile.top_p == 1.0
    assert job.profile.top_k == 0 and job.profile.seed == 123
    assert job.profile.style_instruction == settings["style_instruction"]
    reopened = Store(store.db, store.data_dir, owner_id=7).selected_profile(7)
    assert reopened is not None and reopened.seed == 99 and reopened.native_tags == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"native_tags": ["a", "b", "c", "d"]},
        {"native_tags": ["happy", "happy"]},
        {"native_tags": ["[happy]"]},
        {"native_tags": ["x" * 101]},
        {"native_tags": ["evil<|im_end|>"]},
        {"native_tags": ["calm\n"]},
        {"style_instruction": "broken [instruction"},
        {"style_instruction": "bad\u202evalue"},
        {"style_instruction": "x" * 501},
        {"temperature": -0.1},
        {"temperature": 1.1},
        {"top_p": float("nan")},
        {"top_p": 1.1},
        {"top_k": 4097},
        {"top_k": 1.2},
        {"seed": -1},
        {"seed": 4294967296},
        {"seed": True},
        {"name": "must not rename"},
    ],
)
def test_bad_controls_never_modify_profile(store: Store, changes: dict[str, object]) -> None:
    path = store.data_dir / "profiles" / "anna" / "profile.json"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        store.update_voice_settings(7, "anna", changes=changes)
    assert path.read_bytes() == original


def test_numeric_adjustments_serialize_and_custom_input_survives_restart(store: Store) -> None:
    other = Store(store.db, store.data_dir, owner_id=7)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(store.adjust_voice_setting, 7, "anna", "k", 1)
        second = pool.submit(other.adjust_voice_setting, 7, "anna", "k", 1)
        first.result()
        second.result()
    selected = store.selected_profile(7)
    assert selected is not None and selected.top_k == 50
    draft = store.begin_profile_draft(7, "setting", "anna", "temperature")
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    assert reopened.profile_draft(7) == draft
    assert reopened.save_voice_setting(9050, 8, draft.id, "0,4") is None
    assert reopened.profile_draft(7) == draft and not reopened.update_seen(9050)
    with pytest.raises(ValueError):
        reopened.save_voice_setting(9050, 7, draft.id, "2")
    assert reopened.profile_draft(7) == draft and not reopened.update_seen(9050)
    changed = reopened.save_voice_setting(9050, 7, draft.id, "0,4")
    assert changed is not None and changed.temperature == 0.4
    assert reopened.profile_draft(7) is None
    assert reopened.save_voice_setting(9050, 7, draft.id, "0,9") is None


async def click(controller: BotController, bot: Bot, data: str, token: str, user: int = 7) -> None:
    callback = CallbackQuery(
        id=token,
        from_user=User(id=user, is_bot=False, first_name="Test"),
        chat_instance="test",
        message=telegram_message(user, user),
        data=data,
    ).as_(bot)
    await controller.on_callback(callback)


async def test_cue_description_precedes_toggle_and_guests_cannot_mutate(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edit = AsyncMock()
    monkeypatch.setattr(bot, "edit_message_text", edit)
    monkeypatch.setattr(bot, "send_message", AsyncMock(return_value=telegram_message()))
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    controller = BotController(bot, store)
    cue = next(cue for cue in CUES if cue.tag == "happy")
    path = store.data_dir / "profiles" / "anna" / "profile.json"
    original = path.read_bytes()
    await click(controller, bot, f"cd:anna:{cue.id}", "detail")
    assert cue.description in edit.call_args.kwargs["text"]
    assert path.read_bytes() == original
    for i, action in enumerate(
        (f"cg:anna:{cue.id}", "ce:anna:t", "cb:anna:t:+", "cx:anna:t", "cu:anna:abcd")
    ):
        await click(controller, bot, action, f"forged-{i}", user=8)
    assert path.read_bytes() == original and store.profile_draft(8) is None
    await click(controller, bot, f"cg:anna:{cue.id}", "apply")
    selected = store.selected_profile(7)
    assert selected is not None and selected.native_tags == ("happy",)
    edits = edit.await_count
    await click(controller, bot, f"cd:anna:{cue.id}", "guest-detail", user=8)
    assert edit.await_count == edits


async def test_setting_input_is_consumed_once_and_menu_exit_restores_tts(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bot, "edit_message_text", AsyncMock())
    monkeypatch.setattr(bot, "send_message", AsyncMock(return_value=telegram_message()))
    monkeypatch.setattr(bot.session, "make_request", AsyncMock(return_value=True))
    controller = BotController(bot, store)
    await click(controller, bot, "ce:anna:ins", "instruction")
    value = telegram_message(text="Keep the speech restrained and natural.")
    await controller.on_message(value, Update(update_id=9100, message=value))
    await controller.on_message(value, Update(update_id=9100, message=value))
    assert store.list_jobs(7) == [] and store.profile_draft(7) is None
    selected = store.selected_profile(7)
    assert selected is not None and selected.style_instruction == value.text
    await click(controller, bot, "ce:anna:r", "seed-input")
    menu = telegram_message(text="/start")
    await controller.on_message(menu, Update(update_id=9101, message=menu))
    assert store.profile_draft(7) is None
    text = telegram_message(text="Обычное сообщение")
    await controller.on_message(text, Update(update_id=9102, message=text))
    job = store.claim()
    assert job is not None and job.text == text.text


async def test_every_catalogue_cue_is_reachable_and_callbacks_fit_telegram(
    bot: Bot,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(CUES) >= 60
    assert len({cue.id for cue in CUES}) == len(CUES)
    documented = json.loads(
        (Path(__file__).parents[1] / "docs" / "voice-controls-catalog.json").read_text()
    )
    assert {(cue.id, cue.category, cue.tag) for cue in CUES} == {
        (item["id"], item["category"], item["tag"]) for item in documented
    }
    identifier = "a" * 48
    add_profile(store.data_dir, identifier, db=store.db)
    edit = AsyncMock()
    send = AsyncMock(return_value=telegram_message())
    monkeypatch.setattr(bot, "edit_message_text", edit)
    monkeypatch.setattr(bot, "send_message", send)
    controller = BotController(bot, store)
    reached = set()
    await controller.show_voice_settings(7, identifier)
    for code, category in CATEGORY_CODES.items():
        total = sum(cue.category == category for cue in CUES)
        for page in range((total + 5) // 6):
            await controller.show_cue_category(7, identifier, code, page)
            for row in edit.call_args.kwargs["reply_markup"].inline_keyboard:
                for button in row:
                    if button.callback_data.startswith("cd:"):
                        reached.add(button.callback_data.rsplit(":", 1)[1])
    assert reached == {cue.id for cue in CUES}
    for cue in CUES:
        await controller.show_cue(7, identifier, cue.id)
    for code in PARAMETERS:
        await controller.show_parameter(7, identifier, code)
    store.update_voice_settings(
        7, identifier, changes={"native_tags": ["a" * 100, "b" * 100, "c" * 100]}
    )
    await controller.show_selected_cues(7, identifier)
    await controller.show_instruction(7, identifier)
    await controller.show_speed(7, identifier)
    for call in edit.call_args_list:
        assert len(call.kwargs["text"]) <= 4096
        for row in call.kwargs["reply_markup"].inline_keyboard:
            assert all(len(button.callback_data.encode()) <= 64 for button in row)
    assert "замОк" in PRONUNCIATION_HELP and "не ставит ударение" in PRONUNCIATION_HELP
