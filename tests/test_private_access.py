import secrets
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Audio, CallbackQuery, Document, Update, User, Voice
from aiohttp.test_utils import TestClient, TestServer

from shift.app import RUNTIME, AppConfig, create_app
from shift.bot import BotController
from shift.store import StaleLease, Store
from tests.test_api import headers
from tests.test_bot import bot as bot
from tests.test_bot import telegram_message
from tests.test_store import add_profile


@pytest.fixture
def private_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "shift.sqlite3", tmp_path, 7, admin_user_ids=frozenset({7, 8, 9}))
    store.initialize()
    add_profile(tmp_path, "owner", db=store.db)
    return store


@pytest.mark.parametrize("user_id", [7, 8, 9])
async def test_every_configured_admin_can_generate_with_their_own_voice(
    bot: Bot, private_store: Store, user_id: int
) -> None:
    identifier = "personal-" + str(user_id)
    add_profile(private_store.data_dir, identifier, user_id=user_id, db=private_store.db)
    private_store.select_profile(user_id, identifier)
    controller = BotController(bot, private_store)
    dispatcher = Dispatcher()
    dispatcher.message.register(controller.on_message)
    message = telegram_message(user_id, user_id, text="Разрешённая озвучка")
    await dispatcher.feed_update(bot, Update(update_id=1, message=message))
    job = private_store.claim()
    assert job is not None and job.text == "Разрешённая озвучка"
    assert job.profile is not None and job.profile.name == identifier
    assert private_store.list_jobs(user_id)[0].id == job.id
    assert private_store.account(user_id).is_admin
    assert private_store.account(user_id).reserved_ms == 0


@pytest.mark.parametrize("kind", ["start", "text", "voice", "audio", "document"])
@pytest.mark.parametrize("existing_account", [False, True])
async def test_nonadmin_messages_have_no_database_or_telegram_side_effects(
    bot: Bot,
    private_store: Store,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    existing_account: bool,
) -> None:
    if existing_account:
        private_store.update_account_user(10, "previous_user", "Previous User")
        add_profile(private_store.data_dir, "previous", user_id=10, db=private_store.db)
    with sqlite3.connect(private_store.db) as connection:
        before = connection.execute("SELECT * FROM accounts ORDER BY user_id").fetchall()
    update_user = Mock(wraps=private_store.update_account_user)
    enqueue = Mock(wraps=private_store.enqueue)
    download = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr(private_store, "update_account_user", update_user)
    monkeypatch.setattr(private_store, "enqueue", enqueue)
    monkeypatch.setattr(bot, "download", download)
    monkeypatch.setattr(bot, "send_message", send)
    controller = BotController(bot, private_store)
    dispatcher = Dispatcher()
    dispatcher.message.register(controller.on_message)
    message = telegram_message(10, 10)
    if kind in ("start", "text"):
        message = message.model_copy(update={"text": "/start" if kind == "start" else "Текст"})
    elif kind == "voice":
        message = message.model_copy(
            update={"voice": Voice(file_id="sample", file_unique_id="sample", duration=16)}
        )
    elif kind == "audio":
        message = message.model_copy(
            update={"audio": Audio(file_id="sample", file_unique_id="sample", duration=16)}
        )
    else:
        message = message.model_copy(
            update={
                "document": Document(
                    file_id="sample", file_unique_id="sample", file_name="reference.wav"
                )
            }
        )
    await dispatcher.feed_update(bot, Update(update_id=1, message=message))
    update_user.assert_not_called()
    enqueue.assert_not_called()
    download.assert_not_awaited()
    send.assert_not_awaited()
    with sqlite3.connect(private_store.db) as connection:
        assert connection.execute("SELECT * FROM accounts ORDER BY user_id").fetchall() == before
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM updates").fetchone()[0] == 0
    if existing_account:
        assert private_store.get_profile(10, "previous").name == "previous"


@pytest.mark.parametrize(
    "action",
    [
        "home",
        "account",
        "voices",
        "add",
        "card:previous",
        "listen:previous",
        "speed:previous:0.9",
        "samples:previous",
        "sampleadd:previous",
        "cancel:" + "a" * 32,
        "retry:" + "a" * 32,
    ],
)
async def test_nonadmin_old_buttons_cannot_reopen_or_change_anything(
    bot: Bot, private_store: Store, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    private_store.account(10)
    add_profile(private_store.data_dir, "previous", user_id=10, db=private_store.db)
    update_user = Mock(wraps=private_store.update_account_user)
    monkeypatch.setattr(private_store, "update_account_user", update_user)
    request = AsyncMock()
    monkeypatch.setattr(bot.session, "make_request", request)
    controller = BotController(bot, private_store)
    dispatcher = Dispatcher()
    dispatcher.callback_query.register(controller.on_callback)
    callback = CallbackQuery(
        id="old-button",
        from_user=User(id=10, is_bot=False, first_name="Previous"),
        chat_instance="private",
        message=telegram_message(10, 10),
        data=action,
    )
    await dispatcher.feed_update(bot, Update(update_id=1, callback_query=callback))
    update_user.assert_not_called()
    request.assert_not_awaited()
    assert private_store.profile_draft(10) is None
    assert private_store.list_jobs(10) == []
    assert private_store.get_profile(10, "previous").speed == 1.0


@pytest.mark.parametrize("stage", ["queued", "running", "ready"])
def test_cancel_old_nonadmin_jobs_preserves_accounts_profiles_and_history(
    private_store: Store, stage: str
) -> None:
    store = private_store
    store.account(10)
    add_profile(store.data_dir, "previous", user_id=10, db=store.db)
    request_id = store.enqueue(1, 10, "tts", text="Сохранённый текст прошлого запроса")
    assert request_id is not None
    job = None
    if stage != "queued":
        job = store.claim()
        assert job is not None
    if stage == "ready":
        result = store.data_dir / "old-result.ogg"
        result.write_bytes(b"previously verified output")
        assert job is not None
        store.finish(job.id, job.lease, result, duration_ms=200)
    with sqlite3.connect(store.db) as connection:
        before_accounts = connection.execute("SELECT * FROM accounts ORDER BY user_id").fetchall()
        before_profiles = connection.execute("SELECT * FROM profile_owners").fetchall()
    assert store.cancel_unauthorized_jobs() == 1
    assert store.cancel_unauthorized_jobs() == 0
    assert store.list_jobs(10)[0].status == "cancelled"
    assert store.account(10).reserved_ms == 0
    assert store.get_profile(10, "previous").name == "previous"
    with sqlite3.connect(store.db) as connection:
        accounts = connection.execute("SELECT * FROM accounts ORDER BY user_id").fetchall()
        assert accounts == before_accounts
        assert connection.execute("SELECT * FROM profile_owners").fetchall() == before_profiles
        row = connection.execute("SELECT input_text,status FROM generation_history").fetchone()
        assert row == ("Сохранённый текст прошлого запроса", "cancelled")
    if job is not None:
        with pytest.raises(StaleLease):
            store.check_lease(job.id, job.lease)


async def test_worker_claim_cannot_process_a_legacy_nonadmin_request(tmp_path: Path) -> None:
    add_profile(tmp_path, "owner")
    app = await create_app(
        AppConfig(7, tmp_path, admin_user_ids=frozenset({7, 8, 9})),
        "123:" + secrets.token_urlsafe(30),
        secrets.token_urlsafe(32),
        background=False,
    )
    async with TestClient(TestServer(app)) as client:
        store = app[RUNTIME].store
        store.account(10)
        add_profile(tmp_path, "previous", user_id=10, db=store.db)
        old = store.enqueue(1, 10, "tts", text="Не выполнять")
        admin = store.enqueue(2, 7, "tts", text="Разрешённый запрос")
        response = await client.post("/worker/claim", headers=headers(client))
        assert response.status == 200
        body = await response.json()
        assert body["job"]["id"] == admin
        assert store.list_jobs(10)[0].id == old
        assert store.list_jobs(10)[0].status == "cancelled"


async def test_startup_revokes_queued_work_without_deleting_previous_personal_data(
    private_store: Store,
) -> None:
    store = private_store
    add_profile(store.data_dir, "previous-admin", user_id=8, db=store.db)
    store.select_profile(8, "previous-admin")
    old = store.enqueue(1, 8, "tts", text="До изменения списка")
    keep = store.enqueue(2, 7, "tts", text="Текущий администратор")
    app = await create_app(
        AppConfig(7, store.data_dir, admin_user_ids=frozenset({7, 9})),
        "123:" + secrets.token_urlsafe(30),
        secrets.token_urlsafe(32),
        background=False,
    )
    app_store = app[RUNTIME].store
    async with TestClient(TestServer(app)):
        assert not app_store.allowed_user(8)
        assert app_store.get_profile(8, "previous-admin").name == "previous-admin"
        selected = app_store.selected_profile(8)
        assert selected is not None and selected.id == "previous-admin"
        assert app_store.list_jobs(8)[0].id == old
        assert app_store.list_jobs(8)[0].status == "cancelled"
        assert app_store.list_jobs(7)[0].id == keep
        assert app_store.list_jobs(7)[0].status == "queued"
        with sqlite3.connect(store.db) as connection:
            assert connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 3
            assert connection.execute("SELECT COUNT(*) FROM generation_history").fetchone()[0] == 2


async def test_background_delivery_and_statuses_do_not_contact_former_users(
    bot: Bot, private_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = private_store
    store.account(10)
    add_profile(store.data_dir, "previous", user_id=10, db=store.db)
    store.enqueue(1, 10, "tts", text="Ранее готовый результат")
    old = store.claim()
    assert old is not None
    output = store.data_dir / "old-result.ogg"
    output.write_bytes(b"previously verified output")
    store.finish(old.id, old.lease, output, duration_ms=200)
    request = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr(bot.session, "make_request", request)
    monkeypatch.setattr(bot, "send_message", send)
    controller = BotController(bot, store)
    await controller.refresh_statuses()
    assert not await controller.deliver_once()
    request.assert_not_awaited()
    send.assert_not_awaited()
    assert store.list_jobs(10)[0].status == "cancelled"
