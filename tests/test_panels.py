"""Persistent Telegram panel anchors must not alter jobs, profiles, or drafts."""

import sqlite3
from pathlib import Path

import pytest

from shift.store import Store
from tests.test_store import add_profile, drop_history_schema


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = Store(tmp_path / "queue.sqlite3", tmp_path, owner_id=7)
    result.initialize()
    return result


def test_panels_survive_restart_and_remain_separate_for_owner_and_guest(store: Store) -> None:
    assert store.panel_id(7) is None
    store.set_panel(7, 101)
    store.set_panel(8, 202)
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    assert reopened.panel_id(7) == 101
    assert reopened.panel_id(8) == 202
    reopened.set_panel(7, 103)
    assert store.panel_id(7) == 103
    assert store.panel_id(8) == 202


def test_stale_panel_failure_cannot_clear_a_new_anchor(store: Store) -> None:
    store.set_panel(7, 101)
    concurrent = Store(store.db, store.data_dir, owner_id=7)
    concurrent.set_panel(7, 102)
    assert not store.clear_panel(7, 101)
    assert not store.clear_panel(8, 102)
    assert store.panel_id(7) == 102
    assert store.clear_panel(7, 102)
    assert concurrent.panel_id(7) is None
    assert not store.clear_panel(7, 102)
    concurrent.set_panel(7, 103)
    assert store.panel_id(7) == 103


def test_base_view_survives_restart_and_same_anchor_but_not_replacement(store: Store) -> None:
    keyboard = '{"inline_keyboard":[]}'
    assert store.panel_content(7) is None
    assert not store.set_panel_content(7, 101, "Мои голоса", keyboard)
    store.set_panel(7, 101)
    assert store.panel_content(7) is None
    assert store.set_panel_content(7, 101, "Мои голоса", keyboard)
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    reopened.set_panel(7, 101)
    assert reopened.panel_content(7) == ("Мои голоса", keyboard)
    reopened.set_panel(7, 102)
    assert store.panel_content(7) is None
    assert not store.set_panel_content(7, 101, "Старый экран", keyboard)
    assert reopened.set_panel_content(7, 102, "Настройки", keyboard)
    assert store.panel_content(7) == ("Настройки", keyboard)
    assert not store.clear_panel(7, 101)
    assert store.panel_content(7) == ("Настройки", keyboard)
    assert store.clear_panel(7, 102)
    assert reopened.panel_content(7) is None


def test_replace_panel_persists_anchor_and_base_together_without_changing_other_chats(
    store: Store,
) -> None:
    store.replace_panel(7, 101, "Первая главная", "{}")
    store.replace_panel(8, 201, "Гость", "{}")
    store.replace_panel(7, 102, "Новая главная", '{"inline_keyboard":[]}')
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    assert reopened.panel_id(7) == 102
    assert reopened.panel_content(7) == ("Новая главная", '{"inline_keyboard":[]}')
    assert reopened.panel_id(8) == 201 and reopened.panel_content(8) == ("Гость", "{}")


def test_failed_atomic_replacement_rolls_back_both_anchor_and_base(store: Store) -> None:
    store.replace_panel(7, 101, "Сохранённая главная", "{}")
    with sqlite3.connect(store.db) as connection:
        connection.execute(
            "CREATE TRIGGER reject_panel AFTER UPDATE ON ui_panels "
            "BEGIN SELECT RAISE(ABORT, 'test failure after update'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_panel(7, 102, "Новая главная", '{"inline_keyboard":[]}')
    assert store.panel_id(7) == 101
    assert store.panel_content(7) == ("Сохранённая главная", "{}")


@pytest.mark.parametrize(
    ("text", "keyboard"),
    [("", "{}"), ("x" * 3501, "{}"), ("Экран", " " * 20001), ("Экран", "["), ("Экран", "[]")],
)
def test_invalid_base_view_cannot_replace_last_successful_view(
    store: Store, text: str, keyboard: str
) -> None:
    store.set_panel(7, 101)
    assert store.set_panel_content(7, 101, "Сохранённый экран", "{}")
    with pytest.raises(ValueError):
        store.set_panel_content(7, 101, text, keyboard)
    with pytest.raises(ValueError):
        store.replace_panel(7, 102, text, keyboard)
    assert store.panel_id(7) == 101
    assert store.panel_content(7) == ("Сохранённый экран", "{}")


@pytest.mark.parametrize("invalid", [0, -1, True, 2**63])
def test_invalid_panel_ids_never_replace_or_erase_the_anchor(store: Store, invalid: int) -> None:
    store.set_panel(7, 101)
    with pytest.raises(ValueError):
        store.panel_id(invalid)
    with pytest.raises(ValueError):
        store.set_panel(invalid, 102)
    with pytest.raises(ValueError):
        store.set_panel(7, invalid)
    with pytest.raises(ValueError):
        store.replace_panel(invalid, 102, "Главная", "{}")
    with pytest.raises(ValueError):
        store.replace_panel(7, invalid, "Главная", "{}")
    with pytest.raises(ValueError):
        store.clear_panel(invalid, 101)
    with pytest.raises(ValueError):
        store.clear_panel(7, invalid)
    assert store.panel_id(7) == 101


def test_v5_migration_preserves_all_persisted_state(store: Store) -> None:
    add_profile(store.data_dir, "anna")
    store.update_voice_settings(7, "anna", changes={"native_tags": ("happy",), "seed": 77})
    store.select_profile(7, "anna")
    assert store.enqueue(100, 7, "tts", text="Сохранить задание") is not None
    draft = store.begin_profile_draft(7)
    source = store.data_dir / "upload.audio"
    source.write_bytes(b"saved reference upload")
    assert store.save_profile_upload(101, 7, draft.id, source)
    assert store.accept_callback("saved-click")
    tables = (
        "settings",
        "updates",
        "jobs",
        "delivery_callbacks",
        "profile_drafts",
        "ui_callbacks",
    )
    with sqlite3.connect(store.db) as connection:
        drop_history_schema(connection)
        connection.execute("DROP TABLE ui_panels")
        connection.execute("DROP TABLE reference_changes")
        connection.execute("DROP TABLE job_billing")
        connection.execute("DROP TABLE profile_owners")
        connection.execute("DROP TABLE accounts")
        connection.execute("PRAGMA user_version=5")
        before_rows = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in tables
        }
    before_files = {
        path.relative_to(store.data_dir): path.read_bytes()
        for directory in ("profiles", "jobs", "drafts")
        for path in (store.data_dir / directory).rglob("*")
        if path.is_file()
    }
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    reopened.initialize()
    with sqlite3.connect(store.db) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
        after_rows = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in tables
        }
        assert connection.execute("SELECT * FROM ui_panels").fetchall() == []
    assert after_rows == before_rows
    assert all((store.data_dir / path).read_bytes() == data for path, data in before_files.items())
    assert reopened.profile_draft(7) == store.profile_draft(7)
    reopened.set_panel(8, 102)
    assert reopened.panel_id(8) == 102
