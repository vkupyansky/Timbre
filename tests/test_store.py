import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from shift.protocol import RETENTION_SECONDS, Profile
from shift.store import NoProfile, QueueFull, StaleLease, Store


def add_profile(
    directory: Path, identifier: str, *, user_id: int = 7, db: Path | None = None
) -> None:
    path = directory / "profiles" / identifier
    path.mkdir(parents=True)
    (path / "reference.wav").write_bytes(identifier.encode())
    (path / "profile.json").write_text(
        json.dumps(
            {
                "id": identifier,
                "name": identifier,
                "reference_audio": "reference.wav",
                "reference_text": "Проверочный референс.",
            }
        )
    )
    database = db or directory / "queue.sqlite3"
    if database.is_file():
        with sqlite3.connect(database) as connection:
            account_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='accounts'"
            ).fetchone()
            if account_table is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO accounts(user_id,balance_ms,created) VALUES(?,?,?)",
                    (user_id, 0 if user_id == 7 else 600_000, time.time()),
                )
                connection.execute(
                    "INSERT INTO profile_owners(profile_id,user_id) VALUES(?,?)",
                    (identifier, user_id),
                )
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='voice_metadata'"
                ).fetchone():
                    profile = Profile.model_validate_json((path / "profile.json").read_text())
                    metadata = profile.model_dump(mode="json")
                    metadata["effective_native_tags"] = list(profile.effective_native_tags)
                    connection.execute(
                        "INSERT INTO voice_metadata(profile_id,profile_json,updated) VALUES(?,?,?)",
                        (identifier, json.dumps(metadata), time.time()),
                    )


def drop_history_schema(connection: sqlite3.Connection) -> None:
    """Reconstruct a genuine older schema for migration regression fixtures."""
    for name in (
        "generation_job_state",
        "generation_billing_state",
        "generation_snapshot_immutable",
        "generation_transcript_immutable",
        "generation_result_snapshot_immutable",
    ):
        connection.execute(f"DROP TRIGGER IF EXISTS {name}")
    for name in ("report_users", "report_voices", "report_generations"):
        connection.execute(f"DROP VIEW IF EXISTS {name}")
    for name in ("voice_metadata", "generation_history"):
        connection.execute(f"DROP TABLE IF EXISTS {name}")
    if "is_admin" in {row[1] for row in connection.execute("PRAGMA table_info(accounts)")}:
        connection.execute("ALTER TABLE accounts DROP COLUMN is_admin")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = Store(tmp_path / "queue.sqlite3", tmp_path, owner_id=7)
    result.initialize()
    add_profile(tmp_path, "anna")
    return result


def test_duplicate_update_and_queue_bound_under_concurrency(store: Store) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.enqueue(100, 7, "tts", text="Привет"), range(8)))
    assert sum(result is not None for result in results) == 1
    store.enqueue(101, 7, "tts", text="Второе")
    store.enqueue(102, 7, "tts", text="Третье")
    with pytest.raises(QueueFull):
        store.enqueue(103, 7, "tts", text="Лишнее")
    assert not store.update_seen(103)
    assert len(list((store.data_dir / "jobs").iterdir())) == 3
    assert store.claim() is not None
    assert store.claim() is None
    assert store.enqueue(103, 7, "tts", text="Теперь помещается") is not None


def test_profile_selection_and_audio_snapshot_survive_restart(store: Store) -> None:
    job_id = store.enqueue(1, 7, "tts", text="Анна")
    assert job_id is not None
    add_profile(store.data_dir, "maria")
    store.select_profile(7, "maria")
    (store.data_dir / "profiles" / "anna" / "reference.wav").write_bytes(b"changed")
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    reopened.recover()
    profile = reopened.selected_profile(7)
    assert profile is not None and profile.id == "maria"
    job = reopened.claim()
    assert job is not None and job.profile is not None and job.profile.id == "anna"
    assert (store.data_dir / job.profile.reference_audio).read_bytes() == b"anna"


def test_restart_does_not_regenerate_or_automatically_resend(store: Store) -> None:
    job_id = store.enqueue(1, 7, "tts", text="Тест")
    running = store.claim()
    assert running is not None
    store.recover()
    assert store.list_jobs(7)[0].status == "failed"
    assert not (store.job_dir(running.id) / "reference.wav").exists()
    with pytest.raises(StaleLease):
        store.heartbeat(running.id, running.lease)
    assert store.enqueue(1, 7, "tts", text="Тест") is None
    assert store.claim() is None
    assert job_id is not None
    store.enqueue(2, 7, "tts", text="Другой тест")
    sending = store.claim()
    assert sending is not None
    result = store.data_dir / "result.ogg"
    result.write_bytes(b"durable result")
    store.finish(sending.id, sending.lease, result, duration_ms=10_000)
    assert store.delivery_claim() is not None
    store.recover()
    assert store.list_jobs(7)[0].status == "uncertain"
    assert store.delivery_claim() is None
    assert store.retry_delivery(sending.id, 8) is False
    assert store.retry_delivery(sending.id, 7) is True
    assert store.retry_delivery(sending.id, 7) is False
    assert store.claim() is None
    assert store.delivery_claim() is not None


def test_cancel_is_confirmed_by_worker_and_stale_results_fenced(store: Store) -> None:
    queued = store.enqueue(1, 7, "tts", text="Отмена")
    assert queued is not None
    assert store.request_cancel(queued, 8) == "missing"
    assert store.request_cancel(queued, 7) == "cancelled"
    assert store.claim() is None
    store.enqueue(2, 7, "tts", text="Остановка")
    job = store.claim()
    assert job is not None
    assert store.request_cancel(job.id, 7) == "requested"
    assert store.list_jobs(7)[0].status == "running"
    assert store.heartbeat(job.id, job.lease)
    path = store.data_dir / "result.ogg"
    path.write_bytes(b"result")
    with pytest.raises(StaleLease):
        store.finish(job.id, job.lease, path, duration_ms=10_000)
    store.confirm_cancel(job.id, job.lease)
    assert store.list_jobs(7)[0].status == "cancelled"
    with pytest.raises(StaleLease):
        store.fail(job.id, job.lease, "model_error")


def test_expiry_frees_slot_cleans_private_audio_and_keeps_dedup(store: Store) -> None:
    store.enqueue(1, 7, "tts", text="Исходный приватный текст")
    job = store.claim()
    assert job is not None
    with sqlite3.connect(store.db) as connection:
        connection.execute("UPDATE jobs SET lease_until=?", (time.time() - 1,))
    with pytest.raises(StaleLease):
        store.heartbeat(job.id, job.lease)
    store.maintain()
    assert store.list_jobs(7)[0].failure == "worker_lost"
    store.maintain(time.time() + RETENTION_SECONDS + 1)
    assert store.list_jobs(7) == []
    assert not store.job_dir(job.id).exists()
    assert store.update_seen(1)
    assert (store.data_dir / "profiles" / "anna" / "reference.wav").exists()


def test_profile_path_traversal_rejected(store: Store) -> None:
    profile = store.data_dir / "profiles" / "anna" / "profile.json"
    data = json.loads(profile.read_text())
    data["reference_audio"] = "../../../private.wav"
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid profile"):
        store.profiles(7)


def test_text_limits_and_exact_words(store: Store) -> None:
    with pytest.raises(ValueError):
        store.enqueue(1, 7, "tts", text="я" * 2001)
    text = "  Я был у Анны в 12:30. Всё верно?  "
    store.enqueue(2, 7, "tts", text=text)
    job = store.claim()
    assert job is not None and job.text == text


def test_replayed_callback_cannot_send_result_again_after_success(store: Store) -> None:
    store.enqueue(1, 7, "tts", text="Тест")
    job = store.claim()
    assert job is not None
    result = store.data_dir / "generated.ogg"
    result.write_bytes(b"saved result")
    store.finish(job.id, job.lease, result, duration_ms=10_000)
    assert store.delivery_claim() is not None
    store.delivery_done(job.id, None, uncertain=True)
    assert store.retry_delivery(job.id, 7, "callback-id")
    assert store.delivery_claim() is not None
    store.delivery_done(job.id, 17)
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    assert not reopened.retry_delivery(job.id, 7, "callback-id")
    assert reopened.delivery_claim() is None
    assert reopened.retry_delivery(job.id, 7, "a-new-explicit-click")


def test_profile_draft_survives_restart_then_prepares_without_existing_voice(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "queue.sqlite3", tmp_path, owner_id=7)
    store.initialize()
    draft = store.begin_profile_draft(7)
    source = tmp_path / "upload.audio"
    source.write_bytes(b"profile source fixture")
    assert store.save_profile_upload(10, 7, draft.id, source)
    assert not store.save_profile_upload(10, 7, draft.id, source)
    reopened = Store(store.db, tmp_path, owner_id=7)
    reopened.initialize()
    reopened.recover()
    current = reopened.profile_draft(7)
    assert current is not None and current.id == draft.id and current.stage == "name"
    assert reopened.enqueue(11, 7, "profile", text="Анна", draft_id=current.id)
    assert reopened.profile_draft(7) is None
    assert not reopened.draft_dir(current.id).exists()
    job = reopened.claim()
    assert job is not None and job.kind == "profile" and job.profile is None and job.text == "Анна"
    assert reopened.profiles(7) == []
    prepared = reopened.job_dir(job.id) / "prepared"
    prepared.mkdir()
    (prepared / "reference.wav").write_bytes(b"validated reference fixture")
    (prepared / "example.ogg").write_bytes(b"validated example fixture")
    profile = reopened.finish_profile(job.id, job.lease, prepared, "Точный русский текст.")
    assert reopened.selected_profile(7) == profile
    assert profile.reference_text == "Точный русский текст."
    assert reopened.list_jobs(7)[0].status == "ready"
    assert reopened.list_jobs(7)[0].profile_id == profile.id
    assert (tmp_path / "profiles" / profile.id / "example.ogg").exists()
    with pytest.raises(StaleLease):
        reopened.finish_profile(job.id, job.lease, prepared, "Повтор")


def test_rename_delete_and_selected_fallback_preserve_enqueued_snapshot(store: Store) -> None:
    store.enqueue(1, 7, "tts", text="Старый снимок")
    add_profile(store.data_dir, "maria")
    rename = store.begin_profile_draft(7, "rename", "anna")
    assert store.rename_profile(2, 7, rename.id, "Новое имя")
    assert not store.rename_profile(2, 7, rename.id, "Дубль")
    selected = Store(store.db, store.data_dir, owner_id=7).selected_profile(7)
    assert selected is not None and selected.name == "Новое имя"
    deleting = store.begin_profile_draft(7, "delete", "anna")
    assert not store.delete_profile(8, deleting.id)
    assert store.delete_profile(7, deleting.id)
    assert not store.delete_profile(7, deleting.id)
    selected = store.selected_profile(7)
    assert selected is not None and selected.id == "maria"
    job = store.claim()
    assert job is not None and job.profile is not None
    assert job.profile.id == "anna" and job.profile.name == "anna"
    assert (store.data_dir / job.profile.reference_audio).read_bytes() == b"anna"
    deleting_last = store.begin_profile_draft(7, "delete", "maria")
    assert store.delete_profile(7, deleting_last.id)
    assert store.selected_profile(7) is None
    assert store.profiles(7) == []


def test_stale_confirmation_cannot_delete_replaced_draft(store: Store) -> None:
    first = store.begin_profile_draft(7, "delete", "anna")
    second = store.begin_profile_draft(7, "rename", "anna")
    assert not store.delete_profile(7, first.id)
    assert store.profile_draft(7) == second
    assert store.accept_callback("unique-click")
    assert not Store(store.db, store.data_dir, owner_id=7).accept_callback("unique-click")


def test_v3_upgrade_preserves_existing_jobs_selection_and_dedup(store: Store) -> None:
    store.select_profile(7, "anna")
    job_id = store.enqueue(100, 7, "tts", text="Не потерять")
    # Reconstruct the persisted v3 column layout and keep real queue/settings data.
    with sqlite3.connect(store.db) as connection:
        drop_history_schema(connection)
        connection.execute("ALTER TABLE jobs DROP COLUMN result_profile_id")
        connection.execute("DROP TABLE profile_drafts")
        connection.execute("DROP TABLE ui_callbacks")
        connection.execute("DROP TABLE ui_panels")
        connection.execute("DROP TABLE reference_changes")
        connection.execute("INSERT INTO settings(key,value) VALUES('profile','anna')")
        connection.execute("DROP TABLE job_billing")
        connection.execute("DROP TABLE profile_owners")
        connection.execute("DROP TABLE accounts")
        connection.execute("PRAGMA user_version=3")
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    assert reopened.update_seen(100)
    assert reopened.list_jobs(7)[0].id == job_id
    job = reopened.claim()
    assert job is not None and job.profile is not None and job.text == "Не потерять"
    assert job.profile.id == "anna"


def test_voice_settings_persist_per_profile_without_changing_reference_or_selection(
    store: Store,
) -> None:
    add_profile(store.data_dir, "maria")
    store.select_profile(7, "maria")
    reference = store.data_dir / "profiles" / "anna" / "reference.wav"
    example = reference.with_name("example.ogg")
    example.write_bytes(b"original neutral preview")
    updated = store.update_voice_settings(7, "anna", delivery="happy", speed=0.9)
    assert updated.name == "anna" and updated.reference_text == "Проверочный референс."
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    profiles = {profile.id: profile for profile in reopened.profiles(7)}
    assert profiles["anna"].native_tags == ("delight",) and profiles["anna"].speed == 0.9
    assert profiles["maria"].delivery == "neutral" and profiles["maria"].speed == 1.0
    assert reopened.selected_profile(7) == profiles["maria"]
    assert reference.read_bytes() == b"anna"
    assert example.read_bytes() == b"original neutral preview"
    reset = reopened.update_voice_settings(7, "anna", delivery="neutral", speed=1.0)
    assert reset.delivery == "neutral" and reset.speed == 1.0


def test_voice_settings_cannot_be_mutated_by_guest_or_invalid_value(store: Store) -> None:
    path = store.data_dir / "profiles" / "anna" / "profile.json"
    original = path.read_bytes()
    with pytest.raises(NoProfile):
        store.update_voice_settings(8, "anna", delivery="whisper", speed=1.1)
    for speed in (0.0, 1.05, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            store.update_voice_settings(7, "anna", speed=speed)
    assert path.read_bytes() == original


def test_profile_settings_updates_serialize_between_store_instances(store: Store) -> None:
    other = Store(store.db, store.data_dir, owner_id=7)
    with ThreadPoolExecutor(max_workers=2) as pool:
        delivery = pool.submit(store.update_voice_settings, 7, "anna", delivery="angry")
        speed = pool.submit(other.update_voice_settings, 7, "anna", speed=1.1)
        delivery.result()
        speed.result()
    selected = store.selected_profile(7)
    assert selected is not None and selected.native_tags == ("angry",) and selected.speed == 1.1


def test_preview_and_voice_input_freeze_settings_without_changing_personal_selection(
    store: Store,
) -> None:
    add_profile(store.data_dir, "maria")
    store.select_profile(7, "maria")
    store.update_voice_settings(7, "anna", delivery="sad", speed=0.9)
    preview = store.enqueue(None, 7, "tts", text="Пример", profile_id="anna")
    source = store.data_dir / "voice.audio"
    source.write_bytes(b"speech input")
    store.update_voice_settings(7, "maria", delivery="angry", speed=1.1)
    speech = store.enqueue(11, 7, "vc", source=source)
    assert preview is not None and speech is not None
    store.update_voice_settings(7, "anna", delivery="neutral", speed=1.0)
    store.update_voice_settings(7, "maria", delivery="neutral", speed=1.0)
    with sqlite3.connect(store.db) as connection:
        snapshots = {
            row[0]: Profile.model_validate_json(row[1])
            for row in connection.execute("SELECT id,profile FROM jobs")
        }
        assert connection.execute("SELECT COUNT(*) FROM updates").fetchone()[0] == 1
    assert snapshots[preview].id == "anna"
    assert snapshots[preview].native_tags == ("sad",) and snapshots[preview].speed == 0.9
    assert snapshots[speech].id == "maria"
    assert snapshots[speech].native_tags == ("angry",) and snapshots[speech].speed == 1.1
    selected = store.selected_profile(7)
    assert selected is not None and selected.id == "maria"


def test_legacy_job_snapshot_receives_neutral_speed_defaults(store: Store) -> None:
    store.enqueue(12, 7, "tts", text="Старое задание")
    with sqlite3.connect(store.db) as connection:
        row = connection.execute("SELECT id,profile FROM jobs").fetchone()
        profile = json.loads(row[1])
        profile.pop("delivery")
        profile.pop("speed")
        connection.execute("UPDATE jobs SET profile=? WHERE id=?", (json.dumps(profile), row[0]))
    job = store.claim()
    assert job is not None and job.profile is not None
    assert job.profile.delivery == "neutral" and job.profile.speed == 1.0
