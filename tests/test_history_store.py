"""Durable database reports retain real inputs and immutable synthesis snapshots."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from shift.protocol import RETENTION_SECONDS
from shift.store import InsufficientBalance, StaleLease, Store
from tests.test_accounts_store import create_default_template, output
from tests.test_reference_store import ready_candidate
from tests.test_store import add_profile, drop_history_schema


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = Store(tmp_path / "queue.sqlite3", tmp_path, 7, admin_user_ids=frozenset({7, 8, 9}))
    result.initialize()
    add_profile(tmp_path, "anna")
    add_profile(tmp_path, "member", user_id=10)
    return result


def report(store: Store, job_id: str) -> sqlite3.Row:
    with sqlite3.connect(store.db) as connection:
        connection.row_factory = sqlite3.Row
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM report_generations WHERE job_id=?", (job_id,)
        ).fetchone()
        assert row is not None
        return row


def test_text_recognition_settings_and_charges_survive_voice_and_audio_deletion(
    store: Store,
) -> None:
    store.update_account_user(10, "member", "Личный аккаунт")
    store.update_voice_settings(10, "member", speed=0.9, delivery="happy", changes={"seed": 11})
    first_id = store.enqueue(1, 10, "tts", text="Привет, мИр!")
    assert first_id is not None
    first = report(store, first_id)
    assert first["input_text"] == "Привет, мИр!" and first["synthesis_text"] == "Привет, мИр!"
    snapshot = json.loads(first["voice_snapshot_json"])
    assert snapshot["speed"] == 0.9 and snapshot["effective_native_tags"] == ["delight"]
    assert snapshot["seed"] == 11 and len(first["reference_fingerprint"]) == 64
    store.update_voice_settings(10, "member", speed=1.1, delivery="whisper", changes={"seed": 42})
    second_id = store.enqueue(2, 10, "vc", source=output(store, "source.audio"))
    assert second_id is not None
    text_job = store.claim()
    assert text_job is not None and text_job.id == first_id
    store.finish(text_job.id, text_job.lease, output(store), duration_ms=17_111)
    assert store.delivery_claim() is not None
    store.delivery_done(first_id, None)
    failed_delivery = report(store, first_id)
    assert failed_delivery["generation_status"] == "ready"
    assert failed_delivery["delivery_status"] == "failed"
    assert store.retry_delivery(first_id, 10, "retry-once")
    assert store.delivery_claim() is not None
    store.delivery_done(first_id, 100)
    speech_job = store.claim()
    assert speech_job is not None and speech_job.id == second_id
    recognized = "Это действительно распознанная речь."
    store.record_transcript(second_id, speech_job.lease, recognized)
    store.record_transcript(second_id, speech_job.lease, recognized)
    assert report(store, second_id)["transcript"] == recognized
    store.finish(second_id, speech_job.lease, output(store), duration_ms=12_345)
    assert store.delivery_claim() is not None
    store.delivery_done(second_id, 101)
    deletion = store.begin_profile_draft(10, "delete", "member")
    assert store.delete_profile(10, deletion.id)
    store.maintain(time.time() + RETENTION_SECONDS + 1)
    assert store.list_jobs(10) == [] and list((store.data_dir / "jobs").iterdir()) == []
    assert store.account(10).balance_ms == 570_544
    retained_first, retained_second = report(store, first_id), report(store, second_id)
    assert retained_first["voice_snapshot_json"] == first["voice_snapshot_json"]
    assert retained_first["input_text"] == "Привет, мИр!"
    assert retained_first["charged_ms"] == 17_111 and retained_first["duration_ms"] == 17_111
    assert retained_second["synthesis_text"] == recognized
    assert retained_second["charged_ms"] == 12_345 and retained_second["duration_ms"] == 12_345
    assert retained_first["status"] == retained_second["status"] == "sent"
    assert retained_second["generated_at"] is not None
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM report_voices WHERE user_id=10").fetchone()[0]
            == 0
        )
        user = connection.execute(
            "SELECT username,display_name,balance_ms,selected_profile_id "
            "FROM report_users WHERE user_id=10"
        ).fetchone()
        assert user == ("member", "Личный аккаунт", 570_544, None)


@pytest.mark.parametrize(
    "mode", ["queued_cancel", "running_cancel", "model_error", "restart", "timeout", "balance"]
)
def test_every_terminal_generation_state_is_reported_after_job_cleanup(
    store: Store, mode: str
) -> None:
    if mode == "balance":
        with sqlite3.connect(store.db) as connection:
            connection.execute("UPDATE accounts SET balance_ms=1000 WHERE user_id=10")
    job_id = store.enqueue(1, 10, "tts", text="Сохраняем даже неудачное задание")
    assert job_id is not None
    if mode == "queued_cancel":
        store.request_cancel(job_id, 10)
    else:
        job = store.claim()
        assert job is not None
        if mode == "running_cancel":
            store.request_cancel(job_id, 10)
            assert report(store, job_id)["cancel_requested"] == 1
            store.confirm_cancel(job_id, job.lease)
        elif mode == "model_error":
            store.fail(job_id, job.lease, "model_error")
        elif mode == "restart":
            store.recover()
        elif mode == "timeout":
            store.maintain(time.time() + 1000)
        else:
            with pytest.raises(InsufficientBalance):
                store.finish(job_id, job.lease, output(store), duration_ms=1001)
    expected = store.list_jobs(10)[0]
    store.maintain(time.time() + RETENTION_SECONDS + 1)
    history = report(store, job_id)
    assert history["status"] == expected.status and history["failure"] == expected.failure
    assert history["charged_ms"] == 0 and history["billing_state"] == "released"
    assert history["input_text"] == "Сохраняем даже неудачное задание"
    assert history["generation_status"] in ("failed", "cancelled")
    assert history["delivery_status"] is None


def test_transcript_requires_lease_and_cannot_be_changed_after_recording(store: Store) -> None:
    job_id = store.enqueue(1, 10, "vc", source=output(store, "source.audio"))
    job = store.claim()
    assert job is not None and job_id == job.id
    with pytest.raises(ValueError, match="Recognized text must be recorded"):
        store.finish(job.id, job.lease, output(store), duration_ms=1000)
    with pytest.raises(StaleLease):
        store.record_transcript(job.id, "incorrect-lease", "Чужая запись")
    for text in ("", " ", "Нулевой\0символ", "я" * 2001):
        with pytest.raises(ValueError):
            store.record_transcript(job.id, job.lease, text)
    store.record_transcript(job.id, job.lease, "Исходное распознавание")
    with pytest.raises(ValueError, match="already recorded"):
        store.record_transcript(job.id, job.lease, "Подменённый текст")
    with sqlite3.connect(store.db) as connection:
        for column, value in (
            ("input_text", "Подмена"),
            ("voice_snapshot_json", "{}"),
            ("transcript", "Подмена"),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"UPDATE generation_history SET {column}=? WHERE job_id=?", (value, job.id)
                )
    assert report(store, job.id)["transcript"] == "Исходное распознавание"
    store.request_cancel(job.id, 10)
    with pytest.raises(StaleLease):
        store.record_transcript(job.id, job.lease, "Исходное распознавание")


def test_v8_backfill_uses_stored_snapshot_and_does_not_fabricate_missing_text(store: Store) -> None:
    store.update_voice_settings(10, "member", speed=0.9)
    job_id = store.enqueue(1, 10, "tts", text="Удалённый при завершении текст")
    job = store.claim()
    assert job is not None and job_id == job.id
    store.finish(job.id, job.lease, output(store), duration_ms=1000)
    with sqlite3.connect(store.db) as connection:
        stored_profile = connection.execute(
            "SELECT profile FROM jobs WHERE id=?", (job.id,)
        ).fetchone()[0]
    store.update_voice_settings(10, "member", speed=1.1)
    with sqlite3.connect(store.db) as connection:
        drop_history_schema(connection)
        connection.execute("PRAGMA user_version=8")
    reopened = Store(store.db, store.data_dir, 7, admin_user_ids=frozenset({7, 8, 9}))
    reopened.initialize()
    historical = report(reopened, job.id)
    assert historical["historical"] == 1 and historical["input_text"] is None
    assert historical["synthesis_text"] is None and historical["transcript"] is None
    assert historical["voice_snapshot_json"] == stored_profile
    assert json.loads(historical["voice_snapshot_json"])["speed"] == 0.9
    assert historical["duration_ms"] is None and historical["reference_fingerprint"] is None
    assert historical["charged_ms"] == 1000  # Existing settled billing remains known.
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute(
                "SELECT speed FROM report_voices WHERE profile_id='member'"
            ).fetchone()[0]
            == 1.1
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_current_voice_report_tracks_reference_approval_and_default_copies(store: Store) -> None:
    store.update_voice_settings(7, "anna", speed=0.9)
    candidate = ready_candidate(store)
    assert candidate.job_id is not None
    prepared = report(store, candidate.job_id)
    assert json.loads(prepared["result_voice_snapshot_json"])["speed"] == 0.9
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute(
                "SELECT reference_count FROM report_voices WHERE profile_id='anna'"
            ).fetchone()[0]
            == 1
        )
    store.update_voice_settings(7, "anna", speed=1.1)
    store.apply_reference_change(7, candidate.id)
    rename = store.begin_profile_draft(7, "rename", "anna")
    store.rename_profile(20, 7, rename.id, "Новое имя")
    with sqlite3.connect(store.db) as connection:
        assert connection.execute(
            "SELECT name,speed,reference_count FROM report_voices WHERE profile_id='anna'"
        ).fetchone() == ("Новое имя", 1.1, 2)
    assert (
        report(store, candidate.job_id)["result_voice_snapshot_json"]
        == prepared["result_voice_snapshot_json"]
    )
    create_default_template(store, "anna")
    store.initialize()
    store.account(501)
    with sqlite3.connect(store.db) as connection:
        assert connection.execute(
            "SELECT name,is_default,reference_count FROM report_voices WHERE user_id=501"
        ).fetchone() == ("Вика", 1, 2)
        assert (
            connection.execute("SELECT is_admin FROM report_users WHERE user_id=7").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT is_admin FROM report_users WHERE user_id=501").fetchone()[0]
            == 0
        )


def test_profile_json_and_database_rollback_together_when_commit_fails(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.data_dir / "profiles" / "member" / "profile.json"
    original = path.read_bytes()
    connect = sqlite3.connect

    class FailedCommit(sqlite3.Connection):
        def commit(self) -> None:
            raise sqlite3.OperationalError("Simulated commit failure")

    def failed_connection(database: Path, *, timeout: float) -> sqlite3.Connection:
        return connect(database, timeout=timeout, factory=FailedCommit)

    with monkeypatch.context() as patch:
        patch.setattr("shift.store.sqlite3.connect", failed_connection)
        with pytest.raises(sqlite3.OperationalError, match="Simulated commit failure"):
            store.update_voice_settings(10, "member", speed=0.9)
    assert path.read_bytes() == original
    assert store.get_profile(10, "member").speed == 1.0
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute(
                "SELECT speed FROM report_voices WHERE profile_id='member'"
            ).fetchone()[0]
            == 1.0
        )
    # Simulate a process crash after the file rename but before SQLite commit.
    values = json.loads(original)
    values["speed"] = 0.9
    path.write_text(json.dumps(values))
    store.initialize()
    assert store.get_profile(10, "member").speed == 1.0
    recovered_bytes = path.read_bytes()
    store.initialize()
    assert path.read_bytes() == recovered_bytes


@pytest.mark.parametrize("committed", [False, True])
def test_interrupted_profile_deletion_preserves_only_committed_database_choice(
    store: Store, committed: bool
) -> None:
    store.select_profile(10, "member")
    original = store.get_profile(10, "member")
    draft = store.begin_profile_draft(10, "delete", "member")
    source = store.data_dir / "profiles" / "member"
    tomb = store.data_dir / "profile-trash" / draft.id
    original_bytes = (source / "profile.json").read_bytes()
    source.rename(tomb)
    if committed:
        with sqlite3.connect(store.db) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM profile_owners WHERE profile_id='member'")
            connection.execute("UPDATE accounts SET selected_profile_id=NULL WHERE user_id=10")
            connection.execute("DELETE FROM profile_drafts WHERE id=?", (draft.id,))
    reopened = Store(store.db, store.data_dir, 7, admin_user_ids=frozenset({7, 8, 9}))
    reopened.initialize()
    reopened.recover()
    assert not tomb.exists()
    if committed:
        assert reopened.profiles(10) == [] and not source.exists()
    else:
        assert reopened.get_profile(10, "member") == original
        assert (source / "profile.json").read_bytes() == original_bytes
        assert reopened.selected_profile(10) == original
