"""Reference transactions use opaque payloads here; the API tests validate real audio."""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from shift.config import atomic_private
from shift.protocol import Profile, ReferenceSample, WorkerJob
from shift.store import ReferenceChange, StaleLease, Store
from tests.test_store import add_profile, drop_history_schema


@pytest.fixture
def reference_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "queue.sqlite3", tmp_path, owner_id=7)
    store.initialize()
    add_profile(tmp_path, "anna")
    add_profile(tmp_path, "maria")
    store.select_profile(7, "anna")
    return store


def queue_add(store: Store) -> tuple[ReferenceChange, WorkerJob]:
    source = store.data_dir / "second-sample.wav"
    source.write_bytes(b"independent second recording")
    change = store.begin_reference_add(7, "anna")
    job_id = store.enqueue_reference_add(501, 7, change.id, source)
    job = store.claim()
    assert job is not None and job.id == job_id and job.kind == "reference"
    return change, job


def prepare_candidate(store: Store, job: WorkerJob) -> tuple[Path, list[str]]:
    assert job.profile is not None
    folder = store.job_dir(job.id) / "prepared"
    folder.mkdir(mode=0o700)
    texts = []
    for index, sample in enumerate(job.profile.reference_samples):
        filename = "reference.wav" if index == 0 else f"reference-{index + 1}.wav"
        shutil.copyfile(store.data_dir / sample.reference_audio, folder / filename)
        texts.append(sample.reference_text)
    if job.text == "add":
        shutil.copyfile(
            store.job_dir(job.id) / "input.audio", folder / f"reference-{len(texts) + 1}.wav"
        )
        texts.append("Ещё одна запись того же человека.")
    (folder / "example.ogg").write_bytes(b"separate preview payload")
    return folder, texts


def ready_candidate(store: Store) -> ReferenceChange:
    change, job = queue_add(store)
    folder, texts = prepare_candidate(store, job)
    store.finish_reference(job.id, job.lease, folder, texts)
    ready = store.reference_change(7, "anna")
    assert ready is not None and ready.id == change.id and ready.stage == "ready"
    return ready


def test_v6_migration_keeps_all_rows_private_uploads_and_saved_panel(
    reference_store: Store,
) -> None:
    store = reference_store
    store.update_voice_settings(7, "anna", speed=0.9, changes={"seed": 123})
    store.replace_panel(7, 100, "Сохранённая карточка", '{"inline_keyboard":[]}')
    store.accept_callback("already-accepted")
    store.enqueue(1, 7, "tts", text="Уже готово")
    ready = store.claim()
    assert ready is not None
    result = store.data_dir / "result.ogg"
    result.write_bytes(b"saved voice result")
    store.finish(ready.id, ready.lease, result, duration_ms=10_000)
    store.enqueue(2, 7, "tts", text="Сейчас готовится")
    running = store.claim()
    assert running is not None
    store.enqueue(3, 7, "tts", text="В очереди")
    draft = store.begin_profile_draft(7)
    sample = store.data_dir / "unfinished.audio"
    sample.write_bytes(b"unfinished private reference")
    assert store.save_profile_upload(4, 7, draft.id, sample)
    tables = ("jobs", "settings", "profile_drafts", "updates", "ui_callbacks", "ui_panels")
    with sqlite3.connect(store.db) as connection:
        drop_history_schema(connection)
        before = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in tables
        }
        for table in ("jobs", "profile_drafts"):
            sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            connection.execute(f"ALTER TABLE {table} RENAME TO latest_{table}")
            connection.execute(sql.replace(",'reference'", ""))
            connection.execute(f"INSERT INTO {table} SELECT * FROM latest_{table}")
            connection.execute(f"DROP TABLE latest_{table}")
        connection.execute("CREATE INDEX jobs_status_created ON jobs(status,created)")
        connection.execute(
            "CREATE UNIQUE INDEX one_running_job ON jobs((1)) WHERE status='running'"
        )
        connection.execute("DROP TABLE reference_changes")
        connection.execute("INSERT INTO settings(key,value) VALUES('profile','anna')")
        connection.execute("DROP TABLE job_billing")
        connection.execute("DROP TABLE profile_owners")
        connection.execute("DROP TABLE accounts")
        connection.execute("PRAGMA user_version=6")
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    with sqlite3.connect(store.db) as connection:
        after = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in tables
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
    assert after == before
    assert reopened.panel_id(7) == 100
    assert (reopened.draft_dir(draft.id) / "input.audio").read_bytes() == sample.read_bytes()
    assert (reopened.job_dir(ready.id) / "result.ogg").read_bytes() == b"saved voice result"
    profile = reopened.selected_profile(7)
    assert profile is not None and profile.seed == 123 and profile.speed == 0.9
    assert reopened.heartbeat(running.id, running.lease) is False


def test_apply_preserves_current_name_settings_and_personal_selection(
    reference_store: Store,
) -> None:
    store = reference_store
    change = ready_candidate(store)
    draft = store.begin_profile_draft(7, "rename", "anna")
    assert store.rename_profile(502, 7, draft.id, "Анна новая")
    store.update_voice_settings(
        7, "anna", speed=1.1, changes={"native_tags": ["whisper"], "seed": 999}
    )
    selected = store.select_profile(7, "maria")
    applied = store.apply_reference_change(7, change.id)
    assert applied is not None and applied.id == "anna" and applied.name == "Анна новая"
    assert applied.speed == 1.1 and applied.seed == 999 and applied.native_tags == ("whisper",)
    assert len(applied.reference_samples) == 2 and store.selected_profile(7) == selected
    assert store.reference_change(7, "anna") is None
    assert store.apply_reference_change(7, change.id) is None


def test_discard_ready_candidate_prevents_late_delivery_and_keeps_original(
    reference_store: Store,
) -> None:
    store = reference_store
    original = store.selected_profile(7)
    change = ready_candidate(store)
    assert change.job_id is not None
    assert store.discard_reference_change(7, change.id)
    assert store.delivery_claim() is None
    assert store.list_jobs(7)[0].status == "cancelled"
    assert not (store.job_dir(change.job_id) / "result.ogg").exists()
    assert not (store.data_dir / "reference-changes" / change.id).exists()
    assert store.selected_profile(7) == original


def test_removing_primary_promotes_next_reference_and_keeps_at_least_one(
    reference_store: Store,
) -> None:
    store = reference_store
    added = store.apply_reference_change(7, ready_candidate(store).id)
    assert added is not None
    second = added.reference_samples[1]
    expected_audio = store.reference_sample_path(7, "anna", 1).read_bytes()
    store.enqueue_reference_remove(7, "anna", 0)
    job = store.claim()
    assert job is not None and job.profile is not None
    assert len(job.profile.reference_samples) == 1
    folder, texts = prepare_candidate(store, job)
    store.finish_reference(job.id, job.lease, folder, texts)
    change = store.reference_change(7, "anna")
    assert change is not None
    current = store.apply_reference_change(7, change.id)
    assert current is not None and len(current.reference_samples) == 1
    assert current.reference_text == second.reference_text
    assert store.reference_sample_path(7, "anna", 0).read_bytes() == expected_audio
    with pytest.raises(ValueError, match="At least one"):
        store.enqueue_reference_remove(7, "anna", 0)


def test_stale_reference_bytes_block_apply_without_deleting_candidate(
    reference_store: Store,
) -> None:
    store = reference_store
    change = ready_candidate(store)
    source = store.reference_sample_path(7, "anna", 0)
    source.write_bytes(b"newer recording")
    with pytest.raises(ValueError, match="changed"):
        store.apply_reference_change(7, change.id)
    assert source.read_bytes() == b"newer recording"
    assert store.reference_change(7, "anna") == change
    assert store.reference_change_preview_path(7, change.id).exists()


def test_profile_deleted_during_preparation_is_not_recreated(reference_store: Store) -> None:
    store = reference_store
    change, job = queue_add(store)
    folder, texts = prepare_candidate(store, job)
    draft = store.begin_profile_draft(7, "delete", "anna")
    assert store.delete_profile(7, draft.id)
    with pytest.raises(StaleLease):
        store.finish_reference(job.id, job.lease, folder, texts)
    assert not (store.data_dir / "profiles/anna").exists()
    assert not (store.data_dir / "reference-changes" / change.id).exists()
    assert store.reference_change(7, "anna") is None
    assert job.profile is not None
    assert (store.data_dir / job.profile.reference_audio).read_bytes() == b"anna"


def test_extra_reference_snapshot_is_independent_of_later_live_edits(
    reference_store: Store,
) -> None:
    store = reference_store
    profile = store.selected_profile(7)
    assert profile is not None
    extra = ReferenceSample(
        reference_audio="reference-2.wav", reference_text="Второй оригинальный текст."
    )
    folder = store.data_dir / "profiles/anna"
    (folder / extra.reference_audio).write_bytes(b"original extra")
    changed = profile.model_copy(update={"extra_references": (extra,)})
    atomic_private(folder / "profile.json", changed.model_dump_json().encode())
    store.enqueue(500, 7, "tts", text="Снимок обоих образцов")
    (folder / extra.reference_audio).write_bytes(b"modified extra")
    changed = changed.model_copy(
        update={
            "extra_references": (extra.model_copy(update={"reference_text": "Изменённый текст."}),)
        }
    )
    atomic_private(folder / "profile.json", changed.model_dump_json().encode())
    job = store.claim()
    assert job is not None and job.profile is not None
    copied = job.profile.reference_samples[1]
    assert copied.reference_text == extra.reference_text
    assert (store.data_dir / copied.reference_audio).read_bytes() == b"original extra"
    assert (store.data_dir / copied.reference_audio).is_relative_to(store.job_dir(job.id))


@pytest.mark.parametrize("point", ["first_rename", "second_rename", "committed"])
def test_crash_recovery_restores_old_or_keeps_committed_reference_swap(
    reference_store: Store, point: str
) -> None:
    store = reference_store
    original = store.selected_profile(7)
    change = ready_candidate(store)
    target = store.data_dir / "profiles/anna"
    prepared = store.data_dir / "reference-changes" / change.id
    tomb = store.data_dir / "profile-trash" / change.id
    marker = tomb.with_name(f"reference-{change.id}.json")
    atomic_private(marker, json.dumps({"change_id": change.id, "profile_id": "anna"}).encode())
    target.rename(tomb)
    if point != "first_rename":
        prepared.rename(target)
    if point == "committed":
        committed = Profile.model_validate_json((target / "profile.json").read_text())
        metadata = committed.model_dump(mode="json")
        metadata["effective_native_tags"] = list(committed.effective_native_tags)
        with sqlite3.connect(store.db) as connection:
            connection.execute(
                "UPDATE voice_metadata SET profile_json=? WHERE profile_id=?",
                (json.dumps(metadata), committed.id),
            )
            connection.execute("DELETE FROM reference_changes WHERE id=?", (change.id,))
    reopened = Store(store.db, store.data_dir, owner_id=7)
    reopened.initialize()
    reopened.recover()
    recovered = reopened.selected_profile(7)
    assert recovered is not None
    if point == "committed":
        assert len(recovered.reference_samples) == 2
        assert reopened.reference_change(7, "anna") is None and not prepared.exists()
    else:
        assert recovered == original
        assert reopened.reference_change(7, "anna") == change
        assert Profile.model_validate_json((prepared / "profile.json").read_text()).extra_references
    assert not tomb.exists() and not marker.exists()
    reopened.recover()
    assert reopened.selected_profile(7) == recovered
