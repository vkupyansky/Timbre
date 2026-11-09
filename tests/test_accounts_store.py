"""Personal ownership, one-time trial and durable second-based billing invariants."""

import hashlib
import json
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pytest

from shift.protocol import PREVIEW_TEXT, RETENTION_SECONDS
from shift.store import Account, InsufficientBalance, NoProfile, StaleLease, Store
from tests.test_reference_store import ready_candidate
from tests.test_store import add_profile, drop_history_schema


@pytest.fixture
def personal_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "queue.sqlite3", tmp_path, 7, admin_user_ids=frozenset({7, 8, 9}))
    store.initialize()
    add_profile(tmp_path, "owner")
    add_profile(tmp_path, "member", user_id=10)
    return store


def output(store: Store, name: str = "generated.ogg") -> Path:
    path = store.data_dir / name
    path.write_bytes(b"new generated audio")
    return path


def test_accounts_have_three_zero_balance_admins_and_one_time_public_trial(
    personal_store: Store,
) -> None:
    store = personal_store
    for user_id in (7, 8, 9):
        assert store.account(user_id) == Account(user_id, True, 0, 0)
    assert store.account(10) == Account(10, False, 600_000, 0)
    # Public signup requires no allowlist or administrator action.
    assert store.account(501).available_ms == 600_000
    with sqlite3.connect(store.db) as connection:
        connection.execute("UPDATE accounts SET balance_ms=12345 WHERE user_id=501")
    reopened = Store(store.db, store.data_dir, 7, admin_user_ids=frozenset({7, 8, 9}))
    reopened.initialize()
    reopened.recover()
    reopened.clear_panel(501, 1)
    assert reopened.account(501).balance_ms == 12_345
    assert reopened.account(501).balance_ms == 12_345
    with pytest.raises(ValueError):
        reopened.account(-10)


def test_personal_catalog_selection_and_controls_never_cross_accounts(
    personal_store: Store,
) -> None:
    store = personal_store
    add_profile(store.data_dir, "member-second", user_id=10)
    owner_bytes = (store.data_dir / "profiles" / "owner" / "profile.json").read_bytes()
    assert [profile.id for profile in store.profiles(7)] == ["owner"]
    assert [profile.id for profile in store.profiles(10)] == ["member", "member-second"]
    assert store.profiles(8) == []
    assert store.selected_profile(8) is None
    selected = store.select_profile(10, "member-second")
    assert store.selected_profile(10) == selected
    owner_profile = store.selected_profile(7)
    assert owner_profile is not None and owner_profile.id == "owner"
    changed = store.update_voice_settings(10, "member", speed=0.9, delivery="whisper")
    assert changed.speed == 0.9 and changed.effective_native_tags == ("whisper",)
    draft = store.begin_profile_draft(10, "rename", "member")
    assert store.rename_profile(200, 10, draft.id, "Мой голос")
    assert store.get_profile(10, "member").name == "Мой голос"
    assert (store.data_dir / "profiles" / "owner" / "profile.json").read_bytes() == owner_bytes
    # Even an administrator has no bypass into another person's catalog.
    for user_id in (7, 8, 9, 501):
        with pytest.raises(NoProfile):
            store.get_profile(user_id, "member")
        with pytest.raises(NoProfile):
            store.select_profile(user_id, "member")
        with pytest.raises(NoProfile):
            store.enqueue(None, user_id, "tts", text="Чужой", profile_id="member")


@pytest.mark.parametrize(
    "operation",
    ["settings", "tag", "parameter", "rename", "delete", "setting", "reference", "sample"],
)
def test_every_profile_mutation_checks_personal_ownership(
    personal_store: Store, operation: str
) -> None:
    store = personal_store
    with pytest.raises(NoProfile):
        if operation == "settings":
            store.update_voice_settings(8, "member", speed=1.1)
        elif operation == "tag":
            store.toggle_native_tag(8, "member", "whisper")
        elif operation == "parameter":
            store.adjust_voice_setting(8, "member", "t", 1)
        elif operation == "setting":
            store.begin_profile_draft(8, "setting", "member", "style_instruction")
        elif operation == "reference":
            store.begin_reference_add(8, "member")
        elif operation == "sample":
            store.reference_sample_path(8, "member", 0)
        else:
            store.begin_profile_draft(8, "rename" if operation == "rename" else "delete", "member")


def test_unassigned_profile_files_are_hidden_after_migration(personal_store: Store) -> None:
    store = personal_store
    add_profile(store.data_dir, "unassigned")
    with sqlite3.connect(store.db) as connection:
        connection.execute("DELETE FROM profile_owners WHERE profile_id='unassigned'")
    store.initialize()
    assert [profile.id for profile in store.profiles(7)] == ["owner"]
    with pytest.raises(NoProfile):
        store.get_profile(7, "unassigned")
    assert (store.data_dir / "profiles" / "unassigned" / "reference.wav").is_file()


def test_profile_preparation_assigns_creator_and_preserves_other_selection(
    personal_store: Store,
) -> None:
    store = personal_store
    original = store.selected_profile(7)
    job_id = store.enqueue(101, 501, "profile", text="Личный голос", source=output(store))
    job = store.claim()
    assert job is not None and job.id == job_id
    prepared = store.job_dir(job.id) / "prepared"
    prepared.mkdir()
    (prepared / "reference.wav").write_bytes(b"prepared original recording")
    (prepared / "example.ogg").write_bytes(b"new profile preview")
    profile = store.finish_profile(job.id, job.lease, prepared, "Чистый образец речи.")
    assert store.selected_profile(501) == profile
    assert store.profiles(501) == [profile]
    assert store.selected_profile(7) == original
    assert store.account(501).available_ms == 600_000
    with pytest.raises(NoProfile):
        store.get_profile(7, profile.id)


def test_personal_reference_candidates_and_deletion_stay_private(personal_store: Store) -> None:
    store = personal_store
    source = output(store)
    change = store.begin_reference_add(10, "member")
    assert store.reference_change(8, "member") is None
    assert not store.discard_reference_change(8, change.id)
    assert store.apply_reference_change(8, change.id) is None
    with pytest.raises(NoProfile):
        store.reference_change_preview_path(8, change.id)
    with pytest.raises(ValueError):
        store.enqueue_reference_add(100, 8, change.id, source)
    assert store.reference_change(10, "member") == change
    draft = store.begin_profile_draft(10, "delete", "member")
    assert not store.delete_profile(8, draft.id)
    assert store.delete_profile(10, draft.id)
    assert store.selected_profile(10) is None
    assert store.account(10).balance_ms == 600_000
    assert store.get_profile(7, "owner").id == "owner"


@pytest.mark.parametrize("kind", ["tts", "vc"])
def test_actual_milliseconds_are_charged_once_and_delivery_retries_are_free(
    personal_store: Store, kind: Literal["tts", "vc"]
) -> None:
    store = personal_store
    source = output(store, "input.audio") if kind == "vc" else None
    job_id = store.enqueue(101, 10, kind, text="Текст" if kind == "tts" else None, source=source)
    assert store.account(10) == Account(10, False, 600_000, 180_000)
    assert (
        store.enqueue(101, 10, kind, text="Текст" if kind == "tts" else None, source=source) is None
    )
    assert store.account(10).reserved_ms == 180_000
    job = store.claim()
    assert job is not None and job.id == job_id
    if kind == "vc":
        store.record_transcript(job.id, job.lease, "Распознанный текст")
    store.finish(job.id, job.lease, output(store), duration_ms=17_321)
    assert store.account(10) == Account(10, False, 582_679, 0)
    with pytest.raises(StaleLease):
        store.finish(job.id, job.lease, output(store), duration_ms=17_321)
    assert store.delivery_claim() is not None
    store.delivery_done(job.id, 81)
    assert store.retry_delivery(job.id, 10, "retry")
    assert store.delivery_claim() is not None
    store.delivery_done(job.id, 82)
    assert store.account(10).balance_ms == 582_679
    with sqlite3.connect(store.db) as connection:
        assert connection.execute(
            "SELECT state,charged_ms,reserved_ms FROM job_billing WHERE job_id=?", (job.id,)
        ).fetchone() == ("settled", 17_321, 0)


@pytest.mark.parametrize(
    "mode", ["queued_cancel", "running_cancel", "failure", "restart", "timeout", "retention"]
)
def test_failed_cancelled_and_expired_jobs_release_reserved_time(
    personal_store: Store, mode: str
) -> None:
    store = personal_store
    job_id = store.enqueue(101, 10, "tts", text="Текст")
    assert job_id is not None
    if mode == "queued_cancel":
        assert store.request_cancel(job_id, 10) == "cancelled"
    elif mode == "retention":
        store.maintain(time.time() + RETENTION_SECONDS + 1)
    else:
        job = store.claim()
        assert job is not None
        if mode == "running_cancel":
            assert store.request_cancel(job_id, 10) == "requested"
            store.confirm_cancel(job_id, job.lease)
        elif mode == "failure":
            store.fail(job_id, job.lease, "model_error")
        elif mode == "restart":
            store.recover()
        else:
            store.maintain(time.time() + 1000)
    assert store.account(10) == Account(10, False, 600_000, 0)
    store.maintain(time.time() + RETENTION_SECONDS + 1)
    assert store.account(10) == Account(10, False, 600_000, 0)


def test_parallel_jobs_cannot_spend_or_reserve_the_same_second(personal_store: Store) -> None:
    store = personal_store
    with sqlite3.connect(store.db) as connection:
        connection.execute("UPDATE accounts SET balance_ms=1000 WHERE user_id=10")
    other = Store(store.db, store.data_dir, 7, admin_user_ids=frozenset({7, 8, 9}))

    def enqueue(selected: Store) -> str | None:
        try:
            return selected.enqueue(None, 10, "tts", text="Текст")
        except InsufficientBalance:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(enqueue, (store, other)))
    assert sum(result is not None for result in results) == 1
    assert store.account(10) == Account(10, False, 1000, 1000)
    job = store.claim()
    assert job is not None
    store.finish(job.id, job.lease, output(store), duration_ms=501)
    assert store.account(10) == Account(10, False, 499, 0)
    assert other.enqueue(None, 10, "tts", text="Следующее") is not None
    assert store.account(10).reserved_ms == 499


def test_longer_than_balance_result_is_rejected_without_charging_or_publishing(
    personal_store: Store,
) -> None:
    store = personal_store
    with sqlite3.connect(store.db) as connection:
        connection.execute("UPDATE accounts SET balance_ms=1000 WHERE user_id=10")
    store.enqueue(101, 10, "tts", text="Текст")
    job = store.claim()
    assert job is not None
    result = output(store)
    with pytest.raises(InsufficientBalance):
        store.finish(job.id, job.lease, result, duration_ms=1001)
    assert result.is_file()
    assert not (store.job_dir(job.id) / "result.ogg").exists()
    assert store.account(10) == Account(10, False, 1000, 0)
    failed = store.list_jobs(10)[0]
    assert failed.status == "failed" and failed.failure == "insufficient_balance"
    assert store.delivery_claim() is None


def test_empty_trial_blocks_new_speech_but_allows_only_fixed_short_previews(
    personal_store: Store,
) -> None:
    store = personal_store
    with sqlite3.connect(store.db) as connection:
        connection.execute("UPDATE accounts SET balance_ms=0 WHERE user_id=10")
    with pytest.raises(InsufficientBalance):
        store.enqueue(100, 10, "tts", text="Обычный текст", profile_id="member")
    with pytest.raises(ValueError):
        store.enqueue(None, 10, "tts", text="Любой текст", profile_id="member", preview=True)
    with pytest.raises(ValueError):
        store.enqueue(None, 10, "tts", text=PREVIEW_TEXT, preview=True)
    store.enqueue(None, 10, "tts", text=PREVIEW_TEXT, profile_id="member", preview=True)
    job = store.claim()
    assert job is not None
    with pytest.raises(ValueError):
        store.finish(job.id, job.lease, output(store), duration_ms=30_001)
    store.finish(job.id, job.lease, output(store), duration_ms=12_001)
    assert store.account(10) == Account(10, False, 0, 0)
    assert not store.update_seen(100)


def test_admin_job_remains_free_after_role_change(personal_store: Store) -> None:
    store = personal_store
    add_profile(store.data_dir, "admin", user_id=8)
    store.enqueue(100, 8, "tts", text="Безлимит")
    job = store.claim()
    assert job is not None
    # A running request keeps the exemption granted when it was accepted.
    demoted = Store(store.db, store.data_dir, 7)
    demoted.initialize()
    demoted.finish(job.id, job.lease, output(store), duration_ms=180_000)
    assert demoted.account(8) == Account(8, False, 0, 0)
    with pytest.raises(InsufficientBalance):
        demoted.enqueue(101, 8, "tts", text="Теперь обычный аккаунт")


def test_v7_migration_preserves_existing_audio_candidates_and_owner_selection(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "queue.sqlite3", tmp_path, 7)
    store.initialize()
    add_profile(tmp_path, "anna")
    add_profile(tmp_path, "maria")
    store.select_profile(7, "maria")
    candidate = ready_candidate(store)
    store.replace_panel(7, 11, "Моя карточка", '{"inline_keyboard":[]}')
    store.replace_panel(8, 12, "Чужая старая карточка", '{"inline_keyboard":[]}')
    store.replace_panel(10, 13, "Чужая старая карточка", '{"inline_keyboard":[]}')
    files = {
        path.relative_to(tmp_path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("profiles", "reference-changes", "jobs")
        for path in (tmp_path / folder).rglob("*")
        if path.is_file()
    }
    with sqlite3.connect(store.db) as connection:
        drop_history_schema(connection)
        connection.execute("DROP TABLE job_billing")
        connection.execute("DROP TABLE profile_owners")
        connection.execute("DROP TABLE accounts")
        connection.execute("INSERT INTO settings(key,value) VALUES('profile','maria')")
        connection.execute("PRAGMA user_version=7")
    migrated = Store(store.db, tmp_path, 7, admin_user_ids=frozenset({7, 8, 9}))
    migrated.initialize()
    selected = migrated.selected_profile(7)
    assert selected is not None and selected.id == "maria"
    assert migrated.profiles(8) == [] and migrated.profiles(10) == []
    assert migrated.panel_content(7) is not None
    assert migrated.panel_id(8) == 12 and migrated.panel_content(8) is None
    assert migrated.panel_id(10) == 13 and migrated.panel_content(10) is None
    assert migrated.reference_change(7, "anna") == candidate
    assert migrated.reference_change(8, "anna") is None
    assert all(
        hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == digest
        for path, digest in files.items()
    )
    with sqlite3.connect(store.db) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
        assert (
            connection.execute("SELECT value FROM settings WHERE key='profile'").fetchone() is None
        )
    migrated.initialize()
    assert migrated.account(8) == Account(8, True, 0, 0)
    assert migrated.account(10) == Account(10, False, 600_000, 0)


def test_failed_result_publication_rolls_back_charge(personal_store: Store) -> None:
    store = personal_store
    store.enqueue(101, 10, "tts", text="Текст")
    job = store.claim()
    assert job is not None
    with pytest.raises(FileNotFoundError):
        store.finish(job.id, job.lease, store.data_dir / "missing.ogg", duration_ms=17_321)
    assert store.account(10) == Account(10, False, 600_000, 180_000)
    store.fail(job.id, job.lease, "storage_error")
    assert store.account(10) == Account(10, False, 600_000, 0)
    assert store.delivery_claim() is None


def create_default_template(store: Store, source_profile_id: str = "owner") -> Path:
    """Use opaque local audio for storage/UI tests; API tests validate actual recordings."""
    profile = store.get_profile(store.owner_id, source_profile_id)
    directory = store.data_dir / "default-voice"
    directory.mkdir(mode=0o700)
    source = store.data_dir / "profiles" / source_profile_id
    for sample in profile.reference_samples:
        (directory / sample.reference_audio).write_bytes(
            (source / sample.reference_audio).read_bytes()
        )
    (directory / "profile.json").write_text(
        profile.model_copy(update={"name": "Вика"}).model_dump_json()
    )
    (directory / "example.ogg").write_bytes(b"short synthesized default voice preview")
    (directory / "origin.json").write_text(
        json.dumps({"source_profile_id": source_profile_id, "owner_id": store.owner_id})
    )
    if (source / "whisper-context.json").exists():
        (directory / "whisper-context.json").write_bytes(
            (source / "whisper-context.json").read_bytes()
        )
    return directory


def test_default_voice_backfills_accounts_without_changing_owner_or_existing_choice(
    personal_store: Store,
) -> None:
    store = personal_store
    original = store.get_profile(7, "owner")
    source_path = store.data_dir / "profiles" / "owner"
    (source_path / "whisper-context.json").write_bytes(b"approved whisper context")
    originals = {path.name: path.read_bytes() for path in source_path.iterdir()}
    member_selected = store.select_profile(10, "member")
    with sqlite3.connect(store.db) as connection:
        connection.execute("UPDATE accounts SET balance_ms=12345 WHERE user_id=10")
    template = create_default_template(store)
    store.initialize()
    assert store.get_profile(7, "owner") == original
    assert store.profiles(7) == [original]
    assert {path.name: path.read_bytes() for path in source_path.iterdir()} == originals
    assert not store.is_default_voice(7, "owner")
    assert store.selected_profile(10) == member_selected
    assert store.account(10).balance_ms == 12_345
    for user_id in (8, 9, 10):
        voices = [
            voice for voice in store.profiles(user_id) if store.is_default_voice(user_id, voice.id)
        ]
        assert len(voices) == 1 and voices[0].name == "Вика"
        clone = store.data_dir / "profiles" / voices[0].id
        for filename in ("reference.wav", "example.ogg", "whisper-context.json"):
            assert (clone / filename).read_bytes() == (template / filename).read_bytes()
            assert (clone / filename).stat().st_ino != (template / filename).stat().st_ino
            assert (clone / filename).stat().st_mode & 0o777 == 0o600
        assert (clone / "profile.json").stat().st_mode & 0o777 == 0o600
    first = store.selected_profile(8)
    assert first is not None
    changed = store.update_voice_settings(8, first.id, delivery="whisper", speed=0.9)
    store.initialize()
    assert store.get_profile(8, first.id) == changed
    second = store.selected_profile(9)
    assert second is not None and second.speed == 1.0
    assert store.get_profile(7, "owner") == original


def test_default_voice_for_new_users_survives_source_deletion_but_not_personal_deletion(
    personal_store: Store,
) -> None:
    store = personal_store
    create_default_template(store)
    store.initialize()
    assert store.account(501) == Account(501, False, 600_000, 0)
    first = store.selected_profile(501)
    assert first is not None and first.name == "Вика"
    draft = store.begin_profile_draft(501, "rename", first.id)
    assert store.rename_profile(1000, 501, draft.id, "Мой стартовый голос")
    delete_original = store.begin_profile_draft(7, "delete", "owner")
    assert store.delete_profile(7, delete_original.id)
    second = store.selected_profile(502)
    assert second is not None and second.name == "Вика" and second.id != first.id
    assert store.account(502) == Account(502, False, 600_000, 0)
    remove = store.begin_profile_draft(501, "delete", first.id)
    assert store.delete_profile(501, remove.id)
    store.initialize()
    assert store.account(501).balance_ms == 600_000
    assert store.profiles(501) == [] and store.selected_profile(501) is None
    assert store.profiles(7) == []
    assert store.selected_profile(502) == second


def test_included_voice_hides_originals_without_restricting_personal_controls(
    personal_store: Store,
) -> None:
    store = personal_store
    create_default_template(store)
    store.initialize()
    profile = store.selected_profile(501)
    assert profile is not None and store.is_default_voice(501, profile.id)
    with pytest.raises(NoProfile):
        store.is_default_voice(502, profile.id)
    with pytest.raises(PermissionError):
        store.reference_sample_path(501, profile.id, 0)
    with pytest.raises(PermissionError):
        store.begin_reference_add(501, profile.id)
    with pytest.raises(PermissionError):
        store.enqueue_reference_remove(501, profile.id, 0)
    # The person who provided the source still controls their original recordings.
    assert store.reference_sample_path(7, "owner", 0).is_file()
    assert store.begin_reference_add(7, "owner").profile_id == "owner"
    assert store.update_voice_settings(501, profile.id, speed=1.1).speed == 1.1
    assert store.enqueue(500, 501, "tts", text="Мой текст") is not None


def test_parallel_signup_creates_exactly_one_default_voice(personal_store: Store) -> None:
    store = personal_store
    create_default_template(store)
    other = Store(store.db, store.data_dir, 7, admin_user_ids=frozenset({7, 8, 9}))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda selected: selected.account(501), (store, other)))
    assert results == [Account(501, False, 600_000, 0)] * 2
    assert len(store.profiles(501)) == 1
    assert list((store.data_dir / "profile-trash").glob("default-*.json")) == []
    assert other.selected_profile(501) == store.selected_profile(501)


def test_failed_default_copy_rolls_back_account_and_partial_files(
    personal_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = personal_store
    create_default_template(store)
    original = shutil.copyfile
    original_profiles = {path.name for path in (store.data_dir / "profiles").iterdir()}

    def broken_copy(source: str | Path, target: str | Path) -> str | Path:
        if Path(source).name == "reference.wav":
            raise OSError("Interrupted local copy")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr("shift.store.shutil.copyfile", broken_copy)
        with pytest.raises(OSError, match="Interrupted local copy"):
            store.account(501)
    assert {path.name for path in (store.data_dir / "profiles").iterdir()} == original_profiles
    assert list((store.data_dir / "profile-trash").glob("default-*.json")) == []
    with sqlite3.connect(store.db) as connection:
        assert connection.execute("SELECT 1 FROM accounts WHERE user_id=501").fetchone() is None
    assert store.account(501) == Account(501, False, 600_000, 0)
    assert len(store.profiles(501)) == 1


@pytest.mark.parametrize(
    "name", ["reference.wav", "profile.json", "example.ogg", "origin.json", "whisper-context.json"]
)
def test_default_template_rejects_symbolic_links(personal_store: Store, name: str) -> None:
    store = personal_store
    template = create_default_template(store)
    link = template / name
    link.unlink(missing_ok=True)
    link.symlink_to(store.data_dir / "profiles" / "owner" / "reference.wav")
    with pytest.raises(ValueError, match="regular private files"):
        store.account(501)
    with sqlite3.connect(store.db) as connection:
        assert connection.execute("SELECT 1 FROM accounts WHERE user_id=501").fetchone() is None


def test_default_template_rejects_directory_symbolic_link(personal_store: Store) -> None:
    store = personal_store
    (store.data_dir / "default-voice").symlink_to(store.data_dir / "profiles" / "owner")
    with pytest.raises(ValueError, match="symbolic link"):
        store.account(501)


@pytest.mark.parametrize("committed", [False, True])
def test_default_copy_crash_recovery_keeps_only_committed_profiles(
    personal_store: Store, committed: bool
) -> None:
    store = personal_store
    create_default_template(store)
    if committed:
        selected = store.selected_profile(501)
        assert selected is not None
        profile_id = selected.id
    else:
        profile_id = "voice-" + "b" * 32
        directory = store.data_dir / "profiles" / profile_id
        directory.mkdir()
        (directory / "reference.wav").write_bytes(b"incomplete copy")
    marker = store.data_dir / "profile-trash" / f"default-{profile_id.removeprefix('voice-')}.json"
    marker.write_text(json.dumps({"profile_id": profile_id, "user_id": 501}))
    reopened = Store(store.db, store.data_dir, 7, admin_user_ids=frozenset({7, 8, 9}))
    reopened.initialize()
    assert not marker.exists()
    assert (store.data_dir / "profiles" / profile_id).exists() is committed
    if committed:
        assert reopened.get_profile(501, profile_id).id == profile_id
    else:
        assert reopened.account(501) == Account(501, False, 600_000, 0)
        assert len(reopened.profiles(501)) == 1


def test_initial_owner_is_persisted_and_admin_reordering_does_not_change_privacy(
    personal_store: Store,
) -> None:
    store = personal_store
    create_default_template(store)
    store.initialize()
    reopened = Store(store.db, store.data_dir, 8, admin_user_ids=frozenset({8, 9, 10}))
    reopened.initialize()
    assert reopened.owner_id == 7
    assert not reopened.account(7).is_admin
    assert reopened.account(8) == Account(8, True, 0, 0)
    assert reopened.account(10) == Account(10, True, 0, 0)
    assert [profile.id for profile in reopened.profiles(7)] == ["owner"]
    assert reopened.get_profile(7, "owner").id == "owner"
    with pytest.raises(NoProfile):
        reopened.get_profile(8, "owner")
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute(
                "SELECT value FROM settings WHERE key='initial_owner_id'"
            ).fetchone()[0]
            == "7"
        )


def test_telegram_identity_updates_only_own_profile_and_can_remove_username(
    personal_store: Store,
) -> None:
    store = personal_store
    before = store.account(10)
    account = store.update_account_user(10, "user_name", "Иван 👨‍👩‍👧‍👦")
    assert account.username == "user_name" and account.display_name == "Иван 👨‍👩‍👧‍👦"
    assert account.balance_ms == before.balance_ms and account.is_admin is before.is_admin
    assert store.account(7).username is None and store.account(7).display_name is None
    reopened = Store(store.db, store.data_dir, 7, admin_user_ids=frozenset({7, 8, 9}))
    reopened.initialize()
    assert reopened.account(10) == account
    renamed = reopened.update_account_user(10, None, "Новое имя")
    assert renamed.username is None and renamed.display_name == "Новое имя"
    assert "Новое имя" not in repr(renamed)
    assert "user_name" not in repr(account)
    for username in ("@name", "x" * 33):
        with pytest.raises(ValueError):
            reopened.update_account_user(10, username, "Имя")
    assert reopened.account(10) == renamed
    assert (
        reopened.update_account_user(10, None, "  Иван\nИванов\0  ").display_name == "Иван Иванов"
    )
    assert reopened.update_account_user(10, None, " \0 ").display_name is None
    long_name = reopened.update_account_user(10, None, "Я" * 300).display_name
    assert long_name is not None and len(long_name) == 256


def test_default_clone_keeps_all_reference_recordings_and_whisper_context(
    personal_store: Store,
) -> None:
    from shift.protocol import ReferenceSample

    store = personal_store
    source = store.get_profile(7, "owner")
    folder = store.data_dir / "profiles" / source.id
    samples = tuple(
        ReferenceSample(reference_audio=f"reference-{index}.wav", reference_text=f"Образец {index}")
        for index in (2, 3)
    )
    for sample in samples:
        (folder / sample.reference_audio).write_bytes(sample.reference_text.encode())
    original = source.model_copy(update={"extra_references": samples})
    (folder / "profile.json").write_text(original.model_dump_json())
    (folder / "whisper-context.json").write_bytes(b"approved whisper state")
    template = create_default_template(store)
    clone = store.selected_profile(501)
    assert clone is not None and len(clone.reference_samples) == 3
    target = store.data_dir / "profiles" / clone.id
    assert clone.reference_samples == original.reference_samples
    for filename in ("reference.wav", "reference-2.wav", "reference-3.wav", "whisper-context.json"):
        assert (target / filename).read_bytes() == (template / filename).read_bytes()
    assert store.get_profile(7, "owner") == original


def test_configured_default_template_cannot_silently_disappear(personal_store: Store) -> None:
    store = personal_store
    directory = create_default_template(store)
    store.initialize()
    directory.rename(store.data_dir / "hidden-default-template")
    with pytest.raises(ValueError, match="configured default voice template is missing"):
        store.account(501)
    with pytest.raises(ValueError, match="configured default voice template is missing"):
        store.initialize()
    with sqlite3.connect(store.db) as connection:
        assert connection.execute("SELECT 1 FROM accounts WHERE user_id=501").fetchone() is None


def test_uncommitted_default_clone_is_removed_before_repeating_v7_migration(
    personal_store: Store,
) -> None:
    store = personal_store
    create_default_template(store)
    profile_id = "voice-" + "b" * 32
    directory = store.data_dir / "profiles" / profile_id
    directory.mkdir()
    original = store.get_profile(7, "owner")
    (directory / "profile.json").write_text(
        original.model_copy(update={"id": profile_id}).model_dump_json()
    )
    (directory / "reference.wav").write_bytes(b"uncommitted default copy")
    marker = store.data_dir / "profile-trash" / f"default-{'b' * 32}.json"
    marker.write_text(json.dumps({"profile_id": profile_id, "user_id": 501}))
    with sqlite3.connect(store.db) as connection:
        drop_history_schema(connection)
        connection.execute("DROP TABLE job_billing")
        connection.execute("DROP TABLE profile_owners")
        connection.execute("DROP TABLE accounts")
        connection.execute("PRAGMA user_version=7")
    store.initialize()
    assert not directory.exists() and not marker.exists()
    assert profile_id not in {profile.id for profile in store.profiles(7)}
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM profile_owners WHERE profile_id=?", (profile_id,)
            ).fetchone()
            is None
        )
