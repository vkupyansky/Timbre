"""A processing notice has one durable job anchor and cannot overwrite the menu."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from shift.store import Store
from tests.test_accounts_store import output
from tests.test_store import add_profile


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = Store(tmp_path / "queue.sqlite3", tmp_path, 7)
    result.initialize()
    add_profile(tmp_path, "anna")
    add_profile(tmp_path, "member", user_id=8)
    return result


def queue(store: Store, chat_id: int = 7) -> str:
    identifier = store.enqueue(None, chat_id, "tts", text="Проверка статуса")
    assert identifier is not None
    return identifier


def test_status_lookup_claim_and_anchor_are_bound_to_the_job_owner(store: Store) -> None:
    job_id = queue(store)
    entry = store.status_entry(job_id, 7)
    assert entry is not None and entry[0].id == job_id and entry[1] is None
    assert store.status_entry(job_id, 8) is None
    assert not store.claim_status(job_id, 8)
    assert not store.attach_status(job_id, 7, 100, "queued:0")
    assert store.claim_status(job_id, 7)
    assert not store.attach_status(job_id, 8, 100, "queued:0")
    assert store.attach_status(job_id, 7, 100, "queued:0")
    assert not store.attach_status(job_id, 7, 101, "queued:0")
    assert not store.claim_status(job_id, 7)
    attached = store.status_entry(job_id, 7)
    assert attached is not None and attached[1] == 100
    assert store.status_updates() == []


def test_parallel_initial_announcements_can_claim_only_once(store: Store) -> None:
    job_id = queue(store)
    other = Store(store.db, store.data_dir, 7)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(
            pool.map(lambda selected: selected.claim_status(job_id, 7), (store, other) * 4)
        )
    assert sum(claims) == 1
    assert store.status_updates() == []
    assert store.attach_status(job_id, 7, 100, "queued:0")
    assert store.status_updates() == []


def test_menu_anchor_and_another_job_notice_cannot_be_reused(store: Store) -> None:
    first, second, third = queue(store), queue(store), queue(store, 8)
    store.replace_panel(7, 50, "Мои голоса", '{"inline_keyboard":[]}')
    assert store.claim_status(first, 7)
    assert not store.attach_status(first, 7, 50, "queued:0")
    assert store.attach_status(first, 7, 100, "queued:0")
    assert store.claim_status(second, 7)
    assert not store.attach_status(second, 7, 100, "queued:0")
    assert store.attach_status(second, 7, 101, "queued:0")
    # Telegram message IDs are unique inside one chat, not across all users.
    assert store.claim_status(third, 8)
    assert store.attach_status(third, 8, 100, "queued:0")
    assert store.panel_content(7) == ("Мои голоса", '{"inline_keyboard":[]}')


def test_completion_during_initial_send_is_updated_without_a_second_notice(store: Store) -> None:
    job_id = queue(store)
    assert store.claim_status(job_id, 7)
    job = store.claim()
    assert job is not None
    store.finish(job_id, job.lease, output(store), duration_ms=1000)
    # Telegram has only just confirmed the initial message showing the older queued state.
    assert store.attach_status(job_id, 7, 100, "queued:0")
    updates = store.status_updates()
    assert len(updates) == 1 and updates[0][1:] == (100, "ready:0")
    assert not store.status_notified(job_id, 100, "queued:0")
    assert not store.status_notified(job_id, 99, "ready:0")
    assert store.status_notified(job_id, 100, "ready:0")
    assert store.status_updates() == []


def test_new_job_completed_before_announcement_still_gets_one_correct_notice(store: Store) -> None:
    job_id = queue(store)
    job = store.claim()
    assert job is not None
    store.finish(job_id, job.lease, output(store), duration_ms=1000)
    assert store.delivery_claim() is not None
    store.delivery_done(job_id, 150)
    updates = store.status_updates()
    assert len(updates) == 1 and updates[0][1:] == (None, "sent:0")
    assert store.claim_status(job_id, 7)
    assert store.attach_status(job_id, 7, 151, "sent:0")
    assert store.status_updates() == []


def test_known_send_rejection_can_retry_but_uncertain_send_survives_restart(store: Store) -> None:
    job_id = queue(store)
    assert store.claim_status(job_id, 7)
    assert not store.release_status_claim(job_id, 8)
    assert store.release_status_claim(job_id, 7)
    assert not store.release_status_claim(job_id, 7)
    assert store.status_updates()[0][1:] == (None, "queued:0")
    assert store.claim_status(job_id, 7)
    # After a timeout, Telegram may have received the message. Never repeat that send.
    reopened = Store(store.db, store.data_dir, 7)
    reopened.initialize()
    reopened.recover()
    assert reopened.status_updates() == []
    assert not reopened.claim_status(job_id, 7)
    job = reopened.claim()
    assert job is not None
    reopened.finish(job_id, job.lease, output(store), duration_ms=1000)
    assert reopened.delivery_claim() is not None
    reopened.delivery_done(job_id, 150)
    assert reopened.status_updates() == []


def test_deleted_notice_is_not_recreated_and_stale_acknowledgement_is_ignored(store: Store) -> None:
    job_id = queue(store)
    assert store.claim_status(job_id, 7)
    assert store.attach_status(job_id, 7, 100, "queued:0")
    assert not store.drop_status(job_id, 8, 100)
    assert not store.drop_status(job_id, 7, 99)
    assert store.drop_status(job_id, 7, 100)
    assert not store.status_notified(job_id, 100, "queued:0")
    job = store.claim()
    assert job is not None
    store.fail(job_id, job.lease, "model_error")
    assert store.status_updates() == []
    assert not store.claim_status(job_id, 7)


def test_defensive_panel_guard_stops_a_notice_promoted_to_menu(store: Store) -> None:
    job_id = queue(store)
    assert store.claim_status(job_id, 7)
    assert store.attach_status(job_id, 7, 100, "queued:0")
    store.replace_panel(7, 100, "Главное меню", '{"inline_keyboard":[]}')
    job = store.claim()
    assert job is not None
    assert store.status_updates() == []
    entry = store.status_entry(job_id, 7)
    assert entry is not None and entry[1] is None
    assert not store.claim_status(job_id, 7)
    assert not store.status_notified(job_id, 100, "running:0")
    assert store.panel_content(7) == ("Главное меню", '{"inline_keyboard":[]}')


def test_transition_detaches_all_legacy_panels_and_never_resurrects_completed_jobs(
    store: Store,
) -> None:
    sent_id = queue(store)
    sent = store.claim()
    assert sent is not None
    store.finish(sent_id, sent.lease, output(store), duration_ms=1000)
    assert store.delivery_claim() is not None
    store.delivery_done(sent_id, 90)
    cancelled_id = queue(store)
    store.request_cancel(cancelled_id, 7)
    active_id = queue(store)
    store.replace_panel(7, 501, "Главное меню", '{"inline_keyboard":[]}')
    with sqlite3.connect(store.db) as connection:
        # The first anchor is an older menu that /start already replaced.
        connection.execute(
            "UPDATE jobs SET status_message_id=500,status_notified='queued:0' WHERE id=?",
            (sent_id,),
        )
        connection.execute(
            "UPDATE jobs SET status_message_id=501,status_notified='queued:0' WHERE id=?",
            (cancelled_id,),
        )
        connection.execute(
            "UPDATE jobs SET status_message_id=500,status_notified='queued:0' WHERE id=?",
            (active_id,),
        )
        connection.execute("DELETE FROM settings WHERE key='separate_job_status_messages'")
        audit_before = connection.execute(
            "SELECT * FROM generation_history ORDER BY job_id"
        ).fetchall()
        billing_before = connection.execute("SELECT * FROM job_billing ORDER BY job_id").fetchall()
    reopened = Store(store.db, store.data_dir, 7)
    reopened.initialize()
    updates = reopened.status_updates()
    assert len(updates) == 1 and updates[0][0].id == active_id and updates[0][1] is None
    assert not reopened.claim_status(sent_id, 7)
    assert not reopened.claim_status(cancelled_id, 7)
    assert reopened.claim_status(active_id, 7)
    assert reopened.attach_status(active_id, 7, 600, "queued:0")
    reopened.initialize()
    entry = reopened.status_entry(active_id, 7)
    assert entry is not None and entry[1] == 600
    assert reopened.status_updates() == []
    assert reopened.panel_content(7) == ("Главное меню", '{"inline_keyboard":[]}')
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute("SELECT * FROM generation_history ORDER BY job_id").fetchall()
            == audit_before
        )
        assert (
            connection.execute("SELECT * FROM job_billing ORDER BY job_id").fetchall()
            == billing_before
        )


def test_notice_updates_do_not_change_charges_or_generation_history(store: Store) -> None:
    job_id = queue(store, 8)
    job = store.claim()
    assert job is not None
    store.finish(job_id, job.lease, output(store), duration_ms=17_321)
    balance = store.account(8)
    with sqlite3.connect(store.db) as connection:
        before = connection.execute(
            "SELECT * FROM generation_history WHERE job_id=?", (job_id,)
        ).fetchone()
    assert store.claim_status(job_id, 8)
    assert store.attach_status(job_id, 8, 100, "ready:0")
    assert store.status_notified(job_id, 100, "ready:0")
    assert store.drop_status(job_id, 8, 100)
    assert store.account(8) == balance
    with sqlite3.connect(store.db) as connection:
        assert (
            connection.execute(
                "SELECT * FROM generation_history WHERE job_id=?", (job_id,)
            ).fetchone()
            == before
        )


@pytest.mark.parametrize("state", ["queued", "queued:2", "announcing", "sent:0:extra", "unknown:0"])
def test_invalid_display_states_cannot_be_stored(store: Store, state: str) -> None:
    job_id = queue(store)
    assert store.claim_status(job_id, 7)
    with pytest.raises(ValueError):
        store.attach_status(job_id, 7, 100, state)


@pytest.mark.parametrize("title", ["<b>Озвучка</b>", "<b>Новый голос</b>"])
def test_transition_clears_only_old_processing_menu_base_and_keeps_its_anchor(
    store: Store, title: str
) -> None:
    store.replace_panel(7, 500, title, '{"inline_keyboard":[]}')
    store.replace_panel(8, 501, "Настройки моего голоса", '{"inline_keyboard":[]}')
    with sqlite3.connect(store.db) as connection:
        connection.execute("DELETE FROM settings WHERE key='separate_job_status_messages'")
    store.initialize()
    assert store.panel_id(7) == 500 and store.panel_content(7) is None
    assert store.panel_content(8) == ("Настройки моего голоса", '{"inline_keyboard":[]}')
