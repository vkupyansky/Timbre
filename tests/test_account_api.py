import math
import secrets
import sqlite3
from pathlib import Path

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from shift.app import RUNTIME, AppConfig, create_app
from shift.audio import validate_result
from shift.store import InsufficientBalance
from tests.test_api import AppTestClient, headers
from tests.test_api import client as client
from tests.test_bot import make_ogg, make_reference
from tests.test_store import add_profile


async def test_output_decoding_controls_charge_and_retry_never_charges_again(
    client: AppTestClient,
) -> None:
    store = client.app[RUNTIME].store
    store.account(8)
    add_profile(store.data_dir, "private", user_id=8, db=store.db)
    job_id = store.enqueue(101, 8, "tts", text="Моя озвучка")
    claimed = store.claim()
    assert claimed is not None and claimed.id == job_id
    path = store.data_dir / "result.ogg"
    await make_ogg(path)
    duration_ms = math.ceil((await validate_result(path)).duration * 1000)
    endpoint = f"/worker/jobs/{job_id}/result"
    response = await client.post(
        endpoint,
        headers={**headers(client, claimed.lease), "X-Duration-Ms": "1"},
        data=path.read_bytes(),
    )
    assert response.status == 200
    assert store.account(8).balance_ms == 600_000 - duration_ms
    assert store.account(8).reserved_ms == 0
    assert store.account(7).balance_ms == 0
    assert store.delivery_claim() is not None
    store.delivery_done(claimed.id, 50)
    assert store.retry_delivery(claimed.id, 8, "one-retry")
    assert store.delivery_claim() is not None
    store.delivery_done(claimed.id, 51)
    assert store.account(8).balance_ms == 600_000 - duration_ms
    assert not store.retry_delivery(claimed.id, 7, "foreign-retry")
    repeated = await client.post(
        endpoint, headers=headers(client, claimed.lease), data=path.read_bytes()
    )
    assert repeated.status == 409
    assert store.account(8).balance_ms == 600_000 - duration_ms


@pytest.mark.parametrize("remaining_ms", [199, 200])
async def test_last_fraction_is_charged_exactly_or_released_without_partial_audio(
    client: AppTestClient, remaining_ms: int
) -> None:
    store = client.app[RUNTIME].store
    store.account(8)
    add_profile(store.data_dir, "private", user_id=8, db=store.db)
    with sqlite3.connect(store.db) as connection:
        connection.execute("UPDATE accounts SET balance_ms=? WHERE user_id=8", (remaining_ms,))
    store.enqueue(101, 8, "tts", text="Последняя фраза")
    claimed = store.claim()
    assert claimed is not None
    path = store.data_dir / "result.ogg"
    await make_ogg(path)
    assert (await validate_result(path)).duration == 0.2
    response = await client.post(
        f"/worker/jobs/{claimed.id}/result",
        headers=headers(client, claimed.lease),
        data=path.read_bytes(),
    )
    assert response.status == 200
    account = store.account(8)
    assert account.reserved_ms == 0
    if remaining_ms == 199:
        assert await response.json() == {"status": "failed", "reason": "insufficient_balance"}
        assert account.balance_ms == remaining_ms
        assert store.list_jobs(8)[0].status == "failed"
        assert not (store.job_dir(claimed.id) / "result.ogg").exists()
    else:
        assert await response.json() == {"status": "ready"}
        assert account.balance_ms == 0
        with pytest.raises(InsufficientBalance):
            store.enqueue(102, 8, "tts", text="Больше нет времени")
        assert not store.update_seen(102)


async def test_three_administrators_are_unlimited_and_the_next_user_gets_one_trial(
    tmp_path: Path,
) -> None:
    app = await create_app(
        AppConfig(7, tmp_path, admin_user_ids=frozenset({7, 8, 9})),
        "123:" + secrets.token_urlsafe(30),
        secrets.token_urlsafe(32),
        background=False,
    )
    async with TestClient(TestServer(app)) as client:
        store = app[RUNTIME].store
        for user_id in (7, 8, 9):
            assert store.account(user_id).is_admin
            assert store.account(user_id).balance_ms == 0
        assert not store.account(10).is_admin
        assert store.account(10).balance_ms == 600_000
        assert store.account(10).balance_ms == 600_000
        add_profile(tmp_path, "second-admin", user_id=8, db=store.db)
        store.enqueue(101, 8, "tts", text="Безлимит")
        claimed = store.claim()
        assert claimed is not None
        path = tmp_path / "result.ogg"
        await make_ogg(path)
        response = await client.post(
            f"/worker/jobs/{claimed.id}/result",
            headers=headers(client, claimed.lease),
            data=path.read_bytes(),
        )
        assert response.status == 200
        assert store.account(8).balance_ms == 0
        assert store.account(8).reserved_ms == 0
        assert store.profiles(7) == [] and store.profiles(9) == []


async def test_public_user_can_prepare_voice_without_affecting_owner_or_trial(
    client: AppTestClient,
) -> None:
    store = client.app[RUNTIME].store
    previous = store.selected_profile(7)
    reference = store.data_dir / "reference.wav"
    example = store.data_dir / "example.ogg"
    await make_reference(reference)
    await make_ogg(example)
    store.enqueue(111, 10, "profile", text="Личный голос", source=reference)
    claimed = store.claim()
    assert claimed is not None
    form = FormData()
    form.add_field("reference", reference.read_bytes(), filename="reference.wav")
    form.add_field("example", example.read_bytes(), filename="example.ogg")
    form.add_field("transcript", "Мой образец речи.")
    response = await client.post(
        f"/worker/jobs/{claimed.id}/profile_result",
        headers=headers(client, claimed.lease),
        data=form,
    )
    assert response.status == 200
    profile = store.selected_profile(10)
    assert profile is not None and profile.name == "Личный голос"
    assert store.account(10).balance_ms == 600_000
    assert store.selected_profile(7) == previous
    assert profile.id not in {voice.id for voice in store.profiles(7)}
