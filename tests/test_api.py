import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from shift.app import RUNTIME, AppConfig, configure_safe_logging, create_app
from shift.protocol import ClaimResponse
from tests.test_bot import make_ogg, make_reference
from tests.test_store import add_profile

type AppTestClient = TestClient[web.Request, web.Application]


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[AppTestClient]:
    add_profile(tmp_path, "anna")
    app = await create_app(
        AppConfig(owner_id=7, data_dir=tmp_path),
        "123:" + secrets.token_urlsafe(30),
        secrets.token_urlsafe(32),
        background=False,
    )
    async with TestClient(TestServer(app)) as instance:
        yield instance


def headers(client: AppTestClient, lease: str | None = None) -> dict[str, str]:
    result = {"Authorization": "Bearer " + client.app[RUNTIME].worker_token}
    if lease is not None:
        result["X-Shift-Lease"] = lease
    return result


async def test_api_authorization_and_health_reveal_no_secret(client: AppTestClient) -> None:
    response = await client.get("/health")
    assert response.status == 200
    body = await response.json()
    assert body == {"status": "ok", "worker_ready": False, "profile_ready": True}
    assert (await client.post("/worker/claim")).status == 401
    assert (await client.post("/worker/claim", headers={"Authorization": "wrong"})).status == 401
    assert (
        await client.post("/worker/claim", headers={"Authorization": "Bearer неверный"})
    ).status == 401
    assert (await client.post("/worker/claim", headers=headers(client))).status == 200
    response = await client.get("/health")
    assert (await response.json())["worker_ready"] is True


async def test_real_queue_api_audio_validation_and_fenced_completion(client: AppTestClient) -> None:
    store = client.app[RUNTIME].store
    store.enqueue(1, 7, "tts", text="Один тест.")
    response = await client.post("/worker/claim", headers=headers(client))
    job = ClaimResponse.model_validate_json(await response.read()).job
    assert job is not None
    assert job.text == "Один тест."
    endpoint = f"/worker/jobs/{job.id}"
    assert (
        await client.post(endpoint + "/heartbeat", headers=headers(client, "wrong"))
    ).status == 409
    response = await client.post(endpoint + "/heartbeat", headers=headers(client, job.lease))
    assert (await response.json()) == {"cancel": False}
    response = await client.post(
        endpoint + "/result", headers=headers(client, job.lease), data=b"invalid"
    )
    assert response.status == 422
    assert store.list_jobs(7)[0].status == "running"
    assert not list(store.job_dir(job.id).glob("upload-*.ogg"))
    result = store.data_dir / "test.ogg"
    await make_ogg(result)
    response = await client.post(
        endpoint + "/result",
        headers=headers(client, job.lease),
        data=result.read_bytes(),
    )
    assert response.status == 200
    assert store.list_jobs(7)[0].status == "ready"
    assert (store.job_dir(job.id) / "result.ogg").read_bytes() == result.read_bytes()
    assert (
        await client.post(endpoint + "/heartbeat", headers=headers(client, job.lease))
    ).status == 409
    assert (
        await client.post(
            endpoint + "/result", headers=headers(client, job.lease), data=result.read_bytes()
        )
    ).status == 409


async def test_api_cooperative_cancel_and_bounded_failure_enum(client: AppTestClient) -> None:
    store = client.app[RUNTIME].store
    store.enqueue(1, 7, "tts", text="Остановка.")
    response = await client.post("/worker/claim", headers=headers(client))
    job = ClaimResponse.model_validate_json(await response.read()).job
    assert job is not None
    endpoint = f"/worker/jobs/{job.id}"
    response = await client.post(
        endpoint + "/failed",
        headers=headers(client, job.lease),
        json={"reason": "private traceback or token must not be persisted"},
    )
    assert response.status == 400
    assert store.request_cancel(job.id, 7) == "requested"
    response = await client.post(endpoint + "/heartbeat", headers=headers(client, job.lease))
    assert (await response.json())["cancel"] is True
    response = await client.post(endpoint + "/cancelled", headers=headers(client, job.lease))
    assert response.status == 200
    assert store.list_jobs(7)[0].status == "cancelled"


async def test_second_app_fails_before_database_recovery(client: AppTestClient) -> None:
    store = client.app[RUNTIME].store
    store.enqueue(1, 7, "tts", text="Продолжается")
    assert store.claim() is not None
    other = await create_app(
        AppConfig(owner_id=7, data_dir=store.data_dir),
        "123:" + secrets.token_urlsafe(30),
        secrets.token_urlsafe(32),
        background=False,
    )
    runner = web.AppRunner(other)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await runner.setup()
        assert store.list_jobs(7)[0].status == "running"
    finally:
        await runner.cleanup()


async def test_queue_is_responsive_while_job_is_running(client: AppTestClient) -> None:
    store = client.app[RUNTIME].store
    store.enqueue(1, 7, "tts", text="Долгое задание")
    assert store.claim() is not None
    async with asyncio.timeout(2):
        responses = await asyncio.gather(*[client.get("/health") for _ in range(10)])
        assert all(response.status == 200 for response in responses)


def test_external_logs_keep_level_without_private_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_safe_logging()
    try:
        raise RuntimeError("https://example.invalid/bot/private-token/audio-private-message")
    except RuntimeError:
        logging.getLogger("aiogram.dispatcher").error("private user text", exc_info=True)
    assert "ERROR" in caplog.text and "RuntimeError" in caplog.text
    assert "private" not in caplog.text and "https://" not in caplog.text


async def test_profile_result_validates_and_publishes_from_empty_catalog(
    client: AppTestClient,
) -> None:
    store = client.app[RUNTIME].store
    deleting = store.begin_profile_draft(7, "delete", "anna")
    assert store.delete_profile(7, deleting.id)
    reference = store.data_dir / "reference.wav"
    example = store.data_dir / "example.ogg"
    await make_reference(reference)
    await make_ogg(example)  # Separate preview duration; not a re-encoded reference requirement.
    store.enqueue(100, 7, "profile", text="Анна новая", source=reference)
    response = await client.post("/worker/claim", headers=headers(client))
    job = ClaimResponse.model_validate_json(await response.read()).job
    assert job is not None and job.kind == "profile" and job.profile is None
    form = FormData()
    form.add_field("reference", reference.read_bytes(), filename="../../untrusted.wav")
    form.add_field("example", example.read_bytes(), filename="ignored.ogg")
    form.add_field("transcript", "Точный текст русского образца.")
    response = await client.post(
        f"/worker/jobs/{job.id}/profile_result",
        headers=headers(client, job.lease),
        data=form,
    )
    assert response.status == 200
    profile = store.selected_profile(7)
    assert profile is not None and profile.name == "Анна новая"
    assert profile.reference_text == "Точный текст русского образца."
    assert store.list_jobs(7)[0].status == "ready"
    assert store.list_jobs(7)[0].profile_id == profile.id
    assert (
        store.data_dir / "profiles" / profile.id / "reference.wav"
    ).read_bytes() == reference.read_bytes()
    assert (store.job_dir(job.id) / "result.ogg").read_bytes() == example.read_bytes()
    assert not list(store.job_dir(job.id).glob("profile-upload-*"))


async def test_profile_result_rejects_duplicate_fields_and_cancelled_publish(
    client: AppTestClient,
) -> None:
    store = client.app[RUNTIME].store
    source = store.data_dir / "reference.wav"
    preview = store.data_dir / "preview.ogg"
    await make_reference(source)
    await make_ogg(preview)
    store.enqueue(100, 7, "profile", text="Не публиковать", source=source)
    job = store.claim()
    assert job is not None
    endpoint = f"/worker/jobs/{job.id}/profile_result"
    duplicate = FormData()
    duplicate.add_field("reference", source.read_bytes(), filename="source.wav")
    duplicate.add_field("transcript", "Один")
    duplicate.add_field("transcript", "Два")
    response = await client.post(endpoint, headers=headers(client, job.lease), data=duplicate)
    assert response.status == 400
    assert not list(store.job_dir(job.id).glob("profile-upload-*"))
    assert store.request_cancel(job.id, 7) == "requested"
    form = FormData()
    form.add_field("reference", source.read_bytes(), filename="source.wav")
    form.add_field("example", preview.read_bytes(), filename="preview.ogg")
    form.add_field("transcript", "Речь.")
    response = await client.post(endpoint, headers=headers(client, job.lease), data=form)
    assert response.status == 409
    assert [profile.id for profile in store.profiles(7)] == ["anna"]
    assert store.list_jobs(7)[0].status == "running"
    store.confirm_cancel(job.id, job.lease)
    assert store.list_jobs(7)[0].status == "cancelled"
