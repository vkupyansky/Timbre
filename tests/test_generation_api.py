import json
import sqlite3

import pytest

from shift.app import RUNTIME
from tests.test_api import AppTestClient, headers
from tests.test_api import client as client
from tests.test_audio import make_pcm
from tests.test_bot import make_ogg


async def test_transcript_is_private_bounded_and_saved_once_for_its_active_job(
    client: AppTestClient,
) -> None:
    store = client.app[RUNTIME].store
    source = store.data_dir / "speech.wav"
    make_pcm(source, 2)
    identifier = store.enqueue(101, 7, "vc", source=source)
    claimed = store.claim()
    assert claimed is not None and claimed.id == identifier
    endpoint = f"/worker/jobs/{claimed.id}/transcript"
    text = "Здесь стоит зАмок. А здесь — замОк."
    assert (await client.post(endpoint, json={"text": text})).status == 401
    assert (
        await client.post(endpoint, headers=headers(client, "foreign"), json={"text": text})
    ).status == 409
    for invalid in ({"text": " "}, {"text": "x\x00"}, {"text": "x" * 2001}, {"text": 5}):
        assert (
            await client.post(endpoint, headers=headers(client, claimed.lease), json=invalid)
        ).status == 400
    assert (
        await client.post(endpoint, headers=headers(client, claimed.lease), data=b" " * 25000)
    ).status == 413
    for _ in range(2):
        response = await client.post(
            endpoint, headers=headers(client, claimed.lease), json={"text": text}
        )
        assert response.status == 200 and await response.json() == {"status": "saved"}
    assert (
        await client.post(
            endpoint, headers=headers(client, claimed.lease), json={"text": "Другой текст"}
        )
    ).status == 400
    result = store.data_dir / "result.ogg"
    await make_ogg(result)
    response = await client.post(
        f"/worker/jobs/{claimed.id}/result",
        headers=headers(client, claimed.lease),
        data=result.read_bytes(),
    )
    assert response.status == 200
    with sqlite3.connect(store.db) as connection:
        assert connection.execute(
            "SELECT transcript FROM generation_history WHERE job_id=?", (claimed.id,)
        ).fetchone() == (text,)
    assert (
        await client.post(endpoint, headers=headers(client, claimed.lease), json={"text": text})
    ).status == 409


@pytest.mark.parametrize("cancelled", [False, True])
async def test_transcript_cannot_replace_typed_text_or_write_after_cancel(
    client: AppTestClient, cancelled: bool
) -> None:
    store = client.app[RUNTIME].store
    source = store.data_dir / "speech.wav"
    make_pcm(source, 2)
    identifier = (
        store.enqueue(101, 7, "vc", source=source)
        if cancelled
        else store.enqueue(101, 7, "tts", text="Оригинальный текст")
    )
    claimed = store.claim()
    assert claimed is not None and claimed.id == identifier
    if cancelled:
        store.request_cancel(claimed.id, 7)
    response = await client.post(
        f"/worker/jobs/{claimed.id}/transcript",
        headers=headers(client, claimed.lease),
        json={"text": "Подмена"},
    )
    assert response.status == (409 if cancelled else 400)
    with sqlite3.connect(store.db) as connection:
        assert connection.execute(
            "SELECT transcript FROM generation_history WHERE job_id=?", (claimed.id,)
        ).fetchone() == (None,)


async def test_voice_result_cannot_skip_saving_recognized_text(client: AppTestClient) -> None:
    store = client.app[RUNTIME].store
    source = store.data_dir / "speech.wav"
    make_pcm(source, 2)
    store.enqueue(101, 7, "vc", source=source)
    claimed = store.claim()
    assert claimed is not None
    result = store.data_dir / "result.ogg"
    await make_ogg(result)
    response = await client.post(
        f"/worker/jobs/{claimed.id}/result",
        headers=headers(client, claimed.lease),
        data=result.read_bytes(),
    )
    assert response.status == 400
    assert store.list_jobs(7)[0].status == "running"
    assert not (store.job_dir(claimed.id) / "result.ogg").exists()


async def test_full_length_escaped_russian_transcript_is_accepted(client: AppTestClient) -> None:
    store = client.app[RUNTIME].store
    source = store.data_dir / "speech.wav"
    make_pcm(source, 2)
    store.enqueue(101, 7, "vc", source=source)
    claimed = store.claim()
    assert claimed is not None
    text = "Русский текст. " * 133 + "Салют"
    assert len(text) == 2000
    encoded = json.dumps({"text": text}, ensure_ascii=True).encode()
    assert len(encoded) > 8512
    response = await client.post(
        f"/worker/jobs/{claimed.id}/transcript",
        headers=headers(client, claimed.lease),
        data=encoded,
    )
    assert response.status == 200
