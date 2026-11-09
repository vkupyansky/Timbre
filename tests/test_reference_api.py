import asyncio
import io
import json
from dataclasses import dataclass

import pytest
from aiohttp import FormData

from shift.app import RUNTIME
from shift.protocol import WorkerJob
from tests.test_api import AppTestClient, headers
from tests.test_api import client as client
from tests.test_audio import make_pcm
from tests.test_bot import make_ogg, make_reference


@dataclass(frozen=True)
class CandidateRequest:
    client: AppTestClient
    job: WorkerJob
    reference: bytes
    example: bytes
    texts: list[str]


@pytest.fixture
async def candidate_request(client: AppTestClient) -> CandidateRequest:
    store = client.app[RUNTIME].store
    source = store.data_dir / "normal-reference.wav"
    example = store.data_dir / "candidate-example.ogg"
    await make_reference(source)
    await make_ogg(example)
    reference = await asyncio.to_thread(source.read_bytes)
    await asyncio.to_thread((store.data_dir / "profiles/anna/reference.wav").write_bytes, reference)
    change = store.begin_reference_add(7, "anna")
    store.enqueue_reference_add(700, 7, change.id, source)
    job = store.claim()
    assert job is not None and job.profile is not None
    return CandidateRequest(
        client,
        job,
        reference,
        await asyncio.to_thread(example.read_bytes),
        [job.profile.reference_text, "Ещё одна запись того же человека."],
    )


def multipart(
    candidate: CandidateRequest, *, fault: str | None = None, new: bytes | None = None
) -> FormData:
    form = FormData()
    old = candidate.reference
    texts = list(candidate.texts)
    if fault == "changed_audio":
        old = old[:-2] + bytes([old[-2] ^ 1, old[-1]])
    elif fault == "changed_text":
        texts[0] = "Подменённая расшифровка прежнего образца."
    form.add_field("reference_0", io.BytesIO(old), filename="../../outside.wav")
    form.add_field("reference_1", io.BytesIO(new or candidate.reference), filename="ignored.wav")
    form.add_field("example", candidate.example, filename="ignored.ogg")
    form.add_field("reference_texts", json.dumps(texts, ensure_ascii=False))
    if fault == "duplicate":
        form.add_field("reference_0", old, filename="again.wav")
    elif fault == "unexpected":
        form.add_field("reference_2", old, filename="unexpected.wav")
    return form


@pytest.mark.parametrize(
    "fault,status",
    [
        ("wrong_lease", 409),
        ("duplicate", 400),
        ("unexpected", 400),
        ("changed_audio", 400),
        ("changed_text", 400),
        ("short_new", 422),
        ("regular_endpoint", 400),
    ],
)
async def test_reference_api_rejects_incomplete_or_tampered_results_without_publishing(
    candidate_request: CandidateRequest, fault: str, status: int
) -> None:
    candidate = candidate_request
    client, job = candidate.client, candidate.job
    store = client.app[RUNTIME].store
    original = store.selected_profile(7)
    new = None
    if fault == "short_new":
        source = store.data_dir / "short.wav"
        make_pcm(source, 2)
        new = source.read_bytes()
    if fault == "regular_endpoint":
        response = await client.post(
            f"/worker/jobs/{job.id}/result",
            headers=headers(client, job.lease),
            data=candidate.example,
        )
    else:
        response = await client.post(
            f"/worker/jobs/{job.id}/reference_result",
            headers=headers(client, "wrong" if fault == "wrong_lease" else job.lease),
            data=multipart(candidate, fault=fault, new=new),
        )
    assert response.status == status
    assert str(store.data_dir) not in await response.text()
    assert store.selected_profile(7) == original
    assert store.list_jobs(7)[0].status == "running"
    assert store.delivery_claim() is None
    assert not (store.job_dir(job.id) / "result.ogg").exists()
    assert not list(store.job_dir(job.id).glob("reference-upload-*"))
    assert not list(store.job_dir(job.id).glob("upload-*.ogg"))
    change = store.reference_change(7, "anna")
    assert change is not None and change.stage == "processing"


@pytest.mark.parametrize("new_rate", [24000, 44100])
async def test_valid_reference_result_only_prepares_candidate_and_fences_replay(
    candidate_request: CandidateRequest,
    new_rate: int,
) -> None:
    candidate = candidate_request
    client, job = candidate.client, candidate.job
    store = client.app[RUNTIME].store
    original = store.selected_profile(7)
    new_path = store.data_dir / "new-reference.wav"
    make_pcm(new_path, 15, rate=new_rate)
    new = new_path.read_bytes()
    endpoint = f"/worker/jobs/{job.id}/reference_result"
    response = await client.post(
        endpoint, headers=headers(client, job.lease), data=multipart(candidate, new=new)
    )
    assert response.status == 200
    assert await response.json() == {"status": "ready", "profile_id": "anna"}
    assert store.selected_profile(7) == original
    change = store.reference_change(7, "anna")
    assert change is not None and change.stage == "ready" and change.candidate_count == 2
    assert store.reference_change_preview_path(7, change.id).read_bytes() == candidate.example
    assert not (store.data_dir / "outside.wav").exists()
    assert not list(store.job_dir(job.id).glob("reference-upload-*"))
    repeated = await client.post(
        endpoint, headers=headers(client, job.lease), data=multipart(candidate)
    )
    assert repeated.status == 409
    applied = store.apply_reference_change(7, change.id)
    assert applied is not None and len(applied.reference_samples) == 2
    assert store.reference_sample_path(7, "anna", 0).read_bytes() == candidate.reference
    assert store.reference_sample_path(7, "anna", 1).read_bytes() == new
