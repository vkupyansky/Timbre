import asyncio
import fcntl
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from shift.audio import AudioError
from shift.worker import Job, reference_path, run_model, terminate


def test_worker_reference_cannot_escape_job_storage(tmp_path: Path) -> None:
    jobs = tmp_path / "data/jobs/a"
    jobs.mkdir(parents=True)
    reference = jobs / "reference.wav"
    reference.write_bytes(b"fixture")
    assert reference_path(tmp_path, "jobs/a/reference.wav") == reference
    outside = tmp_path / "private.wav"
    outside.write_bytes(b"private")
    for value in ("../private.wav", str(outside), "jobs/missing.wav"):
        with pytest.raises(AudioError):
            reference_path(tmp_path, value)
    (jobs / "escape.wav").symlink_to(outside)
    with pytest.raises(AudioError):
        reference_path(tmp_path, "jobs/a/escape.wav")


async def test_cancel_terminates_real_process_before_return(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    script = tmp_path / "child.py"
    script.write_text(
        "import pathlib,sys,time\n"
        "pathlib.Path(sys.argv[1]).write_text('started')\n"
        "time.sleep(120)\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        str(marker),
        start_new_session=True,
    )
    try:
        for _ in range(100):
            if await asyncio.to_thread(marker.exists):
                break
            await asyncio.sleep(0.01)
        assert await asyncio.to_thread(marker.exists)
        await terminate(process)
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        await terminate(process)  # Repeated cancellation is harmless.
    finally:
        await terminate(process)


def test_job_id_cannot_escape_or_delete_local_files() -> None:
    for identifier in ("../../secrets", "/tmp", "", "not-a-uuid"):
        with pytest.raises(ValidationError):
            Job.model_validate(
                {
                    "id": identifier,
                    "kind": "tts",
                    "text": "test",
                    "lease": "lease",
                    "profile": {
                        "id": "anna",
                        "name": "Anna",
                        "reference_audio": "jobs/ref.wav",
                        "reference_text": "reference",
                    },
                }
            )


async def test_cancel_while_gpu_is_busy_starts_no_model(tmp_path: Path) -> None:
    (tmp_path / ".shift").mkdir()
    cancel = asyncio.Event()
    watcher = asyncio.create_task(asyncio.sleep(60))
    with (tmp_path / ".shift/gpu.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        task = asyncio.create_task(
            run_model(tmp_path, "tts", tmp_path / "request", cancel, watcher)
        )
        try:
            await asyncio.sleep(0.05)
            assert not task.done()
            cancel.set()
            assert await asyncio.wait_for(task, 1) is None
        finally:
            task.cancel()
            watcher.cancel()
            await asyncio.gather(task, watcher, return_exceptions=True)


async def test_cancel_before_next_stage_starts_no_model(tmp_path: Path) -> None:
    (tmp_path / ".shift").mkdir()
    cancel = asyncio.Event()
    cancel.set()
    watcher = asyncio.create_task(asyncio.sleep(60))
    try:
        assert await run_model(tmp_path, "tts", tmp_path / "missing", cancel, watcher) is None
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
