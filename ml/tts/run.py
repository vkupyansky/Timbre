"""Private JSON request path in argv, safe technical summary on stdout."""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from pydantic import ValidationError


def main() -> int:
    started = time.perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.tts.request import Request
    from ml.tts.style_context import WhisperUnavailable

    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    stage = "validate_request"
    try:
        if args.request.stat().st_size > 64 * 1024:
            raise ValueError("request_too_large")
        request = Request.model_validate_json(args.request.read_text())
        stage = "model_generation"
        # Upstream's load diagnostics go to stderr; verbose=False avoids text logs.
        with contextlib.redirect_stdout(sys.stderr):
            from ml.tts.runtime import generate

            result = generate(request)
        result.elapsed_seconds = time.perf_counter() - started
        print(result.model_dump_json())
        return 0
    except WhisperUnavailable:
        print(json.dumps({"error": "WhisperUnavailable", "stage": stage}))
        return 76
    except (ValidationError, ValueError, OSError, RuntimeError, ImportError) as error:
        print(json.dumps({"error": type(error).__name__, "stage": stage}))
        if (
            stage == "model_generation"
            and isinstance(error, RuntimeError)
            and "out of memory" in str(error).lower()
        ):
            return 75
        return 1
    except Exception as error:
        # Boundary must not expose private prompts through upstream exceptions.
        print(json.dumps({"error": type(error).__name__, "stage": stage}))
        if isinstance(error, MemoryError):
            return 75
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
