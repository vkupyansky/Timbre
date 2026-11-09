FROM python:3.12-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg=7:7.1.5-0+deb13u1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir uv==0.11.28
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY NOTICE ./
COPY licenses/FISH-AUDIO-RESEARCH-LICENSE.md ./licenses/
COPY src ./src
RUN /opt/uv/bin/uv sync --frozen --no-dev --no-editable \
    && rm -f .venv/.gitignore \
    && useradd --system --uid 501 --create-home shift
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 501:20
ENTRYPOINT ["python", "-W", "error", "-m", "shift", "--state", "/state", "app", "--container"]
