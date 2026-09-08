FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade pip before installing any Python dependencies.
RUN python -m pip install --upgrade pip

# ffmpeg also provides ffprobe, both required for audio normalization/chunking.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Keep dependency installation in a separate cacheable layer.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# Run the bot as an unprivileged user with no login shell.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser transcribot.py ./

USER appuser

CMD ["python", "transcribot.py"]
