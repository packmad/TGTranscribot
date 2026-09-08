# TGTranscribot

Minimal Telegram bot that:

- Authenticates users by **Telegram username** (allowlist)
- Accepts **audio files / voice notes**
- Transcribes audio using **OpenAI `gpt-transcribe`** by default
- Replies with the **transcript as a message** and also as a **timestamp-named `.txt` document**
- Does **not** store or re-forward past conversations (each reply is one-off)

## Under the hood

The default transcription model is **`gpt-transcribe`**. It can be overridden with `OPENAI_TRANSCRIBE_MODEL` or, for the CLI, `--model`.

For `gpt-transcribe`, TGTranscribot enables OpenAI's server-side `chunking_strategy="auto"`, which normalizes loudness and uses VAD to choose transcription boundaries.

The Transcriptions API has a **25 MB per-upload limit**. Files at or below that limit are uploaded directly when they are already in a supported format. Unsupported inputs are normalized with ffmpeg to **AAC in an `.m4a` container**. Files still above 25 MB are split locally into **64 kbps mono AAC/M4A chunks** before upload. The default local chunk duration is 2400 seconds (~40 minutes, ~19.2 MB of encoded audio plus container overhead), leaving margin below the API limit.

You can request smaller local chunks with `OPENAI_TRANSCRIBE_CHUNK_SECONDS`. The legacy `OPENAI_TRANSCRIBE_MAX_SECONDS` variable is still accepted as a fallback for backward compatibility. Values are clamped to 60–2400 seconds.

This replaces the old model-specific `gpt-4o-transcribe` ~1400-second splitting rule: chunking is now driven by the **25 MB upload limit**, while `gpt-transcribe` uses OpenAI's server-side automatic VAD chunking inside each upload.

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH` (most distro packages ship both)

## Setup

1. Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create your `.env` (copy from example):

```bash
cp .env.example .env
```

Set:

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `ALLOWED_USERNAMES` (comma/space separated Telegram usernames, without `@`)
- optionally `OPENAI_TRANSCRIBE_MODEL` (defaults to `gpt-transcribe`)
- optionally `OPENAI_TRANSCRIBE_CHUNK_SECONDS` (defaults to 2400; only relevant when local splitting is required)

3. Run:

```bash
# Export env vars (example using a local .env file)
set -a; source .env; set +a
python transcribot.py
```

## Command-line transcription (local)

Use `transcribe_cli.py` to transcribe a file on disk (prints the transcript to stdout and writes `<stem>_transcript.txt` next to the input unless you pass `-o`). Progress (conversion tiers, local segment i/N, etc.) goes to stderr so stdout stays pipe-friendly:

```bash
set -a; source .env; set +a   # needs OPENAI_API_KEY; optional OPENAI_TRANSCRIBE_MODEL
python transcribe_cli.py /path/to/audio.ogg
python transcribe_cli.py /path/to/audio.m4a -o /tmp/out.txt
python transcribe_cli.py /path/to/audio.wav --model gpt-transcribe
```

The Docker image is only set up to run the Telegram bot (`CMD ["python", "transcribot.py"]`); run the CLI on the host (or a custom image) with Python, `ffmpeg`, and the same dependencies.

## Docker

Build:

```bash
docker build -t tgtranscribot .
```

Run with an env-file (Docker injects env vars; the app does not parse the file):

```bash
docker run --rm --env-file .env tgtranscribot
```

## Docker Compose

Uses your local `.env` via `env_file`:

```bash
docker compose up --build
```

## Commands

- `/start`: basic help (authorized users only)
- Long transcripts are split across several chat messages when needed; the bot sends a short notice first and labels each part (`Part 1 of N`, …).
