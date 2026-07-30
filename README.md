# TGTranscribot

Minimal Telegram bot that:

- Authenticates users by **Telegram username** (allowlist)
- Accepts **audio files / voice notes**
- Transcribes audio using **OpenAI**
- Replies with the **transcript as a message** and also as a **timestamp-named `.txt` document**
- Does **not** store or re-forward past conversations (each reply is one-off)


## Under the hood

Large inputs are re-encoded with ffmpeg to **AAC in an `.m4a` container** (bitrate ladder: 128 → 96 → 64 kbps stereo, then 48 kbps mono, then 32 kbps mono at 16 kHz) until under the 25 MB API limit. (Some OpenAI speech models reject certain FFmpeg MP3 outputs; M4A avoids that class of failures.) For **`gpt-4o-*-transcribe`** models, audio longer than the API limit (~1400 seconds) is split into time segments with ffmpeg/ffprobe, transcribed sequentially, and the text is joined with blank lines. Tune chunk length with **`OPENAI_TRANSCRIBE_MAX_SECONDS`** (default 1350). **`whisper-1`** is not split by duration in this app.


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

3. Run:

```bash
# Export env vars (example using a local .env file)
set -a; source .env; set +a
python transcribot.py
```

## Command-line transcription (local)

Use `transcribe_cli.py` to transcribe a file on disk (prints the transcript to stdout and writes `<stem>_transcript.txt` next to the input unless you pass `-o`). Progress (compress tiers, segment i/N, etc.) goes to stderr so stdout stays pipe-friendly:

```bash
set -a; source .env; set +a   # needs OPENAI_API_KEY; optional OPENAI_TRANSCRIBE_MODEL
python transcribe_cli.py /path/to/audio.ogg
python transcribe_cli.py /path/to/audio.m4a -o /tmp/out.txt
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


