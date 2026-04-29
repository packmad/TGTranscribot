# TGTranscribot

Minimal Telegram bot that:

- Authenticates users by **Telegram username** (allowlist)
- Accepts **audio files / voice notes**
- Transcribes audio using **OpenAI**
- Replies with the **transcript as a message** and also as a **timestamp-named `.txt` document**
- Does **not** store or re-forward past conversations (each reply is one-off)

## Requirements

- Python 3.10+
- `ffmpeg` installed and available in `PATH`

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
python bot.py
```

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
