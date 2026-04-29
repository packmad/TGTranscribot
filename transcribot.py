import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


def _safe_username(u: Optional[str]) -> str:
    return (u or "").strip().lstrip("@").lower()


def _parse_allowed_usernames(raw: str) -> Set[str]:
    usernames: Set[str] = set()
    for part in re.split(r"[,\s]+", (raw or "").strip()):
        if part:
            usernames.add(_safe_username(part))
    return usernames


def _is_authorized(update: Update, allowed: Set[str]) -> bool:
    user = update.effective_user
    if not user:
        return False
    return _safe_username(user.username) in allowed


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")


def _ffmpeg_to_mp3(src: Path, dst: Path) -> None:
    _require_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")


def _maybe_convert_to_mp3(src: Path) -> Path:
    """
    OpenAI's transcription endpoint accepts several formats, but Telegram voice notes are often
    OGG/OPUS. Converting to MP3 keeps things predictable.
    """
    ext = src.suffix.lower().lstrip(".")
    if ext in {"mp3", "m4a", "wav", "webm", "mp4", "mpeg", "mpga", "oga", "ogg", "opus", "flac", "aac"}:
        # Convert everything except mp3 to mp3 for a uniform pipeline.
        if ext == "mp3":
            return src
        dst = src.with_suffix(".mp3")
        _ffmpeg_to_mp3(src, dst)
        return dst
    # Unknown extension; try converting anyway.
    dst = src.with_suffix(".mp3")
    _ffmpeg_to_mp3(src, dst)
    return dst


def _size_ok(path: Path, max_bytes: int = 25 * 1024 * 1024) -> bool:
    try:
        return path.stat().st_size <= max_bytes
    except FileNotFoundError:
        return False


def _chunk_text_for_telegram(text: str, max_len: int = 3900) -> List[str]:
    """
    Telegram message limit is ~4096 chars. Use a conservative chunk size and try to split on newlines.
    """
    s = (text or "").strip()
    if not s:
        return []
    chunks: List[str] = []
    while len(s) > max_len:
        cut = s.rfind("\n", 0, max_len)
        if cut < max_len * 0.5:
            cut = max_len
        chunks.append(s[:cut].rstrip())
        s = s[cut:].lstrip("\n").lstrip()
    if s:
        chunks.append(s)
    return chunks


def _env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed = context.application.bot_data["allowed_usernames"]
    if not _is_authorized(update, allowed):
        await update.effective_message.reply_text("Not authorized.")
        return
    await update.effective_message.reply_text(
        "👋 Send me an audio file/voice note and I'll transcribe it."
    )


async def _send_transcript_txt(update: Update, transcript: str) -> None:
    with tempfile.TemporaryDirectory(prefix="tgtranscribe_") as td:
        filename = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.txt")
        path = Path(td) / filename
        path.write_text((transcript or "").strip() + "\n", encoding="utf-8")
        f = path.open("rb")
        try:
            await update.effective_message.reply_document(document=f, filename=path.name)
        finally:
            f.close()
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def _openai_transcribe(client: OpenAI, audio_path: Path, model: str) -> str:
    with audio_path.open("rb") as f:
        tr = client.audio.transcriptions.create(
            model=model,
            file=f,
        )
    # SDK returns an object with .text
    return getattr(tr, "text", "") or ""


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed = context.application.bot_data["allowed_usernames"]
    if not _is_authorized(update, allowed):
        await update.effective_message.reply_text("Not authorized.")
        return

    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not (msg and chat and user):
        return

    audio = msg.audio or msg.voice or msg.document
    if not audio:
        return

    status_msg = await msg.reply_text("✅ Audio received — 🧠 transcribing now…")
    await msg.chat.send_action(action=ChatAction.TYPING)

    openai_model = context.application.bot_data["openai_model"]
    openai_client: OpenAI = context.application.bot_data["openai_client"]

    with tempfile.TemporaryDirectory(prefix="tgtranscribe_") as td:
        td_path = Path(td)
        tg_file = await context.bot.get_file(audio.file_id)
        src_path = td_path / (getattr(audio, "file_name", None) or f"audio_{audio.file_id}.bin")
        await tg_file.download_to_drive(custom_path=str(src_path))

        try:
            mp3_path = await asyncio.to_thread(_maybe_convert_to_mp3, src_path)
        except Exception as e:
            await msg.reply_text(f"Failed to convert audio: {e}")
            return

        if not _size_ok(mp3_path):
            await msg.reply_text("Audio exceeds 25 MB after conversion; can't transcribe.")
            return

        await msg.chat.send_action(action=ChatAction.TYPING)
        try:
            transcript = await asyncio.to_thread(_openai_transcribe, openai_client, mp3_path, openai_model)
        except Exception as e:
            await msg.reply_text(f"Transcription failed: {e}")
            return

    transcript = (transcript or "").strip()
    if not transcript:
        transcript = "(empty transcription)"

    for part in _chunk_text_for_telegram(transcript):
        await msg.reply_text(part)
    await _send_transcript_txt(update, transcript)
    try:
        await status_msg.delete()
    except Exception:
        pass


def build_app() -> Application:
    token = _env_required("TELEGRAM_BOT_TOKEN")
    allowed = _parse_allowed_usernames(_env_required("ALLOWED_USERNAMES"))
    if not allowed:
        raise RuntimeError("ALLOWED_USERNAMES is empty.")

    openai_key = _env_required("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
    client = OpenAI(api_key=openai_key)

    app = Application.builder().token(token).build()
    app.bot_data["allowed_usernames"] = allowed
    app.bot_data["openai_client"] = client
    app.bot_data["openai_model"] = openai_model

    app.add_handler(CommandHandler("start", cmd_start))
    # Accept audio, voice notes, and audio sent as "documents".
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio))
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
