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
from telegram import Message, Update
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


def _require_ffprobe() -> None:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH. Please install ffmpeg (includes ffprobe).")


_MAX_TRANSCRIBE_BYTES = 25 * 1024 * 1024

# AAC bitrate (kbps), mono, optional sample rate (Hz) — OpenAI accepts m4a; some gpt-4o-transcribe
# deployments reject certain FFmpeg MP3 streams, so we standardize on AAC-in-M4A here.
_AAC_TIERS: List[tuple[int, bool, Optional[int]]] = [
    (128, False, None),
    (96, False, None),
    (64, False, None),
    (48, True, None),
    (32, True, 16000),
]


def _ffmpeg_reencode_m4a_aac(
    src: Path,
    dst: Path,
    bitrate_k: int,
    *,
    mono: bool = False,
    sample_rate: Optional[int] = None,
) -> None:
    _require_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        f"{bitrate_k}k",
        "-movflags",
        "+faststart",
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.append(str(dst))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")


def _prepare_transcription_file(src: Path) -> Path:
    """
    Produce audio under ~25 MB for the transcription API.

    If the file is already small enough and in a common API format, returns it unchanged.
    Otherwise re-encodes to AAC in an .m4a container (bitrate ladder on the original source).
    """
    ext = src.suffix.lower().lstrip(".")
    api_ready = {"mp3", "m4a", "wav", "webm", "mp4", "mpeg", "mpga", "oga", "ogg", "opus", "flac", "aac"}
    if ext in api_ready and _size_ok(src, _MAX_TRANSCRIBE_BYTES):
        return src

    out = src.parent / f"{src.stem}_tgapi.m4a"
    for bitrate_k, mono, sample_rate in _AAC_TIERS:
        _ffmpeg_reencode_m4a_aac(src, out, bitrate_k, mono=mono, sample_rate=sample_rate)
        if _size_ok(out, _MAX_TRANSCRIBE_BYTES):
            return out

    try:
        sz = out.stat().st_size
    except OSError:
        sz = -1
    raise RuntimeError(
        "Audio is still larger than 25 MB after maximum compression (~32 kbps AAC mono, 16 kHz). "
        f"Current size about {sz / (1024 * 1024):.1f} MB. Split into shorter segments or use a lower-quality source."
    )


# Backwards-compatible name (CLI / external scripts).
_maybe_convert_to_mp3 = _prepare_transcription_file


def _size_ok(path: Path, max_bytes: int = 25 * 1024 * 1024) -> bool:
    try:
        return path.stat().st_size <= max_bytes
    except FileNotFoundError:
        return False


def _user_error_detail(exc: BaseException, max_len: int = 3200) -> str:
    """Short, user-facing error text safe for Telegram message length."""
    s = str(exc).strip() or type(exc).__name__
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


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


async def on_bot_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Report uncaught handler errors to the user when a message context exists."""
    exc = context.error
    if exc is None:
        return
    detail = _user_error_detail(exc) if isinstance(exc, BaseException) else str(exc)
    if isinstance(update, Update):
        m = update.effective_message
        if m:
            try:
                await m.reply_text(f"Something went wrong: {detail}")
            except Exception:
                pass


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


async def _reply_transcript_in_chat_chunks(message: Message, transcript: str) -> None:
    """Send transcript as one or more Telegram messages; warn and label parts when splitting."""
    chunks = _chunk_text_for_telegram(transcript)
    if not chunks:
        return
    total = len(chunks)
    if total > 1:
        await message.reply_text(
            "📄 The transcript is long — I'll split it across multiple messages "
            f"(Telegram allows about 4096 characters per message). Total parts: {total}."
        )
    for i, part in enumerate(chunks, start=1):
        if total > 1:
            body = f"Part {i} of {total}\n\n{part}"
        else:
            body = part
        await message.reply_text(body)


def _transcription_max_chunk_seconds(model: str) -> Optional[int]:
    """
    Max seconds per upload for models with a duration cap (e.g. gpt-4o-transcribe ~1400s).
    None = do not split by duration (e.g. whisper-1).
    Override with OPENAI_TRANSCRIBE_MAX_SECONDS (integer, min 60).
    """
    ml = model.lower()
    if "whisper" in ml:
        return None
    if "gpt-4o" in ml and "transcribe" in ml:
        raw = (os.getenv("OPENAI_TRANSCRIBE_MAX_SECONDS") or "").strip()
        if raw.isdigit():
            return max(60, int(raw))
        return 1350
    return None


def _ffprobe_duration_seconds(path: Path) -> float:
    _require_ffprobe()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr[-1000:]}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("ffprobe returned no duration")
    first = out.splitlines()[0].strip()
    return float(first)


def _ffmpeg_extract_audio_segment(src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
    """Extract [start_sec, start_sec+duration) to a fresh AAC/M4A file (clean timestamps for the API)."""
    _require_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extract failed: {proc.stderr[-2000:]}")


def _openai_transcribe_one(client: OpenAI, audio_path: Path, model: str) -> str:
    with audio_path.open("rb") as f:
        tr = client.audio.transcriptions.create(
            model=model,
            file=(audio_path.name, f),
        )
    return getattr(tr, "text", "") or ""


def _openai_transcribe(client: OpenAI, audio_path: Path, model: str) -> str:
    max_chunk = _transcription_max_chunk_seconds(model)
    if max_chunk is None:
        return _openai_transcribe_one(client, audio_path, model)

    try:
        total = _ffprobe_duration_seconds(audio_path)
    except Exception:
        return _openai_transcribe_one(client, audio_path, model)

    if total <= max_chunk:
        return _openai_transcribe_one(client, audio_path, model)

    parts: List[str] = []
    start = 0.0
    idx = 0
    while start < total - 1e-3:
        dur_seg = min(float(max_chunk), total - start)
        seg: Optional[Path] = audio_path.parent / f"{audio_path.stem}_seg{idx:04d}.m4a"
        try:
            _ffmpeg_extract_audio_segment(audio_path, seg, start, dur_seg)
            if not seg.stat().st_size:
                raise RuntimeError(f"empty segment file for chunk {idx + 1}")
            if not _size_ok(seg):
                raise RuntimeError(
                    f"Segment {idx + 1} exceeds 25 MB; shorten OPENAI_TRANSCRIBE_MAX_SECONDS or use lower bitrate."
                )
            parts.append(_openai_transcribe_one(client, seg, model).strip())
        finally:
            if seg is not None:
                try:
                    seg.unlink(missing_ok=True)
                except Exception:
                    pass
        start += dur_seg
        idx += 1

    return "\n\n".join(p for p in parts if p)


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
        await msg.reply_text("I did not receive a usable audio attachment. Send an audio file or voice note.")
        return

    openai_model = context.application.bot_data["openai_model"]
    openai_client: OpenAI = context.application.bot_data["openai_client"]

    status_msg = None
    try:
        status_msg = await msg.reply_text("✅ Audio received — 🧠 transcribing now…")
        await msg.chat.send_action(action=ChatAction.TYPING)

        with tempfile.TemporaryDirectory(prefix="tgtranscribe_") as td:
            td_path = Path(td)
            try:
                tg_file = await context.bot.get_file(audio.file_id)
                src_path = td_path / (getattr(audio, "file_name", None) or f"audio_{audio.file_id}.bin")
                await tg_file.download_to_drive(custom_path=str(src_path))
            except Exception as e:
                detail = _user_error_detail(e)
                lines = [f"Failed to download audio from Telegram: {detail}"]
                low = detail.lower()
                if "too big" in low or "too_large" in low or "file is too" in low:
                    lines.append(
                        "Note: On the default Telegram Bot API, bots can only download files up to "
                        "about 20 MB. Our transcription limit after conversion is 25 MB. "
                        "Very large files fail at download before that check."
                    )
                await msg.reply_text("\n".join(lines))
                return

            try:
                audio_path = await asyncio.to_thread(_prepare_transcription_file, src_path)
            except Exception as e:
                await msg.reply_text(f"Failed to convert audio: {_user_error_detail(e)}")
                return

            if not _size_ok(audio_path):
                await msg.reply_text(
                    "The converted audio is still larger than 25 MB, so it cannot be transcribed. "
                    "Try a shorter recording or a more compressed format."
                )
                return

            await msg.chat.send_action(action=ChatAction.TYPING)
            try:
                transcript = await asyncio.to_thread(_openai_transcribe, openai_client, audio_path, openai_model)
            except Exception as e:
                await msg.reply_text(f"Transcription failed: {_user_error_detail(e)}")
                return

        transcript = (transcript or "").strip()
        if not transcript:
            transcript = "(empty transcription)"

        await _reply_transcript_in_chat_chunks(msg, transcript)
        try:
            await _send_transcript_txt(update, transcript)
        except Exception as e:
            await msg.reply_text(f"Transcript was sent in chat, but uploading the .txt file failed: {_user_error_detail(e)}")
    except Exception as e:
        await msg.reply_text(f"Something went wrong: {_user_error_detail(e)}")
    finally:
        if status_msg is not None:
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
    app.add_error_handler(on_bot_error)
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
