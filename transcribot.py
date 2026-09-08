import asyncio
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Set

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


_DEFAULT_TRANSCRIBE_MODEL = "gpt-transcribe"
# The Transcriptions API accepts uploads up to 25 MB. Keep the threshold in
# decimal bytes, matching the documented MB unit rather than MiB.
_MAX_TRANSCRIBE_BYTES = 25_000_000
# Oversized files are locally split into AAC/M4A segments at 64 kbps mono.
# 2400 s ~= 19.2 MB of encoded audio, leaving comfortable container overhead.
_CHUNK_AAC_BITRATE_K = 64
_SAFE_CHUNK_SECONDS = 2400

# Formats documented by the current OpenAI Transcriptions API. Other formats
# are normalized to AAC-in-M4A with ffmpeg before upload.
_API_AUDIO_EXTENSIONS = {
    "flac",
    "mp3",
    "mp4",
    "mpeg",
    "mpga",
    "m4a",
    "ogg",
    "wav",
    "webm",
}

# AAC bitrate (kbps), mono, optional sample rate (Hz). We stop at 64 kbps;
# files still above the API size limit are chunked instead of being crushed to
# very low bitrates just to fit in a single request.
_AAC_TIERS: List[tuple[int, bool, Optional[int]]] = [
    (128, False, None),
    (96, False, None),
    (64, True, None),
]


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


def _report_progress(on_progress: Optional[Callable[[str], None]], msg: str) -> None:
    if on_progress is not None:
        on_progress(msg)


def _mb(nbytes: int) -> str:
    return f"{nbytes / 1_000_000:.1f} MB"


def _size_ok(path: Path, max_bytes: int = _MAX_TRANSCRIBE_BYTES) -> bool:
    try:
        return path.stat().st_size <= max_bytes
    except FileNotFoundError:
        return False


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


def _prepare_transcription_file(
    src: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Normalize audio for the OpenAI Transcriptions API.

    Supported files are kept unchanged. Unsupported containers/codecs are
    converted to AAC/M4A. Files larger than the API's 25 MB per-request limit
    are intentionally left larger than the limit; _openai_transcribe() splits
    them into upload-safe chunks instead of degrading them to very low bitrate.
    """
    ext = src.suffix.lower().lstrip(".")
    try:
        src_size = src.stat().st_size
    except OSError:
        src_size = 0

    if ext in _API_AUDIO_EXTENSIONS:
        if src_size <= _MAX_TRANSCRIBE_BYTES:
            _report_progress(on_progress, f"Using original file ({_mb(src_size)})")
        else:
            _report_progress(
                on_progress,
                f"Original file is {_mb(src_size)}; it will be split into upload-safe chunks",
            )
        return src

    out = src.parent / f"{src.stem}_tgapi.m4a"
    for i, (bitrate_k, mono, sample_rate) in enumerate(_AAC_TIERS):
        extras: List[str] = []
        if mono:
            extras.append("mono")
        if sample_rate is not None:
            extras.append(f"{sample_rate // 1000}kHz")
        detail = f" ({', '.join(extras)})" if extras else ""
        _report_progress(on_progress, f"Converting to AAC {bitrate_k} kbps{detail}…")
        _ffmpeg_reencode_m4a_aac(src, out, bitrate_k, mono=mono, sample_rate=sample_rate)
        try:
            sz = out.stat().st_size
        except OSError:
            sz = -1

        if _size_ok(out):
            _report_progress(on_progress, f"→ {_mb(sz)} — OK")
            return out

        if i + 1 < len(_AAC_TIERS):
            _report_progress(
                on_progress,
                f"→ {_mb(sz)} (25 MB/request limit), trying next tier",
            )

    _report_progress(
        on_progress,
        f"→ {_mb(out.stat().st_size)}; will split into upload-safe chunks",
    )
    return out


# Backwards-compatible name (CLI / external scripts).
_maybe_convert_to_mp3 = _prepare_transcription_file


def _user_error_detail(exc: BaseException, max_len: int = 3200) -> str:
    """Short, user-facing error text safe for Telegram message length."""
    s = str(exc).strip() or type(exc).__name__
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _chunk_text_for_telegram(text: str, max_len: int = 3900) -> List[str]:
    """Split text conservatively under Telegram's ~4096-character message limit."""
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
    """Send transcript as one or more Telegram messages."""
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
        body = f"Part {i} of {total}\n\n{part}" if total > 1 else part
        await message.reply_text(body)


def _transcription_chunk_seconds() -> int:
    """
    Duration for local upload-size chunks.

    The current API limit is byte-based (25 MB/request), not the old
    gpt-4o-transcribe ~1400-second cap. Segments are re-encoded at 64 kbps mono,
    so 2400 seconds is about 19.2 MB plus small M4A overhead.

    OPENAI_TRANSCRIBE_CHUNK_SECONDS can request a smaller chunk. The legacy
    OPENAI_TRANSCRIBE_MAX_SECONDS variable is accepted as a fallback for
    backwards compatibility. Values are clamped to the safe 60..2400 range.
    """
    raw = (
        os.getenv("OPENAI_TRANSCRIBE_CHUNK_SECONDS")
        or os.getenv("OPENAI_TRANSCRIBE_MAX_SECONDS")
        or ""
    ).strip()
    if raw.isdigit():
        return min(_SAFE_CHUNK_SECONDS, max(60, int(raw)))
    return _SAFE_CHUNK_SECONDS


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
    return float(out.splitlines()[0].strip())


def _ffmpeg_extract_audio_segment(
    src: Path,
    dst: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    """Extract one upload-safe AAC/M4A audio segment with clean timestamps."""
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
        f"{_CHUNK_AAC_BITRATE_K}k",
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
        kwargs = {
            "model": model,
            "file": (audio_path.name, f),
        }
        # gpt-transcribe supports server-side automatic VAD chunking. This is
        # independent of the 25 MB HTTP upload limit handled locally below.
        if model.lower() == "gpt-transcribe":
            kwargs["chunking_strategy"] = "auto"
        tr = client.audio.transcriptions.create(**kwargs)
    return getattr(tr, "text", "") or ""


def _openai_transcribe(
    client: OpenAI,
    audio_path: Path,
    model: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> str:
    try:
        file_size = audio_path.stat().st_size
    except OSError as e:
        raise RuntimeError(f"cannot stat audio file: {e}") from e

    if file_size <= _MAX_TRANSCRIBE_BYTES:
        _report_progress(
            on_progress,
            f"Transcribing {_mb(file_size)} with {model}…",
        )
        return _openai_transcribe_one(client, audio_path, model)

    try:
        total = _ffprobe_duration_seconds(audio_path)
    except Exception as e:
        raise RuntimeError(
            "Audio exceeds the 25 MB transcription upload limit and its duration "
            f"could not be determined for local chunking: {e}"
        ) from e

    chunk_seconds = _transcription_chunk_seconds()
    n_segments = max(1, math.ceil(total / float(chunk_seconds)))
    _report_progress(
        on_progress,
        f"Input is {_mb(file_size)}; splitting into {n_segments} upload-safe segments "
        f"(up to {chunk_seconds}s each)",
    )

    parts: List[str] = []
    start = 0.0
    idx = 0
    while start < total - 1e-3:
        dur_seg = min(float(chunk_seconds), total - start)
        end = start + dur_seg
        _report_progress(
            on_progress,
            f"Transcribing segment {idx + 1}/{n_segments} "
            f"({start:.0f}–{end:.0f}s)…",
        )
        seg: Optional[Path] = audio_path.parent / f"{audio_path.stem}_seg{idx:04d}.m4a"
        try:
            _ffmpeg_extract_audio_segment(audio_path, seg, start, dur_seg)
            if not seg.stat().st_size:
                raise RuntimeError(f"empty segment file for chunk {idx + 1}")
            if not _size_ok(seg):
                raise RuntimeError(
                    f"Segment {idx + 1} is {_mb(seg.stat().st_size)}, above the 25 MB API limit. "
                    "Set OPENAI_TRANSCRIBE_CHUNK_SECONDS to a smaller value."
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
                src_path = td_path / (
                    getattr(audio, "file_name", None) or f"audio_{audio.file_id}.bin"
                )
                await tg_file.download_to_drive(custom_path=str(src_path))
            except Exception as e:
                detail = _user_error_detail(e)
                lines = [f"Failed to download audio from Telegram: {detail}"]
                low = detail.lower()
                if "too big" in low or "too_large" in low or "file is too" in low:
                    lines.append(
                        "Note: the default Telegram Bot API may reject large bot downloads "
                        "before TGTranscribot can locally split them for OpenAI."
                    )
                await msg.reply_text("\n".join(lines))
                return

            try:
                audio_path = await asyncio.to_thread(
                    _prepare_transcription_file,
                    src_path,
                )
            except Exception as e:
                await msg.reply_text(f"Failed to convert audio: {_user_error_detail(e)}")
                return

            await msg.chat.send_action(action=ChatAction.TYPING)
            try:
                transcript = await asyncio.to_thread(
                    _openai_transcribe,
                    openai_client,
                    audio_path,
                    openai_model,
                )
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
            await msg.reply_text(
                "Transcript was sent in chat, but uploading the .txt file failed: "
                f"{_user_error_detail(e)}"
            )
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
    openai_model = os.getenv("OPENAI_TRANSCRIBE_MODEL", _DEFAULT_TRANSCRIBE_MODEL)
    client = OpenAI(api_key=openai_key)

    app = Application.builder().token(token).build()
    app.bot_data["allowed_usernames"] = allowed
    app.bot_data["openai_client"] = client
    app.bot_data["openai_model"] = openai_model

    app.add_handler(CommandHandler("start", cmd_start))
    # Accept audio, voice notes, and audio sent as "documents".
    app.add_handler(
        MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio)
    )
    app.add_error_handler(on_bot_error)
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
