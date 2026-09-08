#!/usr/bin/env python3
"""
Command-line transcription: print transcript to stdout and save to a text file.

Requires OPENAI_API_KEY. Optional OPENAI_TRANSCRIBE_MODEL (default: gpt-transcribe).
Requires ffmpeg/ffprobe on PATH (same as the Telegram bot).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

from transcribot import (
    _DEFAULT_TRANSCRIBE_MODEL,
    _openai_transcribe,
    _prepare_transcription_file,
)


def _default_out_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_transcript.txt")


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Transcribe an audio file with OpenAI and print + save the transcript."
    )
    p.add_argument(
        "audio",
        type=Path,
        help="Path to the audio file (voice/video formats supported via ffmpeg conversion).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .txt path (default: <input_stem>_transcript.txt next to the input file).",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "OpenAI transcription model "
            f"(default: env OPENAI_TRANSCRIBE_MODEL or {_DEFAULT_TRANSCRIBE_MODEL})."
        ),
    )
    args = p.parse_args(argv)

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        print(f"error: not a file: {audio_path}", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    model = args.model or os.getenv("OPENAI_TRANSCRIBE_MODEL", _DEFAULT_TRANSCRIBE_MODEL)
    out_path = args.output.expanduser().resolve() if args.output else _default_out_path(audio_path)

    client = OpenAI(api_key=api_key)

    _progress(f"Preparing {audio_path} (model={model})")
    try:
        with tempfile.TemporaryDirectory(prefix="tgtranscribe_cli_") as td:
            td_path = Path(td)
            work_src = td_path / audio_path.name
            shutil.copy2(audio_path, work_src)
            audio_path = _prepare_transcription_file(work_src, on_progress=_progress)
            transcript = (
                _openai_transcribe(client, audio_path, model, on_progress=_progress) or ""
            ).strip()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not transcript:
        transcript = "(empty transcription)"

    print(transcript)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
