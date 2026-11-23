from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from pathlib import Path

from edge_tts import Communicate

from .config import DEFAULT_VOICE, WORD_GROUP_SIZE, env


def _tokens(text: str) -> list:
    return re.findall(r"[\w’'-]+", text)


def _chunk_captions(words: list, group_size: int = WORD_GROUP_SIZE) -> list:
    chunks = []
    for i in range(0, len(words), group_size):
        group = words[i : i + group_size]
        chunks.append((group[0][0], group[-1][1], " ".join(w[2] for w in group)))
    return chunks


def _sentence_slots(text: str) -> list:
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
    return [(s, len(_tokens(s))) for s in sentences]


def _assign_sentences(text: str, words: list) -> list:
    sentence_segments = []
    index = 0
    for sentence, expected in _sentence_slots(text):
        taken = []
        while expected > 0 and index < len(words):
            taken.append(words[index])
            index += 1
            expected -= 1
        if taken:
            sentence_segments.append((taken[0][0], taken[-1][1], sentence))
    if index < len(words):
        remainder = " ".join(w[2] for w in words[index:])
        if sentence_segments:
            prev = sentence_segments[-1]
            sentence_segments[-1] = (prev[0], words[-1][1], prev[2] + " " + remainder)
        else:
            sentence_segments.append((words[index][0], words[-1][1], remainder))
    return sentence_segments


def _build_segments(text: str, words: list) -> dict:
    return {
        "captions": _chunk_captions(words),
        "caption_words": list(words),
        "sentences": _assign_sentences(text, words),
    }


def _estimate_words(audio_duration: float, text: str) -> list:
    total_chars = max(len(text), 1)
    words = []
    for match in re.finditer(r"\S+", text):
        start = audio_duration * match.start() / total_chars
        end = audio_duration * match.end() / total_chars
        words.append((start, max(end, start + 0.08), match.group(0).strip(".,!?…;:")))
    return words


def _estimate_segments(audio_path: Path, text: str) -> dict:
    from moviepy import AudioFileClip

    with AudioFileClip(str(audio_path)) as clip:
        duration = float(clip.duration)
    words = _estimate_words(duration, text)
    return {
        "captions": _chunk_captions(words),
        "caption_words": words,
        "sentences": _assign_sentences(text, words),
    }


async def _edge_speak(text: str, voice: str, out_path: Path) -> list:
    words = []
    communicate = Communicate(text, voice, boundary="WordBoundary")
    with open(out_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                end = (chunk["offset"] + chunk["duration"]) / 1e7
                cleaned = chunk["text"].strip()
                if cleaned:
                    words.append((start, end, cleaned))
    return words


def _elevenlabs_speak(text: str, voice: str, out_path: Path) -> list:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    payload = json.dumps(
        {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.75},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"xi-api-key": env("ELEVENLABS_API_KEY"), "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        out_path.write_bytes(response.read())
    return []


def synthesize(text: str, out_path: str | Path, voice: str | None = None) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = env("ELEVENLABS_API_KEY")
    if api_key:
        words = _elevenlabs_speak(text, voice or env("ELEVENLABS_VOICE", "pFZP5JQG7iQjIQuC4Bku"), out_path)
    else:
        words = asyncio.run(_edge_speak(text, voice or DEFAULT_VOICE, out_path))
    if words:
        return _build_segments(text, words)
    return _estimate_segments(out_path, text)
