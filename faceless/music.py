from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .audiofx import SR, _write_wav
from .config import ASSETS_DIR

MUSIC_DIR = ASSETS_DIR / "music"
MUSIC_GAIN = 0.14
FADE_IN = 1.6
FADE_OUT = 2.8
AUDIO_EXTS = {".mp3", ".ogg", ".wav", ".m4a", ".flac"}


def available_tracks() -> list[Path]:
    if not MUSIC_DIR.exists():
        return []
    return sorted(p for p in MUSIC_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS)


def pick_track(seed_text: str) -> Path | None:
    tracks = available_tracks()
    if not tracks:
        return None
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return tracks[int(digest[:8], 16) % len(tracks)]


def build_music_layer(track: str | Path, total: float, out_path: str | Path) -> Path:
    from moviepy import AudioFileClip

    source = AudioFileClip(str(track))
    arr = source.to_soundarray(fps=SR)
    source.close()
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    arr = arr[:, :2].astype(np.float32)

    peak = np.max(np.abs(arr)) or 1.0
    arr = arr * (0.9 / peak)

    frames = int(total * SR)
    if len(arr) >= frames:
        arr = arr[:frames]
    else:
        loops = int(np.ceil(frames / len(arr)))
        arr = np.tile(arr, (loops, 1))[:frames]

    arr = arr * MUSIC_GAIN

    fade_in = int(min(FADE_IN, total / 3.0) * SR)
    fade_out = int(min(FADE_OUT, total / 3.0) * SR)
    if fade_in > 0:
        arr[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    if fade_out > 0:
        arr[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]

    return _write_wav(Path(out_path), arr)
