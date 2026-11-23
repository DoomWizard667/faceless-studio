from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100


def _write_wav(path: Path, stereo: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(stereo, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())
    return path


def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0, 1, n)
    brown = np.cumsum(white)
    brown -= brown[0]
    t = np.arange(n)
    brown -= brown[0] + (brown[-1] - brown[0]) * t / max(n - 1, 1)
    peak = np.max(np.abs(brown)) or 1.0
    return brown / peak


def _whoosh(duration: float, rng: np.random.Generator) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    noise = _brown_noise(n, rng)
    envelope = np.sin(np.pi * t / duration) ** 1.6
    sweep = np.sin(2 * np.pi * (180 + 320 * t / duration) * t)
    return (noise * 0.75 + sweep * 0.25) * envelope


def _boom(duration: float = 1.0) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    envelope = np.exp(-t * 4.5)
    low = np.sin(2 * np.pi * 46 * t) + 0.5 * np.sin(2 * np.pi * 92 * t)
    rumble = _brown_noise(n, np.random.default_rng(7)) * 0.25
    return (low * 0.8 + rumble) * envelope


def build_sound_design(total: float, scene_starts: list, out_path: Path, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    n = int(total * SR)
    t = np.arange(n) / SR

    amb = _brown_noise(n, rng) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.07 * t + 1.0))
    drone = 0.5 * np.sin(2 * np.pi * 55.0 * t) + 0.3 * np.sin(2 * np.pi * 110.7 * t)
    track = amb * 0.30 + drone * 0.05

    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[:, 0] = track
    stereo[:, 1] = np.roll(track, 90)

    def add(sample: np.ndarray, at: float, gain: float) -> None:
        start = int(at * SR)
        end = min(start + len(sample), n)
        if start < 0 or end <= start:
            return
        piece = sample[: end - start] * gain
        stereo[start:end, 0] += piece
        stereo[start:end, 1] += piece

    whoosh = _whoosh(0.45, rng)
    for scene_start in scene_starts:
        if scene_start > 0.8:
            add(whoosh, scene_start - 0.35, 0.16)

    add(_boom(1.1), 0.05, 0.34)
    if total > 8:
        add(_boom(0.9), total - 1.4, 0.26)

    peak = np.max(np.abs(stereo)) or 1.0
    stereo = stereo * (0.55 / peak)
    return _write_wav(Path(out_path), stereo)
