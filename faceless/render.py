from __future__ import annotations

import math
import random
import shutil
import wave
from pathlib import Path

import numpy as np
from moviepy import (
    AudioArrayClip,
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    vfx,
)

from .audiofx import build_sound_design
from .captions import chunk_words_by_time, render_caption_png, render_karaoke_variants
from .images import PALETTES


def _scene_clip(image_path, size, start, duration, pan_sign, breath_phase=0.0):
    scale = 1.24
    clip = (
        ImageClip(str(image_path))
        .resized(scale)
        .with_start(max(start, 0.0))
        .with_duration(max(duration, 0.6))
    )
    clip = clip.resized(
        lambda t, ph=breath_phase: 1.0 + 0.014 * math.sin(2 * math.pi * 0.25 * t + ph)
    )
    margin_x = (size[0] * scale - size[0]) / 2
    margin_y = (size[1] * scale - size[1]) / 2
    d = max(clip.duration, 0.01)

    def frame(t, mx=margin_x, my=margin_y, d=d, sign=pan_sign, ph=breath_phase):
        progress = t / d
        x = -mx / 2 + sign * mx * (progress - 0.5)
        y = -my / 2 + my * 0.25 * math.sin(2 * math.pi * 0.5 * progress + ph)
        return x, y

    return clip.with_position(frame)


def _particles(size, total, accent):
    rng = random.Random(3)
    clips = []
    dot_png = None
    from PIL import Image, ImageDraw, ImageFilter

    radius = max(int(min(size) * 0.005), 3)
    layer = Image.new("RGBA", (radius * 2 + 4, radius * 2 + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(
        (2, 2, 2 + radius * 2, 2 + radius * 2),
        fill=(235, 232, 226, 70),
    )
    import tempfile

    tmp = Path(tempfile.gettempdir()) / "faceless_dot.png"
    layer.save(tmp)
    sprite = ImageClip(str(tmp), transparent=True)
    width, height = size
    sprite_w = sprite.size[0]
    for i in range(14):
        phase = rng.uniform(0, 6.28)
        speed = rng.uniform(20, 44)
        x0 = rng.uniform(0.05, 0.95) * width
        amp = rng.uniform(10, 40)
        y0 = rng.uniform(0, 1.0)
        clip = (
            sprite.with_duration(total)
            .with_start(0)
            .with_position(
                lambda t, x0=x0, phase=phase, amp=amp, speed=speed, y0=y0, h=height, sw=sprite_w, w=width: (
                    max(0, min(x0 + amp * math.sin(0.5 + phase + 0.35 * t), w - sw)),
                    (y0 * h - speed * t) % max(h - sw, 1),
                )
            )
            .with_opacity(0.55)
        )
        clips.append(clip)
    big = layer.filter(ImageFilter.GaussianBlur(6))
    big_path = Path(tempfile.gettempdir()) / "faceless_dot_big.png"
    big.save(big_path)
    sprite_big = ImageClip(str(big_path), transparent=True)
    for i in range(6):
        phase = rng.uniform(0, 6.28)
        speed = rng.uniform(10, 22)
        x0 = rng.uniform(0.08, 0.9) * width
        y0 = rng.uniform(0, 1.0)
        clip = (
            sprite_big.with_duration(total)
            .with_start(0)
            .with_position(
                lambda t, x0=x0, phase=phase, speed=speed, y0=y0, h=height, sw=sprite_w, w=width: (
                    max(0, min(x0 + 30 * math.sin(0.3 + phase + 0.25 * t), w - sw)),
                    (y0 * h - speed * t) % max(h - sw, 1),
                )
            )
            .with_opacity(0.4)
        )
        clips.append(clip)
    return clips
    bar_h = 6
    bar = ColorClip(size=(size[0], bar_h), color=accent).with_start(0).with_duration(total)
    bar = bar.resized(lambda t: (max(int(size[0] * t / total), 2), bar_h))
    return bar.with_position((0, 0))


def _progress_bar(total, size, accent):
    bar_h = 6
    bar = ColorClip(size=(size[0], bar_h), color=accent).with_start(0).with_duration(total)
    bar = bar.resized(lambda t: (max(int(size[0] * t / total), 2), bar_h))
    return bar.with_position((0, 0))


def _karaoke_clips(words, size, tmp_dir, accent):
    chunks = chunk_words_by_time(words, 3)
    clips = []
    font_size = max(int(min(size) * 0.052), 26)
    top = int(size[1] * (0.66 if size[1] > size[0] else 0.74))
    max_width = int(size[0] * 0.88)
    for chunk_index, chunk in enumerate(chunks):
        variants = render_karaoke_variants(chunk["words"], tmp_dir, max_width, font_size, accent)
        chunk_words = chunk["words"]
        next_chunk_start = chunks[chunk_index + 1]["start"] if chunk_index + 1 < len(chunks) else None
        for i, (w_start, w_end, _) in enumerate(chunk_words):
            visible_end = w_end
            if i + 1 < len(chunk_words):
                visible_end = max(visible_end, chunk_words[i + 1][0])
            elif next_chunk_start is not None:
                visible_end = max(visible_end, next_chunk_start)
            clip = (
                ImageClip(str(variants[i]), transparent=True)
                .with_start(max(w_start, 0.0))
                .with_duration(max(visible_end - w_start, 0.12))
                .with_position(("center", top))
            )
            clip = clip.resized(lambda t, dur=clip.duration: 1.0 + 0.10 * math.exp(-t / 0.06))
            clip = clip.with_position(
                lambda t, base_top=top: (
                    "center",
                    base_top + 5 * math.sin(2 * math.pi * 0.7 * t),
                )
            )
            clips.append(clip)
    return clips


def _fallback_caption_clips(segments, size, tmp_dir):
    clips = []
    font_size = max(int(min(size) * 0.052), 26)
    top = int(size[1] * (0.66 if size[1] > size[0] else 0.74))
    max_width = int(size[0] * 0.88)
    for i, (start, end, text) in enumerate(segments["captions"]):
        png = render_caption_png(text, tmp_dir / f"cap_{i:03d}.png", max_width, font_size)
        clip = (
            ImageClip(str(png), transparent=True)
            .with_start(max(start, 0.0))
            .with_duration(max(end - start, 0.2))
            .with_position(("center", top))
        )
        clips.append(clip)
    return clips


def _mixed_audio(voice_clip, total, scene_starts, sfx_on, out_dir):
    if not sfx_on:
        return voice_clip
    sfx_path = Path(out_dir) / "sfx.wav"
    build_sound_design(total, scene_starts, sfx_path)
    sfx = AudioFileClip(str(sfx_path))
    fps = 44100
    voice_arr = voice_clip.to_soundarray(fps=fps)
    if voice_arr.ndim == 1:
        voice_arr = np.stack([voice_arr, voice_arr], axis=1)
    sfx_arr = sfx.to_soundarray(fps=fps)
    if sfx_arr.ndim == 1:
        sfx_arr = np.stack([sfx_arr, sfx_arr], axis=1)
    n = max(len(voice_arr), len(sfx_arr))
    voice_arr = np.pad(voice_arr, ((0, n - len(voice_arr)), (0, 0)))
    sfx_arr = np.pad(sfx_arr, ((0, n - len(sfx_arr)), (0, 0)))
    mixed = np.clip(voice_arr + sfx_arr, -1.0, 1.0)
    sfx.close()
    return AudioArrayClip(mixed, fps=fps)


def render_video(
    voice_path: str | Path,
    segments: dict,
    image_paths: list,
    size: tuple,
    out_path: str | Path,
    captions_on: bool = True,
    style: str = "dark_cinematic",
    sfx_on: bool = True,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accent = PALETTES.get(style, PALETTES["dark_cinematic"])["accent"]

    audio = AudioFileClip(str(voice_path))
    total = float(audio.duration)
    sentences = segments["sentences"]
    if not sentences:
        sentences = [(0.0, total, "")]

    n_images = len(image_paths)
    caption_dir = out_path.parent / "tmp_captions"
    shutil.rmtree(caption_dir, ignore_errors=True)
    caption_dir.mkdir(parents=True, exist_ok=True)

    starts = [max(s[0], 0.0) for s in sentences]
    layers = []
    for i, (start, end, _) in enumerate(sentences):
        scene_start = starts[i]
        scene_end = starts[i + 1] if i + 1 < len(sentences) else total
        duration = max(scene_end - scene_start, 0.6)
        image_path = image_paths[i % n_images]
        clip = _scene_clip(
            image_path,
            size,
            scene_start,
            duration,
            pan_sign=1 if i % 2 == 0 else -1,
            breath_phase=0.9 * i,
        )
        if i == 0:
            d0 = max(clip.duration, 0.01)
            clip = clip.resized(lambda t, d0=d0: 1.0 + 0.22 * math.exp(-t / 0.35))
        kind = i % 3
        if i > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(0.35 if kind == 1 else 0.22)])
            if kind == 0:
                d = max(clip.duration, 0.01)
                clip = clip.resized(lambda t, d=d: 1.0 + 0.14 * math.exp(-t / 0.12))
            elif kind == 2:
                margin = (size[0] * 1.16 - size[0]) / 2
                clip = clip.with_position(
                    lambda t, d=duration, sign=1 if i % 2 == 0 else -1, m=margin: (
                        -m / 2
                        + sign * m * (t / d - 0.5)
                        + sign * (1 - min(t / 0.35, 1.0)) * size[0] * 0.25,
                        "center",
                    )
                )
        layers.append(clip)

    layers += _particles(size, total, accent)

    if captions_on:
        words = segments.get("caption_words")
        if words:
            layers += _karaoke_clips(words, size, caption_dir, accent)
        elif segments["captions"]:
            layers += _fallback_caption_clips(segments, size, caption_dir)

    layers.append(_progress_bar(total, size, accent))

    mixed = _mixed_audio(
        audio, total, [s for s in starts], sfx_on, out_path.parent
    )
    final = CompositeVideoClip(layers, size=size).with_duration(total).with_audio(mixed)
    final = final.with_effects([vfx.FadeIn(0.4), vfx.FadeOut(0.7)])
    final.write_videofile(
        str(out_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        threads=2,
        logger=None,
    )
    final.close()
    audio.close()
    if sfx_on and mixed is not audio:
        mixed.close()
    shutil.rmtree(caption_dir, ignore_errors=True)
    return out_path
