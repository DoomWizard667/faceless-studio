from __future__ import annotations

import shutil
from pathlib import Path

from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip, vfx

from .captions import chunk_words_by_time, render_caption_png, render_karaoke_variants
from .images import PALETTES


def _scene_clip(image_path, size, start, duration, pan_sign):
    scale = 1.16
    clip = (
        ImageClip(str(image_path))
        .resized(scale)
        .with_start(max(start, 0.0))
        .with_duration(max(duration, 0.6))
    )
    margin = (size[0] * scale - size[0]) / 2
    d = max(clip.duration, 0.01)
    return clip.with_position(
        lambda t: (
            -margin / 2 + pan_sign * margin * (t / d - 0.5),
            "center",
        )
    )


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
    for chunk in chunks:
        variants = render_karaoke_variants(chunk["words"], tmp_dir, max_width, font_size, accent)
        for i, (w_start, w_end, _) in enumerate(chunk["words"]):
            clip = (
                ImageClip(str(variants[i]), transparent=True)
                .with_start(max(w_start, 0.0))
                .with_duration(max(w_end - w_start, 0.08))
                .with_position(("center", top))
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


def render_video(
    voice_path: str | Path,
    segments: dict,
    image_paths: list,
    size: tuple,
    out_path: str | Path,
    captions_on: bool = True,
    style: str = "dark_cinematic",
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

    layers = []
    starts = [max(s[0], 0.0) for s in sentences]
    for i, (start, end, _) in enumerate(sentences):
        scene_start = starts[i]
        scene_end = starts[i + 1] if i + 1 < len(sentences) else total
        duration = max(scene_end - scene_start, 0.6)
        image_path = image_paths[i % n_images]
        clip = _scene_clip(image_path, size, scene_start, duration, pan_sign=1 if i % 2 == 0 else -1)
        if i > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(0.35)])
        layers.append(clip)

    if captions_on:
        words = segments.get("caption_words")
        if words:
            layers += _karaoke_clips(words, size, caption_dir, accent)
        elif segments["captions"]:
            layers += _fallback_caption_clips(segments, size, caption_dir)

    layers.append(_progress_bar(total, size, accent))

    final = CompositeVideoClip(layers, size=size).with_duration(total).with_audio(audio)
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
    shutil.rmtree(caption_dir, ignore_errors=True)
    return out_path
