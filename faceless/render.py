from __future__ import annotations

from pathlib import Path

from moviepy import AudioFileClip, CompositeVideoClip, ImageClip, vfx

from .captions import render_caption_png


def _zoom_clip(image_path, size, duration, zoom_in=True):
    base = ImageClip(str(image_path)).resized(1.15).with_duration(duration)
    if zoom_in:
        base = base.resized(lambda t: 1.0 + 0.10 * (t / duration))
    else:
        base = base.resized(lambda t: 1.10 - 0.10 * (t / duration))
    clip = CompositeVideoClip([base.with_position("center")], size=size)
    return clip.with_duration(duration)


def _caption_clips(segments, size, tmp_dir: Path):
    clips = []
    max_width = int(size[0] * 0.86)
    font_size = max(int(min(size) * 0.055), 28)
    top_ratio = 0.66 if size[1] > size[0] else 0.74
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, (start, end, text) in enumerate(segments["captions"]):
        png = render_caption_png(
            text, tmp_dir / f"cap_{i:03d}.png", max_width=max_width, font_size=font_size
        )
        clip = (
            ImageClip(str(png), transparent=True)
            .with_start(max(start, 0.0))
            .with_duration(max(end - start, 0.2))
            .with_position(("center", int(size[1] * top_ratio)))
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
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = AudioFileClip(str(voice_path))
    total = float(audio.duration)
    sentences = segments["sentences"]
    if not sentences:
        sentences = [(0.0, total, "")]

    beats = []
    n_images = len(image_paths)
    for i, (start, end, _) in enumerate(sentences):
        duration = (end if i < len(sentences) - 1 else total) - max(start, 0.0)
        duration = max(duration, 0.4)
        image_path = image_paths[i % n_images]
        beats.append(_zoom_clip(image_path, size, duration, zoom_in=(i % 2 == 0)))

    layers = beats
    if captions_on and segments["captions"]:
        layers = beats + _caption_clips(segments, size, out_path.parent / "tmp_captions")

    final = CompositeVideoClip(layers, size=size).with_duration(total).with_audio(audio)
    final = final.with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.8)])
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
    return out_path
