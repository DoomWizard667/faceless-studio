from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import FONT_PATH, FONT_PATH_FALLBACK, WORD_GROUP_SIZE


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    if FONT_PATH_FALLBACK.exists():
        return ImageFont.truetype(str(FONT_PATH_FALLBACK), size)
    return ImageFont.load_default(size)


def _layout_lines(words: list, font, max_width) -> list:
    lines = []
    current = []
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)
    space = draw.textlength(" ", font=font)
    current_w = 0.0
    for word in words:
        w = draw.textlength(word, font=font)
        candidate = current_w + (space if current else 0) + w
        if current and candidate > max_width:
            lines.append(current)
            current = [word]
            current_w = w
        else:
            current.append(word)
            current_w = candidate
    if current:
        lines.append(current)
    return lines


def _draw_words(
    lines: list,
    highlight: int,
    font,
    max_width: int,
    line_height: int,
    fill=(255, 255, 255, 255),
    accent=(255, 190, 90, 255),
    stroke=(0, 0, 0, 255),
    stroke_width=6,
    box=False,
    shadow=False,
) -> Image.Image:
    flat = [w for line in lines for w in line]
    active = {flat[highlight]} if 0 <= highlight < len(flat) else set()
    height = len(lines) * line_height + 40
    image = Image.new("RGBA", (max_width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    y = 20
    ink = (14, 12, 16, 255)
    pad_x = 10
    pad_y = 6
    for line in lines:
        total = sum(draw.textlength(w, font=font) for w in line)
        total += draw.textlength(" ", font=font) * (len(line) - 1)
        x = (max_width - total) / 2
        for word in line:
            tw = draw.textlength(word, font=font)
            ascent, descent = font.getmetrics()
            th = ascent + descent
            is_active = word in active
            if shadow:
                draw.text((x + 3, y + 6), word, font=font, fill=(0, 0, 0, 120),
                          stroke_width=stroke_width, stroke_fill=(0, 0, 0, 120))
            if box and is_active:
                draw.rounded_rectangle(
                    (x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y),
                    radius=12,
                    fill=accent,
                )
                draw.text((x, y), word, font=font, fill=ink, stroke_width=2, stroke_fill=ink)
            else:
                color = accent if is_active else fill
                draw.text((x, y), word, font=font, fill=color,
                          stroke_width=stroke_width, stroke_fill=stroke)
            x += tw + draw.textlength(" ", font=font)
        y += line_height
    return image


def render_caption_png(
    text: str,
    out_path: str | Path,
    max_width: int,
    font_size: int = 72,
    fill=(255, 255, 255),
    stroke=(0, 0, 0),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = _load_font(font_size)
    uppercase = text.strip().upper()
    wrapped = textwrap.wrap(uppercase, 24) or [uppercase]
    line_height = int(font_size * 1.24)
    image = _draw_words(
        [line.split() for line in wrapped],
        -1,
        font,
        max_width,
        line_height,
        fill=fill + (255,),
        stroke=stroke + (255,),
        stroke_width=max(int(font_size * 0.13), 5),
        shadow=True,
    )
    image.save(out_path)
    return out_path


def render_karaoke_variants(
    words: list,
    out_dir: str | Path,
    max_width: int,
    font_size: int = 72,
    accent=(255, 190, 90),
) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(font_size)
    display = [w[2].strip().upper() for w in words]
    lines = _layout_lines(display, font, max_width)
    line_height = int(font_size * 1.24)
    variants = []
    for i, w in enumerate(words):
        image = _draw_words(
            lines,
            i,
            font,
            max_width,
            line_height,
            fill=(252, 252, 252, 255),
            accent=accent + (255,),
            stroke=(10, 9, 12, 255),
            stroke_width=max(int(font_size * 0.13), 5),
            box=True,
            shadow=True,
        )
        path = out_dir / f"karaoke_{i:03d}.png"
        image.save(path)
        variants.append(path)
    return variants


def chunk_words_by_time(words: list, group_size: int = WORD_GROUP_SIZE) -> list:
    chunks = []
    for i in range(0, len(words), group_size):
        group = words[i : i + group_size]
        chunks.append(
            {
                "start": group[0][0],
                "end": group[-1][1],
                "text": " ".join(w[2] for w in group),
                "words": group,
            }
        )
    return chunks
