from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import FONT_PATH


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default(size)


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
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)
    candidate = 10
    for candidate in range(min(26, max(len(uppercase), 12)), 7, -1):
        wrapped = textwrap.wrap(uppercase, candidate)
        if not wrapped:
            break
        widest = max(draw.textlength(line, font=font) for line in wrapped)
        if widest <= max_width - 60:
            break
    wrapped = textwrap.wrap(uppercase, candidate) or [uppercase]
    line_height = int(font_size * 1.22)
    image = Image.new("RGBA", (max_width, len(wrapped) * line_height + 30), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    y = 15
    for line in wrapped:
        width = draw.textlength(line, font=font)
        draw.text(
            ((max_width - width) / 2, y),
            line,
            font=font,
            fill=fill,
            stroke_width=6,
            stroke_fill=stroke,
        )
        y += line_height
    image.save(out_path)
    return out_path
