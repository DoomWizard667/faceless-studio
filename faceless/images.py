from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import FONT_PATH

PALETTES = {
    "dark_cinematic": [(10, 12, 20), (32, 28, 44), (218, 116, 62)],
    "office_revanche": [(8, 16, 22), (18, 48, 62), (240, 160, 80)],
    "liminal_night": [(6, 8, 12), (26, 34, 40), (170, 200, 220)],
    "ember": [(16, 10, 8), (52, 28, 16), (255, 140, 60)],
}


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default(size)


def _vignette(size: tuple, strength: int = 170) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    margin_x = int(size[0] * 0.18)
    margin_y = int(size[1] * 0.12)
    draw.ellipse(
        (-margin_x, -margin_y, size[0] + margin_x, size[1] + margin_y),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=min(size) // 6))
    black = Image.new("RGBA", size, (0, 0, 0, strength))
    black.putalpha(Image.eval(mask, lambda v: strength * (255 - v) // 255))
    return black


def render_placeholder(
    text: str,
    index: int,
    total: int,
    size: tuple,
    out_path: str | Path,
    style: str = "dark_cinematic",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    palette = PALETTES.get(style, PALETTES["dark_cinematic"])
    width, height = size
    gradient = Image.new("RGBA", size)
    draw = ImageDraw.Draw(gradient)
    top, middle, accent = palette
    for y in range(height):
        ratio = y / height
        if ratio < 0.55:
            k = ratio / 0.55
            color = tuple(int(top[i] + (middle[i] - top[i]) * k) for i in range(3))
        else:
            k = (ratio - 0.55) / 0.45
            color = tuple(int(middle[i] + (top[i] - middle[i]) * k) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color + (255,))

    glow_radius = int(min(size) * 0.25)
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(
        (
            int(width * 0.25),
            int(height * 0.25),
            int(width * 0.25) + 2 * glow_radius,
            int(height * 0.25) + 2 * glow_radius,
        ),
        fill=accent + (54,),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(glow_radius // 2, 1)))
    image = Image.alpha_composite(gradient.convert("RGBA"), glow)
    image = Image.alpha_composite(image, _vignette(size))

    caption = text.strip().strip(".")
    if len(caption) > 90:
        caption = caption[:90].rsplit(" ", 1)[0] + "…"
    label_font = _load_font(max(min(width, height) // 18, 18))
    text_font = _load_font(max(min(width, height) // 11, 24))
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    label = f"{index:02d} / {total:02d}"
    label_width = overlay_draw.textlength(label, font=label_font)
    overlay_draw.text(
        ((width - label_width) / 2, int(height * 0.30)),
        label,
        font=label_font,
        fill=accent + (255,),
    )
    wrapped = textwrap.wrap(caption, width=28)
    text_y = int(height * 0.5)
    for line in wrapped[:5]:
        line_width = overlay_draw.textlength(line, font=text_font)
        overlay_draw.text(
            ((width - line_width) / 2, text_y),
            line,
            font=text_font,
            fill=(235, 232, 226, 235),
        )
        text_y += int(text_font.size * 1.35)
    image = Image.alpha_composite(image, overlay)
    image.convert("RGB").save(out_path, quality=95)
    return out_path


def fit_image(path: str | Path, size: tuple, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = Image.open(path).convert("RGB")
    target_w, target_h = size
    ratio = max(target_w / source.width, target_h / source.height)
    new_size = (int(source.width * ratio) + 1, int(source.height * ratio) + 1)
    source = source.resize(new_size, Image.LANCZOS)
    left = (source.width - target_w) // 2
    top = (source.height - target_h) // 2
    source.crop((left, top, left + target_w, top + target_h)).save(out_path, quality=95)
    return out_path
