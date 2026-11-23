from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .config import FONT_PATH

PALETTES = {
    "dark_cinematic": {
        "stops": [(10, 6, 26), (46, 18, 50), (30, 30, 66), (100, 44, 44)],
        "accent": (255, 150, 70),
        "glow": (255, 120, 60),
        "wash": (130, 70, 220),
    },
    "office_revanche": {
        "stops": [(6, 16, 34), (16, 66, 88), (34, 84, 104), (235, 135, 60)],
        "accent": (255, 175, 70),
        "glow": (90, 210, 255),
        "wash": (40, 130, 190),
    },
    "liminal_night": {
        "stops": [(8, 8, 30), (24, 36, 84), (44, 76, 122), (140, 185, 225)],
        "accent": (120, 225, 255),
        "glow": (195, 130, 255),
        "wash": (150, 95, 235),
    },
    "ember": {
        "stops": [(24, 8, 6), (80, 26, 12), (150, 46, 18), (255, 155, 65)],
        "accent": (255, 175, 75),
        "glow": (255, 120, 40),
        "wash": (255, 64, 64),
    },
}

MOTIF_KEYWORDS = [
    (r"(porte|clé|verrou|garage|hall d'entrée)", "door"),
    (r"(ascenseur|sous-sol\b|escalier|étage)", "stairs"),
    (r"(fenêtre|vitres|rideau|lumière|lampe)", "window"),
    (r"(maison|grand-mère|héritage|notaire|cave\b)", "house"),
    (r"(téléphone|sms|messages|mail|bâbysitter|baby-sitter|smartphone)", "phone"),
    (r"(couloir|immeuble|bureau|restaurant|chambre|hôtel)", "corridor"),
    (r"(ville|rue\b|voiture|route|chauffeur)", "city"),
    (r"(yeux|regard|visage|signature|écrivait|mot\b)", "eyes"),
    (r"(patron|voisin|ami|femme|homme|fille|garçon|personne|employer|employé|équipe)", "figures"),
    (r"(je\b|j'|ma\b|mon\b|mes\b)", "figure"),
]

MOTIF_CYCLE = ["door", "figure", "corridor", "window", "stairs", "eyes", "phone", "city"]


def pick_motif(text: str, index: int, rng) -> str:
    lowered = text.lower()
    for pattern, motif in MOTIF_KEYWORDS:
        if re.search(pattern, lowered):
            return motif
    return MOTIF_CYCLE[index % len(MOTIF_CYCLE)]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default(size)


def _gradient(size: tuple, stops: list) -> Image.Image:
    width, height = size
    ys = np.linspace(0.0, 1.0, height)
    ctrl = np.linspace(0.0, 1.0, len(stops))
    array = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        column = np.interp(
            ys,
            ctrl,
            np.array([s[c] for s in stops], dtype=np.float32),
        )
        array[:, :, c] = column[:, None]
    return Image.fromarray(array.astype(np.uint8), "RGB")


def _grain(image: Image.Image, amount: float = 9.0) -> Image.Image:
    array = np.asarray(image).astype(np.int16)
    noise = np.random.normal(0, amount, array.shape[:2])
    array[..., :3] += noise[..., None].astype(np.int16)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), image.mode)


def _vignette(image: Image.Image, strength: int = 165) -> Image.Image:
    size = image.size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    margin_x = int(size[0] * 0.25)
    margin_y = int(size[1] * 0.18)
    draw.ellipse(
        (-margin_x, -margin_y, size[0] + margin_x, size[1] + margin_y),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=min(size) // 5))
    dark = Image.new("RGBA", size, (2, 2, 4, strength))
    alpha = Image.eval(mask, lambda v: strength * (255 - v) // 255)
    dark.putalpha(alpha)
    return Image.alpha_composite(image.convert("RGBA"), dark)


def _glow(image: Image.Image, color: tuple, center: tuple, radius: int, strength: int = 90) -> Image.Image:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        fill=color + (strength,),
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=radius // 2))
    return Image.alpha_composite(image, layer)


def _bokeh(image: Image.Image, colors: list, rng, count: int = 18) -> Image.Image:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    radius_ref = min(width, height)
    for _ in range(count):
        radius = rng.randint(radius_ref // 90, radius_ref // 22)
        x = rng.randint(0, width)
        y = rng.randint(0, int(height * 0.75))
        alpha = rng.randint(26, 80)
        color = colors[rng.randrange(len(colors))]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color + (alpha,),
        )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=radius_ref // 120))
    return Image.alpha_composite(image, layer)


def _light_shaft(image: Image.Image, accent: tuple, rng, strength: int = 46) -> Image.Image:
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    top_x = rng.randint(int(width * 0.25), int(width * 0.75))
    lean = rng.randint(-int(width * 0.18), int(width * 0.18))
    base = int(width * 0.05)
    draw.polygon(
        [
            (top_x - base, -10),
            (top_x + base, -10),
            (top_x + lean + base * 3, height),
            (top_x + lean - base * 3, height),
        ],
        fill=accent + (strength,),
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=int(width * 0.03)))
    return Image.alpha_composite(image, layer)


def _silhouette(draw, cx: int, floor: int, height: int, width: int, color, rim_color=None):
    head_r = int(width * 0.05)
    head_y = floor - int(height * 0.30)
    body_top = head_y + head_r
    draw.ellipse(
        (cx - int(width * 0.075), floor - int(height * 0.012), cx + int(width * 0.075), floor + int(height * 0.014)),
        fill=(8, 8, 12, 150),
    )
    draw.ellipse(
        (cx - head_r, head_y - head_r, cx + head_r, head_y + head_r),
        fill=color,
        outline=rim_color if rim_color else None,
        width=max(int(width * 0.004), 3) if rim_color else 0,
    )
    draw.polygon(
        [
            (cx - head_r * 2.1, floor),
            (cx + head_r * 2.1, floor),
            (cx + head_r * 0.9, body_top),
            (cx - head_r * 0.9, body_top),
        ],
        fill=color,
    )
    if rim_color:
        tilt = 1 if cx < width // 2 else -1
        draw.line(
            [
                (cx + tilt * head_r * 1.9, floor),
                (cx + tilt * head_r * 0.75, body_top + head_r * 0.2),
            ],
            fill=rim_color,
            width=max(int(width * 0.0035), 3),
        )
    return head_y


def _motif_layer(image: Image.Image, motif: str, palette: dict, rng):
    accent = palette["accent"]
    glow = palette.get("glow", accent)
    cool = palette.get("wash", (120, 180, 255))
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    dark = (4, 4, 8, 235)
    horizon = int(height * 0.25)
    floor = int(height * 0.93)
    outline_light = (190, 195, 205, 110)

    if motif == "door":
        door_w = int(width * 0.30)
        door_h = int(height * 0.25)
        x0 = (width - door_w) // 2
        y0 = floor - door_h
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.rectangle(
            (int(x0 * 0.8), floor - int(height * 0.25), int((x0 + door_w) * 1.2), floor),
            fill=accent + (54,),
        )
        glow = glow.filter(ImageFilter.GaussianBlur(radius=int(width * 0.03)))
        layer = Image.alpha_composite(glow, layer)
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle(
            (x0, y0, x0 + door_w, floor),
            radius=int(door_w * 0.06),
            fill=(14, 15, 22, 252),
            outline=outline_light,
            width=max(int(width * 0.0022), 2),
        )
        gap_h = int(height * 0.005)
        draw.rectangle((x0 + 4, floor - gap_h, x0 + door_w - 4, floor + 2), fill=accent + (200,))
        mid_h = int(height * 0.004)
        draw.rectangle((x0 + 4, y0 + int(door_h * 0.55), x0 + door_w - 4, y0 + int(door_h * 0.55) + mid_h), fill=accent + (60,))

    elif motif == "corridor":
        cx = width // 2
        frames = 5
        for i in range(frames, 0, -1):
            scale = i / frames
            shrink = scale * 0.5
            fw = int(width * shrink)
            fh = int((floor - horizon) * shrink)
            x0 = cx - fw // 2
            y1 = int(floor - (floor - horizon) * (1 - (1 - scale) * 0.25))
            draw.rectangle(
                (x0, y1 - fh, x0 + fw, y1),
                outline=(160, 160, 175, 160),
                width=max(int(width * 0.0022), 2),
            )
            lamp_c = accent if i % 2 == 0 else cool
            lamp_r = max(int(width * 0.006), 3)
            lamp_y = y1 - fh + int(fh * 0.25)
            draw.ellipse(
                (cx - lamp_r, lamp_y - lamp_r, cx + lamp_r, lamp_y + lamp_r),
                fill=lamp_c + (200,),
            )
            if i == 1:
                draw.rectangle((x0 + 6, y1 - fh + 6, x0 + fw - 6, y1), fill=accent + (34,))
        depth = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ddraw = ImageDraw.Draw(depth)
        ddraw.ellipse(
            (cx - width * 0.05, floor - height * 0.05, cx + width * 0.05, floor),
            fill=accent + (54,),
        )
        depth = depth.filter(ImageFilter.GaussianBlur(radius=int(width * 0.03)))
        layer = Image.alpha_composite(depth, layer)

    elif motif == "figure":
        cx = int(width * rng.uniform(0.38, 0.62))
        _silhouette(draw, cx, floor, height, width, dark, rim_color=accent + (150,))

    elif motif == "figures":
        cx1 = int(width * 0.30)
        cx2 = int(width * 0.70)
        _silhouette(draw, cx1, floor, height, width, dark, rim_color=accent + (150,))
        _silhouette(draw, cx2, floor, height, width, (8, 8, 14, 240))

    elif motif == "eyes":
        eye_y = int(height * 0.45)
        spread = int(width * 0.09)
        cx = width // 2 + rng.randint(-int(width * 0.1), int(width * 0.1))
        for dx in (-spread, spread):
            draw.ellipse(
                (cx + dx - int(width * 0.05), eye_y - int(height * 0.012), cx + dx + int(width * 0.05), eye_y + int(height * 0.012)),
                fill=accent + (235,),
            )
            draw.ellipse(
                (cx + dx - int(width * 0.012), eye_y - int(height * 0.008), cx + dx + int(width * 0.012), eye_y + int(height * 0.008)),
                fill=(2, 2, 4, 255),
            )
        haze = Image.new("RGBA", image.size, (0, 0, 0, 0))
        hdraw = ImageDraw.Draw(haze)
        hdraw.ellipse(
            (cx - int(width * 0.22), eye_y - int(height * 0.25), cx + int(width * 0.22), eye_y + int(height * 0.25)),
            fill=accent + (26,),
        )
        haze = haze.filter(ImageFilter.GaussianBlur(radius=int(width * 0.05)))
        layer = Image.alpha_composite(haze, layer)
        draw = ImageDraw.Draw(layer)

    elif motif == "window":
        win_w = int(width * 0.34)
        win_h = int(height * 0.30)
        x0 = (width - win_w) // 2
        y0 = horizon
        draw.rectangle((x0, y0, x0 + win_w, y0 + win_h), fill=glow + (86,))
        bar = max(3, int(width * 0.006))
        draw.rectangle((x0, y0 + win_h // 2 - bar, x0 + win_w, y0 + win_h // 2 + bar), fill=dark)
        draw.rectangle((x0 + win_w // 2 - bar, y0, x0 + win_w // 2 + bar, y0 + win_h), fill=dark)
        draw.rectangle((x0, y0, x0 + win_w, y0 + win_h), outline=dark, width=bar * 2)
        figure_cx = x0 + int(win_w * 0.30)
        _silhouette(draw, figure_cx, y0 + win_h - 2, int(win_h * 2.4), int(width * 0.8), (6, 6, 10, 245))

    elif motif == "phone":
        phone_w = int(width * 0.30)
        phone_h = int(phone_w * 1.9)
        x0 = (width - phone_w) // 2
        y0 = floor - phone_h - int(height * 0.02)
        screen_glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(screen_glow)
        sdraw.rounded_rectangle(
            (x0 - int(width * 0.03), y0 - int(height * 0.02), x0 + phone_w + int(width * 0.03), y0 + phone_h + int(height * 0.25)),
            radius=int(width * 0.05),
            fill=accent + (54,),
        )
        screen_glow = screen_glow.filter(ImageFilter.GaussianBlur(radius=int(width * 0.03)))
        layer = Image.alpha_composite(screen_glow, layer)
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle(
            (x0, y0, x0 + phone_w, y0 + phone_h),
            radius=int(width * 0.035),
            fill=(10, 12, 18, 252),
            outline=outline_light,
            width=max(int(width * 0.0022), 2),
        )
        inner = int(width * 0.015)
        draw.rounded_rectangle(
            (x0 + inner, y0 + inner, x0 + phone_w - inner, y0 + phone_h - inner),
            radius=int(width * 0.025),
            fill=(225, 228, 235, 235),
        )
        line_h = int(phone_h * 0.055)
        palette_line = (30, 32, 40, 160)
        for li in range(1, 7):
            if rng.random() < 0.75:
                lw = int(phone_w * rng.uniform(0.35, 0.75))
                lx = x0 + inner + int(width * 0.012)
                ly = y0 + inner + int(line_h * li * 1.5)
                draw.rounded_rectangle(
                    (lx, ly, lx + lw, ly + line_h),
                    radius=int(line_h / 2),
                    fill=accent + (170,) if li % 3 == 0 else palette_line,
                )
        hand_dark = (5, 5, 9, 250)
        draw.polygon(
            [
                (x0 - int(width * 0.05), height),
                (x0 + phone_w + int(width * 0.05), height),
                (x0 + phone_w + int(width * 0.02), y0 + phone_h - int(height * 0.05)),
                (x0 - int(width * 0.02), y0 + phone_h - int(height * 0.05)),
            ],
            fill=hand_dark,
        )

    elif motif == "house":
        house_w = int(width * 0.25)
        house_h = int(height * 0.25)
        x0 = (width - house_w) // 2
        y0 = floor - house_h
        draw.polygon(
            [
                (x0 - int(house_w * 0.1), y0),
                (x0 + house_w + int(house_w * 0.1), y0),
                (x0 + house_w // 2, y0 - int(house_h * 0.45)),
            ],
            fill=dark,
        )
        draw.rectangle((x0, y0, x0 + house_w, floor), fill=dark)
        win_w2 = int(house_w * 0.18)
        wx = x0 + house_w // 2 - win_w2 // 2
        wy = y0 + int(house_h * 0.25)
        draw.rectangle((wx, wy, wx + win_w2, wy + int(win_w2 * 1.4)), fill=accent + (170,))
        figure_cx = x0 - int(width * 0.12)
        if figure_cx > int(width * 0.05):
            _silhouette(draw, figure_cx, floor, height, width, dark, rim_color=accent + (120,))
        ground = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(ground)
        gdraw.rectangle((0, floor, width, height), fill=(2, 2, 4, 245))
        layer = Image.alpha_composite(ground, layer)

    elif motif == "stairs":
        steps = 6
        step_h = int((floor - horizon) / steps)
        for i in range(steps):
            y1 = floor - i * step_h
            inset = int(width * 0.25 * (i / steps) * 0.6)
            draw.rectangle(
                (inset, y1 - step_h, width - inset, y1),
                outline=(150, 152, 160, 150),
                width=max(int(width * 0.0035), 2),
            )
        top_glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(top_glow)
        tdraw.ellipse(
            (width // 2 - width * 0.25, horizon - width * 0.1, width // 2 + width * 0.25, horizon + width * 0.25),
            fill=accent + (50,),
        )
        top_glow = top_glow.filter(ImageFilter.GaussianBlur(radius=int(width * 0.04)))
        layer = Image.alpha_composite(top_glow, layer)

    elif motif == "city":
        rng_sk = rng
        x = -int(width * 0.05)
        while x < width:
            bw = rng_sk.randint(int(width * 0.05), int(width * 0.13))
            bh = rng_sk.randint(int(height * 0.25), int(height * 0.55))
            draw.rectangle((x, floor - bh, x + bw, floor), fill=dark)
            for wy in range(floor - bh + int(height * 0.03), floor - int(height * 0.03), int(height * 0.045)):
                for wx in range(x + int(width * 0.012), x + bw - int(width * 0.012), int(width * 0.03)):
                    if rng_sk.random() < 0.55:
                        lw = max(2, int(width * 0.004))
                        light_c = glow if rng_sk.random() < 0.55 else (60, 200, 255) if rng_sk.random() < 0.5 else accent
                        draw.rectangle(
                            (wx, wy, wx + lw, wy + int(height * 0.012)),
                            fill=light_c + (rng_sk.randint(120, 230),),
                        )
            x += bw + int(width * 0.012)
    return layer


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
    rng = __import__("random").Random(f"{index}-{style}")
    palette = PALETTES.get(style, PALETTES["dark_cinematic"])
    motif = pick_motif(text, index, rng)

    image = _gradient(size, palette["stops"]).convert("RGBA")
    gx = int(size[0] * rng.uniform(0.3, 0.7))
    gy = int(size[1] * rng.uniform(0.28, 0.5))
    image = _glow(
        image,
        palette["glow"],
        (gx, gy),
        int(min(size) * 0.30),
        strength=140,
    )
    image = _glow(
        image,
        palette.get("wash", palette["glow"]),
        (int(size[0] * 0.25 + gx * 0.5) % size[0], int(size[1] * 0.75)),
        int(min(size) * 0.25),
        strength=90,
    )
    if motif != "city":
        image = _light_shaft(image, palette["glow"], rng, strength=58)
    motif_layer = _motif_layer(image, motif, palette, rng)
    image = Image.alpha_composite(image, motif_layer)
    image = ImageEnhance.Color(image).enhance(1.25)
    image = _bokeh(image, [palette["accent"], palette.get("glow", palette["accent"]), palette.get("wash", (140, 180, 255))], rng)
    image = _grain(image)
    image = _vignette(image)
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
