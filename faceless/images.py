from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

PALETTES = {
    "dark_cinematic": {
        "stops": [(8, 6, 26), (36, 16, 50), (26, 26, 64), (92, 46, 54), (214, 112, 74)],
        "accent": (255, 150, 70),
        "glow": (255, 120, 60),
        "wash": (140, 70, 230),
        "fog": (70, 44, 104),
        "shadow": (6, 6, 12),
    },
    "office_revanche": {
        "stops": [(6, 16, 34), (16, 62, 84), (30, 78, 100), (60, 96, 110), (235, 135, 62)],
        "accent": (255, 175, 70),
        "glow": (90, 210, 255),
        "wash": (40, 130, 195),
        "fog": (46, 96, 130),
        "shadow": (5, 10, 18),
    },
    "liminal_night": {
        "stops": [(8, 8, 32), (22, 34, 86), (40, 72, 122), (80, 112, 150), (150, 190, 228)],
        "accent": (120, 225, 255),
        "glow": (195, 130, 255),
        "wash": (150, 95, 235),
        "fog": (80, 92, 150),
        "shadow": (6, 7, 14),
    },
    "night_pov": {
        "stops": [(4, 6, 14), (10, 18, 38), (18, 32, 58), (34, 52, 84), (96, 122, 154)],
        "accent": (238, 72, 80),
        "glow": (120, 182, 255),
        "wash": (58, 88, 158),
        "fog": (38, 56, 94),
        "shadow": (3, 4, 9),
        "grain": 14,
    },
    "ember": {
        "stops": [(26, 8, 6), (74, 24, 12), (126, 42, 18), (172, 76, 28), (255, 158, 70)],
        "accent": (255, 175, 75),
        "glow": (255, 120, 40),
        "wash": (255, 64, 64),
        "fog": (120, 48, 28),
        "shadow": (8, 4, 3),
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


def _mix(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def _gradient(size: tuple, stops: list) -> Image.Image:
    width, height = size
    ys = np.linspace(0.0, 1.0, height)
    ctrl = np.linspace(0.0, 1.0, len(stops))
    t = ys * (ys * 0.18 + 0.82)
    array = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        column = np.interp(t, ctrl, np.array([s[c] for s in stops], dtype=np.float32))
        array[:, :, c] = column[:, None]
    xs = np.linspace(0.0, 1.0, width)
    array += (np.abs(xs - 0.5) * 2)[None, :, None] * np.array([8, 4, 16], dtype=np.float32)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def _glow(
    image: Image.Image,
    color: tuple,
    center: tuple,
    radius: int,
    strength: int = 100,
    squash: float = 1.0,
) -> Image.Image:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ry = max(int(radius * squash), 2)
    draw.ellipse(
        (center[0] - radius, center[1] - ry, center[0] + radius, center[1] + ry),
        fill=color + (strength,),
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(radius // 2, 4)))
    return Image.alpha_composite(image, layer)


def _fog(size: tuple, color: tuple, rng, strength: int = 60, center: float = 0.8, spread: float = 0.22) -> Image.Image:
    width, height = size
    ys = np.linspace(0.0, 1.0, height)
    band = np.exp(-((ys - center) ** 2) / (2 * max(spread, 0.01) ** 2))
    noise = np.random.rand(height // 8 + 1, width // 8 + 1)
    noise_img = Image.fromarray((noise * 255).astype("uint8"), "L").resize((width, height), Image.BICUBIC)
    n = np.asarray(noise_img, dtype=np.float32) / 255.0
    alpha = band[:, None] * (0.5 + 0.5 * n)
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[..., 0] = color[0]
    layer[..., 1] = color[1]
    layer[..., 2] = color[2]
    layer[..., 3] = np.clip(alpha * (strength / 255.0) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def _stars(size: tuple, rng, count: int = 90) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = size
    for _ in range(count):
        x = rng.randint(0, max(width - 2, 1))
        y = rng.randint(0, max(int(height * 0.45) - 2, 1))
        r = rng.choice([1, 1, 1, 2])
        a = rng.randint(90, 210)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(235, 240, 255, a))
    return layer.filter(ImageFilter.GaussianBlur(1))


def _moon(size: tuple, x: float, y: float, color=(240, 240, 220)) -> Image.Image:
    width, height = size
    r = max(int(width * 0.05), 8)
    cx, cy = int(width * x), int(height * y)
    halo = Image.new("RGBA", size, (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(halo)
    hdraw.ellipse((cx - r * 3, cy - r * 3, cx + r * 3, cy + r * 3), fill=color + (70,))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=max(r * 2, 4)))
    disc = Image.new("RGBA", size, (0, 0, 0, 0))
    ddraw = ImageDraw.Draw(disc)
    ddraw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color + (240,))
    ddraw.ellipse((cx - int(r * 0.25), cy - int(r * 0.3), cx + int(r * 0.1), cy + int(r * 0.25)), fill=color + (150,))
    return Image.alpha_composite(halo, disc)


def _light_pool(layer: Image.Image, color: tuple, cx: int, y: int, rx: int, ry: int, strength: int = 70) -> Image.Image:
    pool = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(pool)
    draw.ellipse((cx - rx, y - ry, cx + rx, y + ry), fill=color + (strength,))
    pool = pool.filter(ImageFilter.GaussianBlur(radius=max(rx // 2, 4)))
    return Image.alpha_composite(layer, pool)


def _floor_scene(layer: Image.Image, color: tuple, sheen: tuple, horizon: int, floor: int, vp_x: int, lines: bool = True) -> Image.Image:
    width, height = layer.size
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.polygon([(0, floor), (width, floor), (vp_x + width // 2, horizon), (vp_x - width // 2, horizon)], fill=color + (235,))
    grad = np.zeros((height, width, 4), dtype=np.uint8)
    ramp = np.clip(np.linspace(0.0, 1.0, max(floor - horizon, 2)), 0, 1)
    for i, y in enumerate(range(horizon, floor)):
        a = int(ramp[min(i, len(ramp) - 1)] * 110)
        grad[y, :, :] = (sheen[0], sheen[1], sheen[2], a)
    out = Image.alpha_composite(out, Image.fromarray(grad, "RGBA"))
    draw = ImageDraw.Draw(out)
    if lines:
        for dx in (-0.34, -0.17, 0.0, 0.17, 0.34, 0.6, -0.6):
            x_bottom = int(vp_x + dx * width * 2.4)
            draw.line([(vp_x, horizon), (x_bottom, height)], fill=sheen + (56,), width=2)
        spread = 1.0
        while True:
            frac = spread / (spread + 1.0)
            y = int(horizon + (floor - horizon) * frac)
            if y >= floor:
                break
            x0 = int(vp_x - width * 0.5 * frac)
            x1 = int(vp_x + width * 0.5 * frac)
            draw.line([(x0, y), (x1, y)], fill=sheen + (44,), width=2)
            spread *= 1.55
    return Image.alpha_composite(layer, out)


def _wall(layer: Image.Image, color: tuple, panel_color: tuple, horizon: int, floor: int, rng) -> Image.Image:
    width, height = layer.size
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, horizon, width, floor), fill=color + (225,))
    rail = floor - int((floor - horizon) * 0.25)
    draw.line([(0, rail), (width, rail)], fill=panel_color + (110,), width=max(int(width * 0.003), 2))
    x = int(width * rng.uniform(0.02, 0.10))
    while x < width:
        pw = int(width * rng.uniform(0.08, 0.14))
        draw.rectangle((x, rail, x + pw, floor - 3), outline=panel_color + (90,), width=max(int(width * 0.002), 2))
        x += pw + int(width * 0.03)
    return Image.alpha_composite(layer, out)


def _silhouette(
    draw,
    cx: int,
    floor: int,
    stand_h: int,
    width: int,
    color,
    rim_key=None,
    rim_fill=None,
    facing: int = 1,
):
    head_r = max(int(stand_h * 0.16), 6)
    head_y = floor - stand_h + head_r
    body_top = head_y + head_r
    shoulder = head_r * 2.5
    hip = head_r * 2.0
    draw.ellipse(
        (cx - int(width * 0.07), floor - int(stand_h * 0.03), cx + int(width * 0.07), floor + int(stand_h * 0.25)),
        fill=(5, 5, 9, 160),
    )
    draw.polygon(
        [
            (cx - hip, floor),
            (cx - head_r * 0.25, floor - int(stand_h * 0.42)),
            (cx + head_r * 0.25, floor - int(stand_h * 0.42)),
            (cx + hip, floor),
        ],
        fill=color,
    )
    draw.polygon(
        [
            (cx - shoulder, body_top + int(stand_h * 0.30)),
            (cx - head_r * 1.0, body_top),
            (cx + head_r * 1.0, body_top),
            (cx + shoulder, body_top + int(stand_h * 0.30)),
        ],
        fill=color,
    )
    draw.rectangle((cx - hip * 0.25, body_top + int(stand_h * 0.25), cx + hip * 0.25, floor), fill=color)
    draw.ellipse(
        (cx - head_r, head_y - head_r, cx + head_r, head_y + head_r),
        fill=color,
    )
    neck_w = max(int(head_r * 0.5), 2)
    draw.rectangle((cx - neck_w, body_top - head_r, cx + neck_w, body_top), fill=color)
    if rim_key:
        side = facing
        draw.line(
            [
                (cx + side * shoulder * 0.92, body_top + int(stand_h * 0.30)),
                (cx + side * head_r * 0.25, floor - int(stand_h * 0.25)),
            ],
            fill=rim_key,
            width=max(int(width * 0.004), 3),
        )
        draw.arc(
            (cx - head_r - 1, head_y - head_r - 1, cx + head_r + 1, head_y + head_r + 1),
            start=300 if side > 0 else 210,
            end=80 if side > 0 else 330,
            fill=rim_key,
            width=max(int(width * 0.0035), 3),
        )
    if rim_fill:
        back = -facing
        draw.line(
            [
                (cx + back * shoulder * 0.88, body_top + int(stand_h * 0.28)),
                (cx + back * head_r * 0.25, floor - int(stand_h * 0.25)),
            ],
            fill=rim_fill,
            width=max(int(width * 0.0022), 2),
        )


def _bokeh(image: Image.Image, colors: list, rng, count: int = 20) -> Image.Image:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    radius_ref = min(width, height)
    for _ in range(count):
        radius = rng.randint(radius_ref // 110, radius_ref // 26)
        x = rng.randint(0, max(width - 2, 1))
        y = rng.randint(0, max(int(height * 0.8) - 2, 1))
        alpha = rng.randint(20, 90)
        color = colors[rng.randrange(len(colors))]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color + (alpha,))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(radius_ref // 130, 1)))
    return Image.alpha_composite(image, layer)


def _light_shaft(image: Image.Image, accent: tuple, glow_col: tuple, rng, strength: int = 52) -> Image.Image:
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for color, s2 in ((accent, strength), (glow_col, max(strength // 2, 14))):
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
            fill=color + (s2,),
        )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=int(width * 0.03)))
    return Image.alpha_composite(image, layer)


def _grade(image: Image.Image, palette: dict) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    shadow_t = np.array(palette.get("wash", (80, 80, 160)), dtype=np.float32) / 255.0
    high_t = np.array(palette.get("glow", (255, 170, 90)), dtype=np.float32) / 255.0
    w_shadow = np.clip(1.0 - lum * 1.7, 0, 1)[..., None] * 0.25
    w_high = np.clip((lum - 0.55) * 2.4, 0, 1)[..., None] * 0.25
    arr = arr * (1 - w_shadow - w_high) + shadow_t * w_shadow + high_t * w_high
    arr = np.clip(arr + 0.22 * (arr - 0.5) * (1 - np.abs(2 * arr - 1)), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB").convert("RGBA")


def _bloom(image: Image.Image, threshold: int = 185) -> Image.Image:
    width, _ = image.size
    rgb = image.convert("RGB")
    highlights = rgb.point(lambda v: max(0, v - threshold) * 255 // max(255 - threshold, 1))
    highlights = highlights.filter(ImageFilter.GaussianBlur(radius=max(width // 40, 5)))
    r, g, b = highlights.split()
    alpha = ImageChops.lighter(ImageChops.lighter(r, g), b).point(lambda v: v // 2)
    return ImageChops.screen(image, Image.merge("RGBA", (r, g, b, alpha)))


def _chromatic(image: Image.Image) -> Image.Image:
    width, _ = image.size
    off = max(int(width * 0.0016), 1)
    r, g, b, a = image.split()
    r = ImageChops.offset(r, off, 0)
    b = ImageChops.offset(b, -off, 0)
    return Image.merge("RGBA", (r, g, b, a))


def _grain(image: Image.Image, amount: float = 7.0) -> Image.Image:
    array = np.asarray(image).astype(np.int16)
    noise = np.random.normal(0, amount, array.shape[:2])
    array[..., :3] += noise[..., None].astype(np.int16)
    array[..., 3] = 255
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGBA")


def _vignette(image: Image.Image, strength: int = 175) -> Image.Image:
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
    return Image.alpha_composite(image, dark)


def _motif_scene(image: Image.Image, motif: str, palette: dict, rng) -> Image.Image:
    width, height = image.size
    accent = palette["accent"]
    glow = palette.get("glow", accent)
    wash = palette.get("wash", (120, 120, 200))
    fog_c = palette.get("fog", (90, 90, 130))
    shadow = palette.get("shadow", (6, 6, 12))
    sheen = _mix(wash, (205, 215, 240), 0.35)
    cool = (140, 220, 255)
    horizon = int(height * 0.36)
    floor = int(height * 0.90)
    scene = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    if motif == "door":
        scene = _wall(scene, fog_c, (120, 120, 150), horizon, floor, rng)
        scene = _floor_scene(scene, tuple(int(c * 0.25) for c in shadow), sheen, horizon, floor, width // 2)
        enter_w = int(width * 0.30)
        enter_h = int(height * 0.30)
        x0 = (width - enter_w) // 2
        y0 = floor - enter_h
        spill = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(spill)
        sdraw.polygon(
            [
                (x0 + 6, floor),
                (x0 + enter_w - 6, floor),
                (x0 + enter_w + int(width * 0.10), height),
                (x0 - int(width * 0.10), height),
            ],
            fill=accent + (54,),
        )
        spill = spill.filter(ImageFilter.GaussianBlur(radius=int(width * 0.025)))
        scene = Image.alpha_composite(scene, spill)
        draw = ImageDraw.Draw(scene)
        draw.rounded_rectangle(
            (x0, y0, x0 + enter_w, floor),
            radius=int(enter_w * 0.05),
            fill=(16, 17, 26, 252),
            outline=(180, 182, 195, 150),
            width=max(int(width * 0.003), 2),
        )
        for k in (0.18, 0.62):
            py = y0 + int(enter_h * k)
            draw.rounded_rectangle(
                (x0 + int(enter_w * 0.12), py, x0 + enter_w - int(enter_w * 0.12), py + int(enter_h * 0.30)),
                radius=int(enter_w * 0.03),
                outline=(150, 152, 170, 110),
                width=max(int(width * 0.0022), 2),
            )
        knob_y = y0 + int(enter_h * 0.52)
        kn_x = x0 + enter_w - int(enter_w * 0.16)
        draw.ellipse((kn_x - 5, knob_y - 5, kn_x + 7, knob_y + 7), fill=accent + (235,))
        draw.ellipse((width // 2 - 3, y0 + int(enter_h * 0.25) - 3, width // 2 + 3, y0 + int(enter_h * 0.25) + 3), fill=(150, 220, 255, 220))
        gap_h = max(int(height * 0.005), 3)
        draw.rectangle((x0 + 4, floor - gap_h, x0 + enter_w - 4, floor + 2), fill=accent + (215,))
        scene = _light_pool(scene, accent, width // 2, floor + int(height * 0.02), int(width * 0.25), int(height * 0.012), 60)
        draw = ImageDraw.Draw(scene)

    elif motif == "corridor":
        scene = _floor_scene(scene, tuple(int(c * 0.30) for c in shadow), sheen, horizon, floor, width // 2)
        draw = ImageDraw.Draw(scene)
        cx = width // 2
        frames = 6
        for i in range(frames, 0, -1):
            scale = i / frames
            shrink = scale * 0.5
            fw = int(width * shrink)
            fh = int((floor - horizon) * shrink * 0.75)
            x0 = cx - fw // 2
            y1 = floor
            draw.rectangle(
                (x0, y1 - fh, x0 + fw, y1),
                outline=(150, 152, 170, 200 if scale > 0.5 else 140),
                width=max(int(width * 0.0022), 2),
            )
            lamp_col = accent if i % 2 == 0 else cool
            lamp_r = max(int(width * 0.014 * scale), 3)
            lamp_y = y1 - fh + int(fh * 0.18)
            draw.ellipse((cx - lamp_r, lamp_y - lamp_r, cx + lamp_r, lamp_y + lamp_r), fill=lamp_col + (225,))
            if scale > 0.45:
                door_side = 1 if i % 2 == 0 else -1
                dw = int(fw * 0.25)
                dh = int(fh * 0.62)
                dx = x0 + fw if door_side > 0 else x0 - dw
                dy = y1 - dh
                fill_c = accent + (46,) if i % 3 == 0 else (12, 13, 22, 200)
                draw.rectangle((dx, dy, dx + dw, y1), fill=fill_c, outline=(175, 178, 195, 210))
        _silhouette(draw, cx - int(width * rng.uniform(-0.03, 0.03)), floor - int(height * 0.005), int(height * 0.25), width, (5, 5, 9, 250), rim_key=accent + (170,))

    elif motif == "figure":
        scene = _wall(scene, fog_c, (120, 120, 150), horizon, floor, rng)
        scene = _floor_scene(scene, tuple(int(c * 0.25) for c in shadow), sheen, horizon, floor, width // 2, lines=False)
        lamp_x = int(width * 0.25)
        lamp_top = horizon + int((floor - horizon) * 0.05)
        scene = _glow(scene, glow, (lamp_x, lamp_top + int(height * 0.03)), int(width * 0.25), 120)
        scene = _light_pool(scene, accent, lamp_x, floor, int(width * 0.30), int(height * 0.014), 65)
        draw = ImageDraw.Draw(scene)
        stem_x = max(int(width * 0.004), 2)
        draw.rectangle((lamp_x - stem_x, lamp_top + int(height * 0.05), lamp_x + stem_x, floor), fill=(10, 10, 16, 245))
        draw.polygon(
            [
                (lamp_x - int(width * 0.055), lamp_top + int(height * 0.05)),
                (lamp_x + int(width * 0.055), lamp_top + int(height * 0.05)),
                (lamp_x + int(width * 0.032), lamp_top),
                (lamp_x - int(width * 0.032), lamp_top),
            ],
            fill=(16, 16, 24, 250),
            outline=accent + (160,),
        )
        cx = int(width * 0.68)
        _silhouette(draw, cx, floor, int(height * 0.25), width, (5, 5, 9, 250), rim_key=glow + (190,), facing=-1)

    elif motif == "figures":
        scene = _wall(scene, fog_c, (120, 120, 150), horizon, floor, rng)
        scene = _floor_scene(scene, tuple(int(c * 0.25) for c in shadow), sheen, horizon, floor, width // 2, lines=False)
        mid_x = width // 2
        scene = _glow(scene, glow, (mid_x, horizon + int((floor - horizon) * 0.25)), int(width * 0.30), 110)
        scene = _light_pool(scene, accent, mid_x, floor, int(width * 0.35), int(height * 0.014), 60)
        draw = ImageDraw.Draw(scene)
        _silhouette(draw, int(width * 0.25), floor, int(height * 0.25), width, (5, 5, 9, 250), rim_key=glow + (190,), facing=1)
        _silhouette(draw, int(width * 0.75), floor, int(height * 0.25), width, (7, 7, 12, 248), rim_key=accent + (150,), facing=-1)

    elif motif == "eyes":
        cx = width // 2 + rng.randint(-int(width * 0.05), int(width * 0.05))
        eye_y = int(height * 0.44)
        face_w = int(width * 0.30)
        face_h = int(face_w * 1.35)
        scene = _glow(scene, (78, 58, 52), (cx, eye_y - int(face_h * 0.25)), int(face_w * 0.75), 60)
        dark_top = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        top_draw = ImageDraw.Draw(dark_top)
        top_draw.rectangle((0, 0, width, eye_y - int(face_h * 0.25)), fill=shadow + (135,))
        scene = Image.alpha_composite(scene, dark_top)
        draw = ImageDraw.Draw(scene)
        spread = int(width * 0.075)
        eye_w = int(width * 0.052)
        eye_h = int(height * 0.011)
        for dx in (-spread, spread):
            ex = cx + dx
            draw.polygon(
                [
                    (ex - eye_w, eye_y),
                    (ex - eye_w * 0.4, eye_y - eye_h),
                    (ex + eye_w * 0.4, eye_y - eye_h),
                    (ex + eye_w, eye_y),
                    (ex + eye_w * 0.25, eye_y + eye_h * 1.5),
                    (ex - eye_w * 0.25, eye_y + eye_h * 1.5),
                ],
                fill=(235, 232, 228, 235),
            )
            iris_r = int(eye_h * 2.1)
            draw.ellipse((ex - iris_r, eye_y - iris_r, ex + iris_r, eye_y + iris_r), fill=accent + (255,))
            draw.ellipse((ex - iris_r // 2, eye_y - iris_r // 2, ex + iris_r // 2, eye_y + iris_r // 2), fill=(3, 3, 5, 255))
            draw.ellipse((ex - iris_r + 1, eye_y - iris_r + 1, ex - iris_r // 2, eye_y - iris_r // 2), fill=(250, 250, 245, 230))
        scene = _glow(scene, accent, (cx, eye_y), int(face_w * 0.25), 54)
        draw = ImageDraw.Draw(scene)

    elif motif == "window":
        win_w = int(width * 0.38)
        win_h = int(height * 0.34)
        x0 = (width - win_w) // 2
        y0 = int(height * 0.25)
        inside = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        idraw = ImageDraw.Draw(inside)
        idraw.rectangle((x0, y0, x0 + win_w, y0 + win_h), fill=_mix(glow, (200, 215, 240), 0.55) + (105,))
        moon_l = _moon((win_w, win_h), 0.72, 0.30, _mix(glow, (235, 235, 215), 0.6))
        inside.paste(moon_l, (x0, y0), moon_l)
        for _ in range(14):
            lx = rng.randint(x0 + 4, x0 + win_w - 8)
            ly = rng.randint(y0 + int(win_h * 0.25), y0 + win_h - 8)
            lc = rng.choice([(255, 190, 110), (120, 210, 255), (255, 220, 160)])
            idraw.rectangle((lx, ly, lx + max(int(win_w * 0.012), 2), ly + max(int(win_h * 0.05), 3)), fill=lc + (rng.randint(120, 210),))
        fog_w = _fog((win_w, win_h), fog_c, rng, strength=110, center=0.72, spread=0.20)
        inside.paste(fog_w, (0, 0), fog_w)
        scene = Image.alpha_composite(scene, inside)
        draw = ImageDraw.Draw(scene)
        bar = max(int(width * 0.007), 3)
        dark_s = shadow + (250,)
        draw.rectangle((x0, y0 + win_h // 2 - bar, x0 + win_w, y0 + win_h // 2 + bar), fill=dark_s)
        draw.rectangle((x0 + win_w // 2 - bar, y0, x0 + win_w // 2 + bar, y0 + win_h), fill=dark_s)
        draw.rectangle((x0 - bar, y0 - bar, x0 + win_w + bar, y0 + win_h + bar), outline=dark_s, width=bar * 2)
        for ccx in (x0 - int(width * 0.055), x0 + win_w + int(width * 0.055)):
            cw = int(width * 0.05)
            draw.rounded_rectangle((ccx - cw // 2, y0, ccx + cw // 2, y0 + win_h + int(height * 0.03)), radius=cw // 3, fill=(8, 8, 14, 240))
            for fy in range(4):
                fy_y = y0 + int(win_h * (fy + 0.5) / 4)
                draw.line([(ccx - cw // 2 + 2, fy_y), (ccx + cw // 2 - 2, fy_y)], fill=(20, 20, 28, 180), width=max(int(width * 0.002), 2))
        streaks = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(streaks)
        for _ in range(3):
            sx = rng.randint(x0, max(x0 + win_w - int(win_w * 0.13), x0 + 2))
            sdraw.polygon(
                [
                    (sx, y0),
                    (sx + int(win_w * 0.05), y0),
                    (sx + int(win_w * 0.13), y0 + win_h),
                    (sx + int(win_w * 0.08), y0 + win_h),
                ],
                fill=(230, 240, 255, 24),
            )
        streaks = streaks.filter(ImageFilter.GaussianBlur(2))
        scene = Image.alpha_composite(scene, streaks)
        draw = ImageDraw.Draw(scene)
        _silhouette(draw, x0 + int(win_w * 0.25), y0 + win_h + int(height * 0.004), int(height * 0.13), int(width * 0.25), (5, 5, 9, 255), rim_key=glow + (170,), facing=-1)

    elif motif == "phone":
        scene = Image.new("RGBA", (width, height), (0, 0, 0, 250))
        scene = _glow(scene, glow, (width // 2, int(height * 0.25)), int(width * 0.30), 110)
        draw = ImageDraw.Draw(scene)
        phone_w = int(width * 0.25)
        phone_h = int(phone_w * 2.0)
        x0 = (width - phone_w) // 2
        y0 = int(height * 0.30)
        draw.rounded_rectangle(
            (x0, y0, x0 + phone_w, y0 + phone_h),
            radius=int(width * 0.035),
            fill=(10, 12, 18, 252),
            outline=(185, 188, 200, 190),
            width=max(int(width * 0.0028), 2),
        )
        inner = int(width * 0.013)
        draw.rounded_rectangle(
            (x0 + inner, y0 + inner, x0 + phone_w - inner, y0 + phone_h - inner),
            radius=int(width * 0.022),
            fill=(225, 230, 240, 245),
        )
        pad = int(phone_w * 0.25)
        sb_x0 = x0 + inner
        sb_x1 = x0 + phone_w - inner
        bubble_w = phone_w - inner * 2 - pad
        line_h = max(int(phone_h * 0.024), 4)
        by = y0 + inner + int(phone_h * 0.03)
        msg_pair = [(0, 3), (1, 2), (0, 2), (1, 3), (0, 2)]
        for side, n_lines in msg_pair:
            bh = line_h * (n_lines + 1) + int(line_h * 0.6)
            if side == 0:
                bx0 = sb_x0 + pad // 2
                bx1 = bx0 + bubble_w
                fill_c = (30, 34, 44, 235)
                line_c = (235, 238, 245, 160)
            else:
                bx1 = sb_x1 - pad // 2
                bx0 = bx1 - bubble_w
                fill_c = accent + (235,)
                line_c = (30, 24, 12, 190)
            draw.rounded_rectangle((bx0, by, bx1, by + bh), radius=line_h, fill=fill_c)
            for li in range(n_lines):
                lw = int(bubble_w * rng.uniform(0.5, 0.85))
                ly = by + int(line_h * 0.25) + li * int(line_h * 1.35)
                draw.rounded_rectangle(
                    (bx0 + int(line_h * 0.25), ly, bx0 + int(line_h * 0.25) + lw, ly + int(line_h * 0.25) + line_h // 2),
                    radius=line_h // 4,
                    fill=line_c,
                )
            by += bh + int(line_h * 0.75)
        ty = min(by + int(line_h * 0.25), y0 + phone_h - inner - line_h * 2)
        tx = sb_x0 + pad // 2
        for di in range(3):
            r = max(line_h // 3, 2)
            dxx = tx + di * (r * 3)
            draw.ellipse((dxx - r, ty - r, dxx + r, ty + r), fill=(40, 44, 56, 220))
        draw.polygon(
            [
                (x0 + int(phone_w * 0.30), height),
                (x0 + int(phone_w * 0.75), height),
                (x0 + int(phone_w * 0.68), y0 + phone_h - int(phone_h * 0.16)),
                (x0 + int(phone_w * 0.30), y0 + phone_h - int(phone_h * 0.12)),
            ],
            fill=(6, 6, 10, 252),
        )
        draw.ellipse(
            (x0 + int(phone_w * 0.25), y0 + phone_h - int(phone_h * 0.13), x0 + int(phone_w * 0.55), y0 + phone_h - int(phone_h * 0.05)),
            fill=(8, 8, 13, 250),
        )
        for _ in range(3):
            gxx = rng.randint(0, max(width - 2, 1))
            gyy = rng.randint(max(y0 - int(height * 0.05), 0), y0 + int(height * 0.25))
            gr = rng.randint(1, 2)
            draw.ellipse((gxx - gr, gyy - gr, gxx + gr, gyy + gr), fill=accent + (rng.randint(120, 210),))

    elif motif == "house":
        scene = Image.alpha_composite(scene, _stars((width, height), rng))
        scene = Image.alpha_composite(scene, _moon((width, height), 0.70, 0.14))
        scene = Image.alpha_composite(scene, _fog((width, height), fog_c, rng, strength=40, center=0.26, spread=0.28))
        draw = ImageDraw.Draw(scene)
        ground_y = int(height * 0.76)
        house_w = int(width * 0.30)
        house_h = int(height * 0.20)
        hx = int(width * 0.25)
        hy = ground_y - house_h
        near = (7, 7, 12, 255)
        draw.rectangle((hx + int(house_w * 7 / 10), hy - int(house_h * 1 / 2), hx + int(house_w * 8 / 10), hy + int(house_h * 1 / 4)), fill=(9, 9, 14, 250))
        draw.rectangle((hx, hy, hx + house_w, ground_y), fill=near)
        draw.polygon(
            [
                (hx - int(house_w * 0.25), hy),
                (hx + house_w + int(house_w * 0.25), hy),
                (hx + house_w // 2, hy - int(house_h * 0.55)),
            ],
            fill=near,
        )
        draw.line([(hx - int(house_w * 0.25), hy), (hx + house_w // 2, hy - int(house_h * 0.55)), (hx + house_w + int(house_w * 0.25), hy)], fill=(185, 195, 220, 140), width=3)
        for si in range(5):
            sy = hy - int(house_h * 0.25) - si * max(int(height * 0.012), 3)
            sw = max(int(house_w * (0.16 - si * 0.02)), 2)
            sx = hx + int(house_w * 0.76)
            draw.ellipse((sx - sw // 2, sy - sw // 4, sx + sw, sy + sw // 2), fill=fog_c + (max(120 - si * 22, 10),))
        for wxf, lit in ((0.16, True), (0.52, True), (0.80, False)):
            wxx = hx + int(house_w * wxf)
            wyy = hy + int(house_h * 0.30)
            ww = int(house_w * 0.25)
            wh = int(house_h * 0.30)
            if lit:
                scene = _light_pool(scene, glow, wxx + ww // 2, wyy + wh // 2, ww * 3, wh * 2, 70)
                draw = ImageDraw.Draw(scene)
                draw.rectangle((wxx, wyy, wxx + ww, wyy + wh), fill=_mix(glow, (255, 230, 180), 0.35) + (238,))
            else:
                draw.rectangle((wxx, wyy, wxx + ww, wyy + wh), fill=(12, 13, 20, 250), outline=(90, 92, 110, 130))
        draw.rectangle((hx + int(house_w * 0.36), ground_y - int(house_h * 0.42), hx + int(house_w * 0.25) + int(house_w * 0.52) - int(house_w * 0.25), ground_y), fill=(10, 10, 16, 252))
        scene = _light_pool(scene, accent, hx + int(house_w * 0.44), ground_y + int(height * 0.004), int(width * 0.25), int(height * 0.010), 55)
        draw = ImageDraw.Draw(scene)
        ground = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(ground)
        gdraw.rectangle((0, ground_y, width, height), fill=(4, 4, 8, 245))
        scene = Image.alpha_composite(scene, ground)
        draw = ImageDraw.Draw(scene)
        fence_y = ground_y + int((height - ground_y) * 0.55)
        for fx in range(0, width, max(int(width * 0.09), 12)):
            fh2 = max(int(height * rng.uniform(0.02, 0.04)), 5)
            draw.rectangle((fx, fence_y - fh2, fx + max(int(width * 0.006), 2), fence_y), fill=(9, 9, 14, 245))
        draw.rectangle((0, fence_y - 2, width, fence_y), fill=(9, 9, 14, 245))
        for txx in (int(width * 0.08), int(width * 0.13), int(width * 0.93)):
            tr = max(int(width * rng.uniform(0.05, 0.09)), 8)
            draw.ellipse((txx - tr, ground_y - int(tr * 1.5), txx + tr, ground_y + int(tr * 0.25)), fill=(6, 7, 10, 250))
        scene = Image.alpha_composite(scene, _fog((width, height), fog_c, rng, strength=90, center=0.88, spread=0.10))
        draw = ImageDraw.Draw(scene)

    elif motif == "stairs":
        scene = _floor_scene(scene, tuple(int(c * 0.30) for c in shadow), sheen, int(height * 0.25), floor, width // 2, lines=False)
        draw = ImageDraw.Draw(scene)
        steps = 9
        step_h = max(int((floor - int(height * 0.25)) / steps), 6)
        top_y = floor - steps * step_h
        for i in range(steps):
            y1 = floor - i * step_h
            inset = int(width * 0.25 * (i / steps) * 1.0)
            draw.rectangle((inset, y1 - int(step_h * 0.25), width - inset, y1), fill=(10, 11, 18, 250))
            draw.rectangle(
                (inset, y1 - step_h, width - inset, y1 - int(step_h * 0.25)),
                fill=(18, 20, 30, 250) if i % 2 == 0 else (14, 15, 24, 250),
            )
            draw.line(
                [(inset, y1 - int(step_h * 0.25)), (width - inset, y1 - int(step_h * 0.25))],
                fill=_mix(glow, (200, 210, 230), 0.4) + (110,),
                width=max(int(width * 0.0022), 2),
            )
        rail_x0 = int(width * 0.30)
        rail_x1 = int(width * 0.70)
        draw.line([(rail_x0, floor - step_h), (rail_x1, max(top_y - step_h, 4))], fill=(150, 152, 170, 200), width=max(int(width * 0.004), 3))
        draw.line([(int(width * 0.25), floor), (int(width * 0.25), max(top_y - int(step_h * 0.5), 4))], fill=(60, 62, 80, 200), width=max(int(width * 0.003), 2))
        scene = _glow(scene, glow, (width // 2, max(top_y - int(height * 0.05), 10)), int(width * 0.30), 120)
        draw = ImageDraw.Draw(scene)
        _silhouette(draw, int(width * 0.25), floor, int(height * 0.23), width, (5, 5, 9, 255), rim_key=glow + (190,), facing=1)

    elif motif == "city":
        scene = Image.alpha_composite(scene, _stars((width, height), rng, count=70))
        scene = Image.alpha_composite(scene, _moon((width, height), 0.26, 0.12))
        draw = ImageDraw.Draw(scene)
        ground_y = int(height * 0.86)
        depth_layers = [
            (_mix(fog_c, (160, 175, 205), 0.65) + (226,), 0.55),
            (_mix(fog_c, (70, 80, 115), 0.75) + (240,), 0.75),
            ((7, 8, 14, 255), 0.30),
        ]
        for li, (build_col, lit_p) in enumerate(depth_layers):
            lift = 1.0 + (2 - li) * 0.18
            x = -int(width * 0.05)
            while x < width:
                bw = rng.randint(int(width * 0.06), int(width * 0.13))
                bh = int(height * rng.uniform(0.14, 0.34) * lift)
                by0 = ground_y - bh
                draw.rectangle((x, by0, x + bw, ground_y), fill=build_col)
                draw.line([(x, by0), (x + bw, by0)], fill=(175, 190, 225, 120), width=2)
                wy = by0 + int(height * 0.015)
                while wy < ground_y - int(height * 0.02):
                    wx = x + int(width * 0.012)
                    while wx < x + bw - int(width * 0.012):
                        if rng.random() < lit_p * 0.5:
                            lcol = glow if rng.random() < 0.5 else cool if rng.random() < 0.55 else accent
                            draw.rectangle(
                                (wx, wy, wx + max(int(width * 0.004), 2), wy + max(int(height * 0.010), 3)),
                                fill=lcol + (rng.randint(120, 225),),
                            )
                        wx += int(width * 0.023)
                    wy += int(height * 0.032)
                if li == 2 and rng.random() < 0.4:
                    ax = x + bw // 2
                    draw.line([(ax, by0), (ax, max(by0 - int(height * 0.03), 2))], fill=(7, 8, 14, 255), width=2)
                    draw.ellipse((ax - 2, by0 - int(height * 0.03) - 2, ax + 2, by0 - int(height * 0.03) + 2), fill=(255, 60, 60, 210))
                x += bw + int(width * 0.010)
        scene = Image.alpha_composite(scene, _fog((width, height), fog_c, rng, strength=120, center=0.80, spread=0.16))
        scene = _light_pool(scene, glow, width // 2, ground_y, int(width * 0.25), int(height * 0.012), 70)
        draw = ImageDraw.Draw(scene)
        for _ in range(5):
            rx = rng.randint(0, max(width - 2, 1))
            ry = rng.randint(int(height * 1 / 4), int(height * 3 / 4))
            rl = rng.randint(int(height * 1 / 4), int(height * 3 / 5))
            draw.line([(rx, ry), (rx - int(rl * 0.25), ry + rl)], fill=(220, 230, 245, 26), width=2)

    if motif not in ("city", "house", "eyes"):
        wash_img = _fog((width, height), fog_c, rng, strength=16, center=0.45, spread=0.38)
        scene = Image.alpha_composite(scene, wash_img)
    return scene


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
    rng = __import__("random").Random(f"v4-{index}-{style}")
    palette = PALETTES.get(style, PALETTES["dark_cinematic"])
    motif = pick_motif(text, index, rng)

    image = _gradient(size, palette["stops"]).convert("RGBA")
    gx = int(size[0] * rng.uniform(0.3, 0.7))
    gy = int(size[1] * rng.uniform(0.26, 0.44))
    image = _glow(image, palette.get("glow", palette["accent"]), (gx, gy), int(min(size) * 0.25), 115)
    image = _glow(image, palette.get("wash", palette["glow"]), (int(size[0] * 0.25 + gx * 0.5) % size[0], int(size[1] * 0.25)), int(min(size) * 0.25), 75)

    scene = _motif_scene(image, motif, palette, rng)
    image = Image.alpha_composite(image, scene)

    if motif not in ("city", "house", "eyes", "phone"):
        image = _light_shaft(image, palette["accent"], palette.get("glow", palette["accent"]), rng)

    image = _bokeh(image, [palette["accent"], palette.get("glow", palette["accent"]), palette.get("wash", (140, 180, 255))], rng)
    image = ImageEnhance.Color(image).enhance(1.22)
    image = _grade(image, palette)
    image = _bloom(image, threshold=180)
    image = _chromatic(image)
    image = _grain(image, amount=palette.get("grain", 7.0))
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
